"""The 'Explore results' window of the GUI: look at the inputs and outputs, and act on them.

Four tabs, all reading the experiment's files (nothing here runs VAME itself):

- Recordings          the input videos, with each frame's motif and community;
                      jump from one bout of a motif or community to the next.
- Motifs & communities  VAME's motif tree and the UMAP of the latent space. Pick
                      the integer at which the tree is cut, see the communities it
                      makes, then apply it (runs 'communities' with that cut).
- Communities & clips   the communities and their motifs, each motif's clip from
                      'videos'; name the communities and save them as
                      ``*_labeled.csv`` (communities.py).
- GIF                 vame.gif: make one (runs 'gif') and play it.

Commands are run through the main window (App.run_command), so they behave as
the buttons there: same run records, and on the GPU server when that is ticked.

Videos are read with OpenCV and maps drawn with matplotlib, both installed
with VAME.
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING

import cv2
import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PIL import Image, ImageTk

from vame_motifs import communities
from vame_motifs.config import ExperimentConfig, load_config
from vame_motifs.export import ExportError, labels_to_frames
from vame_motifs.io import count_frames
from vame_motifs.pipeline import VamePipeline

if TYPE_CHECKING:
    from vame_motifs.gui import App

# tab20's strong colours first, then the light ones: neighbouring numbers look different
PALETTE = [colormaps["tab20"](i) for i in [*range(0, 20, 2), *range(1, 20, 2)]]
NO_MOTIF = -1


def colour(i: int):
    return PALETTE[int(i) % len(PALETTE)]


def bouts(mask: np.ndarray) -> np.ndarray:
    """(first, last) frame of each run of True values, in order."""
    edges = np.diff(np.concatenate([[0], np.asarray(mask, dtype=int), [0]]))
    return np.column_stack([np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1])


# ======================================================================
# The experiment's results, as the window needs them (no Tk here)
# ======================================================================
class Results:
    """What the analysis produced so far, for one segmentation algorithm. Missing parts are None."""

    def __init__(self, cfg: ExperimentConfig, algorithm: str):
        self.cfg, self.algorithm = cfg, algorithm
        self.pipeline = VamePipeline(cfg)
        self.sessions = [p.stem for p in cfg.pose_files]
        self.pose_files = dict(zip(self.sessions, cfg.pose_files))
        self.videos: dict[str, Path] = dict(zip(self.sessions, cfg.videos))
        self._frame_motifs: dict[str, np.ndarray] = {}
        self.tree = self._load_tree()
        self.bag = self._load_bag()
        self.umap = dict(np.load(self.pipeline.umap_path)) if self.pipeline.umap_path.exists() else None
        self.point_motif = self._point_motifs()

    # --- motifs ---------------------------------------------------------
    def window_labels(self, session: str) -> np.ndarray | None:
        path = self.pipeline.label_file(session, self.algorithm)
        return np.load(path) if path.exists() else None

    def frame_motifs(self, session: str) -> np.ndarray | None:
        """Motif of every frame of the recording (NO_MOTIF at the window edges), as in the motif CSV."""
        if session not in self._frame_motifs:
            labels = self.window_labels(session)
            if labels is None:
                return None
            n_frames = count_frames(self.pose_files[session])
            per_frame = labels_to_frames(labels, n_frames, self.cfg.time_window)
            self._frame_motifs[session] = per_frame.fillna(NO_MOTIF).to_numpy(dtype=int)
        return self._frame_motifs[session]

    def usage(self) -> np.ndarray:
        """Share of all labelled time windows in each motif, over all recordings."""
        counts = np.zeros(self.cfg.n_clusters)
        for session in self.sessions:
            labels = self.window_labels(session)
            if labels is not None:
                counts += np.bincount(labels.astype(int), minlength=self.cfg.n_clusters)[: self.cfg.n_clusters]
        return counts / counts.sum() if counts.sum() else counts

    # --- communities ----------------------------------------------------
    def _load_tree(self):
        path = self.pipeline.community_bag_file(self.algorithm).parent / "tree.graphml"
        return communities.load_tree(path) if path.exists() else None

    def _load_bag(self) -> list[list[int]] | None:
        """The communities VAME saved with the last 'communities' run."""
        path = self.pipeline.community_bag_file(self.algorithm)
        if not path.exists():
            return None
        return [[int(m) for m in motifs] for motifs in np.load(path, allow_pickle=True)]

    def community_of(self) -> dict[int, int]:
        return communities.motif_to_community(self.bag) if self.bag else {}

    def names(self) -> dict[int, str]:
        return communities.read_labels(self.cfg, self.algorithm, self.bag) if self.bag else {}

    # --- map and clips --------------------------------------------------
    def _point_motifs(self) -> np.ndarray | None:
        """Motif of every point of the UMAP (NO_MOTIF if its recording has no labels)."""
        if self.umap is None:
            return None
        motifs = np.full(len(self.umap["window"]), NO_MOTIF)
        for i, session in enumerate(self.umap["sessions"]):
            labels = self.window_labels(str(session))
            points = self.umap["session"] == i
            if labels is not None:
                windows = self.umap["window"][points]
                ok = windows < len(labels)
                motifs[np.flatnonzero(points)[ok]] = labels[windows[ok]]
        return motifs

    def clips(self, session: str) -> dict[int, Path]:
        return self.pipeline.motif_clips(session, self.algorithm)

    def gifs(self) -> list[Path]:
        return sorted((self.cfg.output / "gifs").glob("*.gif"), key=lambda p: p.stat().st_mtime, reverse=True)


# ======================================================================
# Video player
# ======================================================================
class VideoPlayer(ttk.Frame):
    """Plays a video or GIF in the window: play/pause, frame steps, a position slider, speed.

    ``describe(frame)`` gives the text shown under the picture (e.g. the frame's motif).
    """

    SPEEDS = ("0.25", "0.5", "1", "2", "4")

    def __init__(self, parent, width: int = 520, height: int = 400):
        super().__init__(parent)
        self.capture: cv2.VideoCapture | None = None
        self.n_frames, self.fps, self.frame = 0, 30.0, 0
        self.playing, self._job, self._photo, self._image = False, None, None, None
        self._sliding = False
        self.describe: Callable[[int], str] | None = None
        self.on_frame: Callable[[int], None] | None = None

        self.screen = tk.Canvas(self, width=width, height=height, background="black", highlightthickness=0)
        self.screen.grid(row=0, column=0, columnspan=7, sticky="nsew")
        self.screen.bind("<Configure>", lambda _: self._draw())
        self.message = "No video"

        self.position = tk.DoubleVar()
        self.slider = ttk.Scale(self, from_=0, to=1, variable=self.position, command=self._slide)
        self.slider.grid(row=1, column=0, columnspan=7, sticky="ew", pady=(4, 0))
        self.play_button = ttk.Button(self, text="▶ Play", width=8, command=self.toggle)
        self.play_button.grid(row=2, column=0, pady=2)
        ttk.Button(self, text="◀ 1", width=4, command=lambda: self.step(-1)).grid(row=2, column=1)
        ttk.Button(self, text="1 ▶", width=4, command=lambda: self.step(1)).grid(row=2, column=2)
        ttk.Label(self, text="Speed").grid(row=2, column=3, padx=(10, 2))
        self.speed = tk.StringVar(value="1")
        ttk.Combobox(self, textvariable=self.speed, values=self.SPEEDS, width=5, state="readonly").grid(row=2, column=4)
        self.time_label = ttk.Label(self, text="", width=24, anchor="e")
        self.time_label.grid(row=2, column=6, sticky="e")
        self.info = ttk.Label(self, text="", anchor="w")
        self.info.grid(row=3, column=0, columnspan=7, sticky="ew")
        self.columnconfigure(5, weight=1)
        self.rowconfigure(0, weight=1)

    def open(self, path: Path, frame: int = 0, describe: Callable[[int], str] | None = None) -> bool:
        self.close()
        self.describe = describe
        if not Path(path).exists():
            self._blank(f"Not found: {path}")
            return False
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            self._blank(f"Cannot open {Path(path).name}")
            return False
        self.capture = capture
        self.n_frames = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        self.fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        self.slider.configure(to=max(self.n_frames - 1, 1))
        self.seek(frame)
        return True

    def close(self) -> None:
        self.pause()
        if self.capture is not None:
            self.capture.release()
        self.capture, self.n_frames, self.frame = None, 0, 0
        self._blank("No video")

    def _blank(self, message: str) -> None:
        self._image, self.message = None, message
        self.time_label.configure(text="")
        self.info.configure(text="")
        self._draw()

    def seek(self, frame: int) -> None:
        if self.capture is None:
            return
        frame = int(min(max(frame, 0), self.n_frames - 1))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        self._read(frame)

    def _read(self, frame: int) -> bool:
        ok, image = self.capture.read()
        if not ok:
            return False
        self.frame, self._image = frame, image
        self._draw()
        self._sliding = True
        self.position.set(frame)
        self._sliding = False
        self.time_label.configure(text=f"frame {frame}/{self.n_frames - 1}   {frame / self.fps:7.2f} s")
        self.info.configure(text=self.describe(frame) if self.describe else "")
        if self.on_frame:
            self.on_frame(frame)
        return True

    def _draw(self) -> None:
        self.screen.delete("all")
        width, height = max(self.screen.winfo_width(), 2), max(self.screen.winfo_height(), 2)
        if self._image is None:
            self.screen.create_text(width / 2, height / 2, text=self.message, fill="white", width=width - 20)
            return
        h, w = self._image.shape[:2]
        scale = min(width / w, height / h)
        size = (max(int(w * scale), 1), max(int(h * scale), 1))
        rgb = cv2.cvtColor(cv2.resize(self._image, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))  # kept: Tk does not hold a reference
        self.screen.create_image(width / 2, height / 2, image=self._photo)

    def _slide(self, _value) -> None:
        if not self._sliding and self.capture is not None:
            self.seek(round(self.position.get()))

    def toggle(self) -> None:
        if self.playing:
            self.pause()
        elif self.capture is not None:
            if self.frame >= self.n_frames - 1:
                self.seek(0)
            self.playing = True
            self.play_button.configure(text="❚❚ Pause")
            self._tick()

    def pause(self) -> None:
        self.playing = False
        self.play_button.configure(text="▶ Play")
        if self._job is not None:
            self.after_cancel(self._job)
            self._job = None

    def step(self, frames: int) -> None:
        self.pause()
        self.seek(self.frame + frames)

    def _tick(self) -> None:
        self._job = None
        if not self.playing or self.capture is None:
            return
        if self.frame >= self.n_frames - 1 or not self._read(self.frame + 1):
            self.pause()
            return
        self._job = self.after(max(1, int(1000 / (self.fps * float(self.speed.get())))), self._tick)


# ======================================================================
# The window
# ======================================================================
class Explorer(tk.Toplevel):
    TABS = ("recordings", "map", "clips", "gif")

    def __init__(self, app: App, cfg: ExperimentConfig):
        super().__init__(app.root)
        self.app, self.cfg = app, cfg
        self.title(f"Explore results — {cfg.yaml_path.name}")
        self.geometry("1280x820")
        self.minsize(1000, 680)
        self.protocol("WM_DELETE_WINDOW", self.close)

        bar = ttk.Frame(self, padding=(8, 8, 8, 0))
        bar.pack(fill="x")
        ttk.Label(bar, text="Segmentation").pack(side="left")
        self.algorithm = tk.StringVar(value=cfg.algorithms[0])
        self.algorithm_box = ttk.Combobox(bar, textvariable=self.algorithm, values=cfg.algorithms, state="readonly",
                                          width=8)
        self.algorithm_box.pack(side="left", padx=4)
        self.algorithm_box.bind("<<ComboboxSelected>>", lambda _: self.reload(read_config=False))
        ttk.Button(bar, text="Reload", command=self.reload).pack(side="left", padx=4)
        self.summary = ttk.Label(bar, text="", foreground="gray")
        self.summary.pack(side="left", padx=8)

        self.notebook = ttk.Notebook(self, padding=8)
        self.notebook.pack(fill="both", expand=True)
        self._build_recordings()
        self._build_map()
        self._build_clips()
        self._build_gif()
        self.reload(read_config=False)

    def show(self, tab: str) -> None:
        self.notebook.select(self.TABS.index(tab))
        self.deiconify()
        self.lift()

    def close(self) -> None:
        for player in (self.recording_player, self.clip_player, self.gif_player):
            player.close()
        self.app.explorer = None
        self.destroy()

    # ==================================================================
    # Loading
    # ==================================================================
    def reload(self, read_config: bool = True) -> None:
        """Read the results again (after a command, or when files changed)."""
        if read_config:
            try:
                self.cfg = load_config(self.cfg.yaml_path)
            except Exception as e:  # any problem with the experiment: say so, keep the old one
                messagebox.showerror("Explore results", f"Cannot read {self.cfg.yaml_path.name}: {e}", parent=self)
                return
            self.algorithm_box.configure(values=self.cfg.algorithms)
            if self.algorithm.get() not in self.cfg.algorithms:
                self.algorithm.set(self.cfg.algorithms[0])
        try:
            self.results = Results(self.cfg, self.algorithm.get())
        except (OSError, ValueError) as e:
            messagebox.showerror("Explore results", f"Cannot read the results: {e}", parent=self)
            return
        r = self.results
        have = [f"{len(r.sessions)} recording(s)", f"{len(r.videos)} video(s)"]
        have.append(f"{len(r.bag)} communities (cut {self.cfg.community_cut_tree})" if r.bag else "no communities yet")
        have.append("motif map" if r.umap is not None else "no motif map yet")
        self.summary.configure(text=" · ".join(have))
        self.names = r.names()
        self.picked = None  # a point of the previous map
        self.play_point_button.configure(state="disabled")
        self._refresh_recordings()
        self._refresh_map()
        self._refresh_clips()
        self._refresh_gifs()

    def _run(self, command: str, extra: list[str] | None = None) -> None:
        """Run a vame-motifs command through the main window, then reload."""
        if self.app.running:
            messagebox.showinfo("Explore results", "A command is already running: wait for it to finish.", parent=self)
            return
        self.app.run_command(command, extra, on_done=lambda code: self.reload() if code == 0 else None)

    def _community_name(self, community: int) -> str:
        name = self.names.get(community, "")
        return f"community {community}" + (f" '{name}'" if name else "")

    # ==================================================================
    # Recordings: the input videos
    # ==================================================================
    def _build_recordings(self) -> None:
        tab = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab, text="Recordings")
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(0, weight=1)

        left = ttk.Frame(tab)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        ttk.Label(left, text="Recordings (input videos)").pack(anchor="w")
        self.session_list = tk.Listbox(left, width=42, exportselection=False)
        self.session_list.pack(fill="both", expand=True)
        self.session_list.bind("<<ListboxSelect>>", lambda _: self._open_recording())
        self.recording_note = ttk.Label(left, text="", foreground="gray", wraplength=300, justify="left")
        self.recording_note.pack(anchor="w", pady=(4, 0))

        self.recording_player = VideoPlayer(tab, width=720, height=540)
        self.recording_player.grid(row=0, column=1, sticky="nsew")

        find = ttk.Frame(tab)
        find.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(find, text="Find").pack(side="left")
        self.find_kind = tk.StringVar(value="motif")
        kind = ttk.Combobox(find, textvariable=self.find_kind, values=["motif", "community"], state="readonly", width=10)
        kind.pack(side="left", padx=4)
        kind.bind("<<ComboboxSelected>>", lambda _: self._fill_find_values())
        self.find_value = tk.StringVar()
        self.find_box = ttk.Combobox(find, textvariable=self.find_value, state="readonly", width=28)
        self.find_box.pack(side="left", padx=4)
        self.find_box.bind("<<ComboboxSelected>>", lambda _: self._update_find())
        ttk.Button(find, text="◀ Previous bout", command=lambda: self._jump(-1)).pack(side="left", padx=(8, 2))
        ttk.Button(find, text="Next bout ▶", command=lambda: self._jump(1)).pack(side="left", padx=2)
        self.find_label = ttk.Label(find, text="", foreground="gray")
        self.find_label.pack(side="left", padx=8)

    def _refresh_recordings(self) -> None:
        r = self.results
        selected = self._selected_session()
        self.session_list.delete(0, "end")
        for session in r.sessions:
            self.session_list.insert("end", session if session in r.videos else f"{session}  (no video)")
        self._fill_find_values()
        if r.sessions:
            index = r.sessions.index(selected) if selected in r.sessions else 0
            self.session_list.selection_set(index)
            self._open_recording(keep_frame=selected == r.sessions[index])

    def _selected_session(self) -> str | None:
        selection = self.session_list.curselection()
        return self.results.sessions[selection[0]] if selection and hasattr(self, "results") else None

    def _open_recording(self, frame: int | None = None, keep_frame: bool = False) -> None:
        session = self._selected_session()
        if session is None:
            return
        r = self.results
        try:
            motifs = r.frame_motifs(session)
        except ExportError as e:
            motifs = None
            self.recording_note.configure(text=str(e))
        if frame is None:
            frame = self.recording_player.frame if keep_frame else 0
        if session not in r.videos:
            self.recording_player.close()
            self.recording_player._blank("No video for this recording: set 'Videos' in the Data tab.")
            self.recording_note.configure(text="")
            return
        video = r.videos[session]
        self.recording_note.configure(
            text=f"{video}\n" + ("" if motifs is not None else "No motifs yet: run 'segment'."))
        self.recording_player.open(video, frame, describe=lambda f: self._describe_frame(session, f))
        self._update_find()

    def _describe_frame(self, session: str, frame: int) -> str:
        motifs = self.results.frame_motifs(session) if self.results.window_labels(session) is not None else None
        if motifs is None:
            return ""
        if frame >= len(motifs) or motifs[frame] == NO_MOTIF:
            return "No motif (first and last frames: outside VAME's time windows)"
        motif = int(motifs[frame])
        community = self.results.community_of().get(motif)
        text = f"Motif {motif}"
        if community is not None:
            text += f"  ·  {self._community_name(community).capitalize()}"
        return text

    def _fill_find_values(self) -> None:
        if self.find_kind.get() == "community" and self.results.bag:
            values = [self._community_name(i) for i in range(len(self.results.bag))]
        elif self.find_kind.get() == "community":
            values = []
        else:
            values = [f"motif {m}" for m in range(self.cfg.n_clusters)]
        self.find_box.configure(values=values)
        if self.find_value.get() not in values:
            self.find_value.set(values[0] if values else "")
        self._update_find()

    def _find_mask(self) -> np.ndarray | None:
        """Frames of the selected recording that show the chosen motif or community."""
        session, text = self._selected_session(), self.find_value.get()
        if session is None or not text or self.results.window_labels(session) is None:
            return None
        motifs = self.results.frame_motifs(session)
        index = int(text.split()[1])
        if self.find_kind.get() == "motif":
            return motifs == index
        return np.isin(motifs, self.results.bag[index])

    def _update_find(self) -> None:
        mask = self._find_mask()
        if mask is None:
            self.find_label.configure(text="")
            return
        n = len(bouts(mask))
        self.find_label.configure(text=f"{n} bouts, {mask.mean():.1%} of the recording")

    def _jump(self, direction: int) -> None:
        mask = self._find_mask()
        if mask is None or not mask.any():
            return
        starts = bouts(mask)[:, 0]
        current = self.recording_player.frame
        later, earlier = starts[starts > current], starts[starts < current]
        target = (later[0] if len(later) else starts[0]) if direction > 0 else (earlier[-1] if len(earlier) else starts[-1])
        self.recording_player.pause()
        self.recording_player.seek(int(target))

    def play_moment(self, session: str, frame: int) -> None:
        """Show a recording at a frame (from the map or a clip)."""
        if session not in self.results.sessions:
            return
        self.session_list.selection_clear(0, "end")
        self.session_list.selection_set(self.results.sessions.index(session))
        self.show("recordings")
        self._open_recording(frame=frame)

    # ==================================================================
    # Motifs & communities: the tree, its cut, the UMAP
    # ==================================================================
    def _build_map(self) -> None:
        tab = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab, text="Motifs & communities")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)

        controls = ttk.Frame(tab)
        controls.grid(row=0, column=0, sticky="ew")
        ttk.Label(controls, text="Cut the tree at").pack(side="left")
        self.cut = tk.StringVar(value=str(self.cfg.community_cut_tree))
        self.cut_box = ttk.Spinbox(controls, from_=0, to=30, textvariable=self.cut, width=4,
                                   command=self._refresh_map)
        self.cut_box.pack(side="left", padx=4)
        self.cut_box.bind("<Return>", lambda _: self._refresh_map())
        self.cut_box.bind("<FocusOut>", lambda _: self._refresh_map())
        ttk.Button(controls, text="Apply this cut (run 'communities')", command=self._apply_cut).pack(side="left", padx=8)
        ttk.Label(controls, text="Colour the map by").pack(side="left", padx=(16, 4))
        self.colour_by = tk.StringVar(value="community")
        for value in ("motif", "community", "recording"):
            ttk.Radiobutton(controls, text=value, value=value, variable=self.colour_by,
                            command=self._refresh_map).pack(side="left")
        self.umap_button = ttk.Button(controls, text="Make the motif map (run 'umap')", command=lambda: self._run("umap"))
        self.umap_button.pack(side="right")

        self.figure = Figure(figsize=(11, 6), layout="constrained")
        self.tree_ax, self.map_ax = self.figure.subplots(1, 2, width_ratios=[1, 1.2])
        self.canvas = FigureCanvasTkAgg(self.figure, master=tab)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        self.canvas.mpl_connect("button_press_event", self._map_click)
        toolbar_frame = ttk.Frame(tab)
        toolbar_frame.grid(row=2, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="left")

        bottom = ttk.Frame(tab)
        bottom.grid(row=3, column=0, sticky="ew")
        self.map_info = ttk.Label(bottom, foreground="gray", text=(
            "Click the tree at a height to cut it there. Click the map to see that moment of a recording."))
        self.map_info.pack(side="left")
        self.play_point_button = ttk.Button(bottom, text="Play this moment", state="disabled",
                                            command=lambda: self.play_moment(*self.picked))
        self.play_point_button.pack(side="right")
        self.picked: tuple[str, int] | None = None

    def _cut_value(self) -> int:
        try:
            return max(int(self.cut.get()), 0)
        except ValueError:
            return self.cfg.community_cut_tree

    def _preview_bag(self) -> list[list[int]] | None:
        r = self.results
        if r.tree is None:
            return None
        return communities.cut_tree(r.tree, min(self._cut_value(), communities.max_cut(r.tree)))

    def _refresh_map(self) -> None:
        r = self.results
        if r.tree is not None:
            self.cut_box.configure(to=communities.max_cut(r.tree))
        bag = self._preview_bag()
        names = communities.read_labels(self.cfg, r.algorithm, bag) if bag else {}
        self._draw_tree(bag, names)
        self._draw_umap(bag, names)
        self.umap_button.configure(text="Redo the motif map (run 'umap --force')" if r.umap is not None
                                   else "Make the motif map (run 'umap')",
                                   command=lambda: self._run("umap", ["--force"] if r.umap is not None else []))
        self.canvas.draw_idle()

    def _draw_tree(self, bag, names: dict[int, str]) -> None:
        ax, r = self.tree_ax, self.results
        ax.clear()
        ax.set_axis_off()
        if r.tree is None:
            ax.text(0.5, 0.5, "No motif tree yet.\nApply a cut once: 'communities' builds the tree.",
                    ha="center", va="center", transform=ax.transAxes)
            return
        cut = min(self._cut_value(), communities.max_cut(r.tree))
        pos = communities.tree_layout(r.tree)
        for a, b in r.tree.edges:
            (xa, ya), (xb, yb) = pos[a], pos[b]
            ax.plot([xa, xb, xb], [ya, ya, yb], color="0.6", lw=1, zorder=1)  # elbow: across, then down
        community_of = communities.motif_to_community(bag)
        usage = r.usage()
        for node, (x, y) in pos.items():
            if communities.is_motif(node):
                motif = int(node)
                c = community_of.get(motif, 0)
                ax.scatter([x], [y], s=60 + 1500 * (usage[motif] if motif < len(usage) else 0),
                           color=colour(c), edgecolor="black", lw=0.5, zorder=2)
                ax.annotate(str(motif), (x, y), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=8)
        ax.axhline(0.5 - cut, color="#c62828", ls="--", lw=1.2)
        ax.text(0.01, 0.5 - cut, f" cut = {cut}", color="#c62828", va="bottom", fontsize=9,
                transform=ax.get_yaxis_transform())
        ax.set_ylim(-communities.max_cut(r.tree) - 0.8, 0.8)
        saved = f"applied: cut {self.cfg.community_cut_tree}, {len(r.bag)} communities" if r.bag else "not applied yet"
        ax.set_title(f"Motif tree: cut {cut} makes {len(bag)} communities ({saved})", fontsize=10)
        handles = [Line2D([0], [0], marker="o", ls="", color=colour(i),
                          label=f"{i}: {' '.join(map(str, motifs))}" + (f" ({names[i]})" if i in names else ""))
                   for i, motifs in enumerate(bag)]
        ax.legend(handles=handles, title="Community: motifs", loc="upper center", fontsize=7, title_fontsize=8,
                  bbox_to_anchor=(0.5, 0), ncol=min(len(bag), 3), frameon=False)

    def _draw_umap(self, bag, names: dict[int, str]) -> None:
        ax, r = self.map_ax, self.results
        ax.clear()
        if r.umap is None:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No motif map yet.\n'Make the motif map' computes it (about a minute).",
                    ha="center", va="center", transform=ax.transAxes)
            return
        ax.set_axis_on()
        xy, motifs = r.umap["embedding"], r.point_motif
        by = self.colour_by.get()
        if by == "community" and bag:
            community_of = communities.motif_to_community(bag)
            values = np.array([community_of.get(int(m), NO_MOTIF) for m in motifs])
            label = lambda v: f"{v}" + (f" {names[v]}" if v in names else "")
        elif by == "recording":
            values = r.umap["session"]
            label = lambda v: str(r.umap["sessions"][v])[:30]
        else:
            values = motifs
            label = lambda v: f"motif {v}"
        colours = np.array([colour(v) if v != NO_MOTIF else (0.8, 0.8, 0.8, 1) for v in range(values.max() + 1)]
                           + [(0.8, 0.8, 0.8, 1)])
        ax.scatter(xy[:, 0], xy[:, 1], s=1, c=colours[values], alpha=0.6, linewidths=0, rasterized=True)
        shown = sorted(v for v in np.unique(values) if v != NO_MOTIF)
        if len(shown) <= 20:
            ax.legend(handles=[Line2D([0], [0], marker="o", ls="", color=colour(v), label=label(v)) for v in shown],
                      fontsize=7, loc="upper right", markerscale=0.8, framealpha=0.7)
        if self.picked:
            index = self._picked_index
            ax.scatter([xy[index, 0]], [xy[index, 1]], s=120, facecolor="none", edgecolor="black", lw=1.5)
        ax.set_title(f"UMAP of the latent space ({len(xy)} time windows), by {by}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", "datalim")

    def _map_click(self, event) -> None:
        if self.toolbar.mode or event.xdata is None:  # zooming or panning, or outside the axes
            return
        r = self.results
        if event.inaxes is self.tree_ax and r.tree is not None:
            self.cut.set(str(int(np.clip(round(0.5 - event.ydata), 0, communities.max_cut(r.tree)))))
            self._refresh_map()
        elif event.inaxes is self.map_ax and r.umap is not None:
            xy = r.umap["embedding"]
            index = int(np.argmin((xy[:, 0] - event.xdata) ** 2 + (xy[:, 1] - event.ydata) ** 2))
            session = str(r.umap["sessions"][r.umap["session"][index]])
            frame = int(r.umap["window"][index]) + self.cfg.time_window // 2  # a window is shown on its centre
            motif = int(r.point_motif[index])
            community_of = communities.motif_to_community(self._preview_bag() or [])
            text = f"{session}, frame {frame} ({frame / self.cfg.fps:.1f} s): motif {motif}"
            if motif in community_of:
                text += f", {self._community_name(community_of[motif])} at this cut"
            self.picked, self._picked_index = (session, frame), index
            self.map_info.configure(text=text, foreground="")
            self.play_point_button.configure(state="normal" if session in r.videos else "disabled")
            self._refresh_map()

    def _apply_cut(self) -> None:
        cut = self._cut_value()
        r = self.results
        if r.tree is not None and cut > communities.max_cut(r.tree):
            cut = communities.max_cut(r.tree)
        if not messagebox.askyesno(
                "Communities", f"Group the motifs into communities with the tree cut at {cut}?\n\n"
                f"This sets 'Community cut tree' to {cut}, saves the experiment and runs 'communities'. "
                "Existing motif clips are moved into the new community folders.", parent=self):
            return
        self.app.vars["community_cut_tree"].set(str(cut))
        self._run("communities")

    # ==================================================================
    # Communities & clips: watch, name, save
    # ==================================================================
    def _build_clips(self) -> None:
        tab = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab, text="Communities & clips")
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(1, weight=1)

        top = ttk.Frame(tab)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ttk.Label(top, text="Clips of recording").pack(side="left")
        self.clip_session = tk.StringVar()
        self.clip_session_box = ttk.Combobox(top, textvariable=self.clip_session, state="readonly", width=60)
        self.clip_session_box.pack(side="left", padx=4)
        self.clip_session_box.bind("<<ComboboxSelected>>", lambda _: self._refresh_clips())
        ttk.Button(top, text="Cut the clips (run 'videos')", command=lambda: self._run("videos")).pack(side="right")

        left = ttk.Frame(tab)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        left.rowconfigure(0, weight=1)
        self.community_tree = ttk.Treeview(left, columns=("name", "usage", "clip"), height=18, selectmode="browse")
        for column, text, width in [("#0", "Community / motif", 150), ("name", "Name", 170),
                                    ("usage", "Time", 60), ("clip", "Clip", 50)]:
            self.community_tree.heading(column, text=text)
            self.community_tree.column(column, width=width, stretch=column == "name")
        self.community_tree.grid(row=0, column=0, columnspan=3, sticky="nsew")
        self.community_tree.bind("<<TreeviewSelect>>", lambda _: self._select_clip())

        ttk.Label(left, text="Name of the selected community").grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(left, textvariable=self.name_var, width=30)
        self.name_entry.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.name_entry.bind("<Return>", lambda _: self._set_name())
        ttk.Button(left, text="Set", width=5, command=self._set_name).grid(row=2, column=2, padx=(4, 0))
        ttk.Button(left, text="Save names (…_labeled.csv)", command=self._save_names).grid(
            row=3, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        ttk.Button(left, text="Find in the recording", command=self._find_selected).grid(
            row=4, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        self.names_note = ttk.Label(left, text="", foreground="gray", wraplength=380, justify="left")
        self.names_note.grid(row=5, column=0, columnspan=3, sticky="w", pady=(6, 0))

        self.clip_player = VideoPlayer(tab, width=640, height=500)
        self.clip_player.grid(row=1, column=1, sticky="nsew")

    def _refresh_clips(self) -> None:
        r = self.results
        self.clip_session_box.configure(values=r.sessions)
        if self.clip_session.get() not in r.sessions:
            self.clip_session.set(r.sessions[0] if r.sessions else "")
        tree = self.community_tree
        selected = tree.selection()
        tree.delete(*tree.get_children())
        clips = r.clips(self.clip_session.get()) if self.clip_session.get() else {}
        usage = r.usage()
        if not r.bag:
            self.names_note.configure(text="No communities yet: choose and apply a cut in 'Motifs & communities'.")
            for motif in range(self.cfg.n_clusters):
                tree.insert("", "end", iid=f"m{motif}", text=f"Motif {motif}",
                            values=("", f"{usage[motif]:.1%}", "yes" if motif in clips else "–"))
        else:
            for i, motifs in enumerate(r.bag):
                tree.insert("", "end", iid=f"c{i}", text=f"Community {i}", open=True,
                            values=(self.names.get(i, ""), f"{usage[motifs].sum():.1%}", ""))
                for motif in motifs:
                    tree.insert(f"c{i}", "end", iid=f"m{motif}", text=f"Motif {motif}",
                                values=("", f"{usage[motif]:.1%}", "yes" if motif in clips else "–"))
            saved = communities.labels_path(self.cfg, r.algorithm)
            self.names_note.configure(text=(
                f"Names are saved in {saved.name} and the …_motifs_labeled.csv files, in {saved.parent}."
                + ("" if clips else "\nNo clips for this recording yet: 'Cut the clips' (needs input videos).")))
        if selected and tree.exists(selected[0]):
            tree.selection_set(selected[0])

    def _selected_item(self) -> tuple[str, int] | None:
        selection = self.community_tree.selection()
        return (selection[0][0], int(selection[0][1:])) if selection else None

    def _select_clip(self) -> None:
        item = self._selected_item()
        if item is None:
            return
        kind, index = item
        if kind == "c":
            self.name_var.set(self.names.get(index, ""))
            motifs = self.results.bag[index]
        else:
            motifs = [index]
        clips = self.results.clips(self.clip_session.get())
        available = [m for m in motifs if m in clips]
        if not available:
            self.clip_player.close()
            self.clip_player._blank("No clip for this motif in this recording. "
                                    "'Cut the clips' makes them (VAME skips motifs that never occur).")
            return
        clip = clips[available[0]]
        self.clip_player.open(clip, describe=lambda _: f"{clip.parent.name} / {clip.name}")
        self.clip_player.toggle()

    def _set_name(self) -> None:
        item = self._selected_item()
        if item is None or item[0] != "c":
            messagebox.showinfo("Names", "Select a community first.", parent=self)
            return
        self.names[item[1]] = self.name_var.get().strip()
        self.community_tree.set(f"c{item[1]}", "name", self.names[item[1]])

    def _save_names(self) -> None:
        r = self.results
        if not r.bag:
            messagebox.showinfo("Names", "There are no communities yet: apply a cut first.", parent=self)
            return
        try:
            written = communities.write_labels(self.cfg, r.algorithm, r.bag, self.names)
        except (communities.CommunityError, ExportError, OSError) as e:
            messagebox.showerror("Names", str(e), parent=self)
            return
        self.app._write_log(f"Saved community names: {', '.join(p.name for p in written)} in {written[0].parent}\n",
                            "info")
        self.names_note.configure(text=f"Saved {len(written)} file(s) in {written[0].parent}")
        self._refresh_recordings()
        self._refresh_map()

    def _find_selected(self) -> None:
        item = self._selected_item()
        if item is None:
            return
        kind, index = item
        session = self.clip_session.get()
        self.find_kind.set("community" if kind == "c" else "motif")
        self._fill_find_values()
        self.find_value.set(self._community_name(index) if kind == "c" else f"motif {index}")
        if session in self.results.sessions:
            self.play_moment(session, 0)
            self._jump(1)

    # ==================================================================
    # GIF: vame.gif
    # ==================================================================
    def _build_gif(self) -> None:
        tab = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab, text="GIF")
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(0, weight=1)

        form = ttk.Frame(tab)
        form.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        ttk.Label(form, wraplength=330, justify="left", text=(
            "vame.gif: the animal, aligned and cropped, next to the UMAP of its latent space with its "
            "recent path drawn on it. Needs the input videos.")).grid(row=0, column=0, columnspan=2, sticky="w")
        self.gif_vars = {key: tk.StringVar(value=value) for key, value in
                         [("session", ""), ("start", ""), ("length", "500"), ("label", "community")]}
        ttk.Label(form, text="Recording").grid(row=1, column=0, sticky="w", pady=(10, 2))
        self.gif_session_box = ttk.Combobox(form, textvariable=self.gif_vars["session"], state="readonly", width=34)
        self.gif_session_box.grid(row=2, column=0, columnspan=2, sticky="ew")
        for row, (label, key, hint) in enumerate([
            ("First time window", "start", "empty = a random, well-tracked moment"),
            ("Length (frames)", "length", "500 frames = 17 s at 30 fps"),
        ], start=3):
            ttk.Label(form, text=label).grid(row=2 * row - 3, column=0, sticky="w", pady=(8, 0))
            ttk.Entry(form, textvariable=self.gif_vars[key], width=10).grid(row=2 * row - 2, column=0, sticky="w")
            ttk.Label(form, text=hint, foreground="gray").grid(row=2 * row - 2, column=1, sticky="w")
        ttk.Label(form, text="Colour the map by").grid(row=7, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(form, textvariable=self.gif_vars["label"], values=["community", "motif", "none"],
                     state="readonly", width=12).grid(row=8, column=0, sticky="w")
        self.gif_background = tk.BooleanVar(value=False)
        ttk.Checkbutton(form, text="Subtract the background (slower)", variable=self.gif_background).grid(
            row=9, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(form, text="Make the GIF (run 'gif')", command=self._make_gif).grid(
            row=10, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        ttk.Label(form, text="GIFs made").grid(row=11, column=0, sticky="w", pady=(16, 2))
        self.gif_list = tk.Listbox(form, height=8, exportselection=False)
        self.gif_list.grid(row=12, column=0, columnspan=2, sticky="nsew")
        self.gif_list.bind("<<ListboxSelect>>", lambda _: self._play_gif())
        form.rowconfigure(12, weight=1)

        self.gif_player = VideoPlayer(tab, width=640, height=480)
        self.gif_player.grid(row=0, column=1, sticky="nsew")

    def _refresh_gifs(self) -> None:
        r = self.results
        self.gif_session_box.configure(values=r.sessions)
        if self.gif_vars["session"].get() not in r.sessions:
            self.gif_vars["session"].set(r.sessions[0] if r.sessions else "")
        self.gifs = r.gifs()
        self.gif_list.delete(0, "end")
        for gif in self.gifs:
            self.gif_list.insert("end", gif.name)

    def _make_gif(self) -> None:
        extra = ["--session", self.gif_vars["session"].get(), "--algorithm", self.algorithm.get(),
                 "--label", self.gif_vars["label"].get()]
        for key in ("start", "length"):
            text = self.gif_vars[key].get().strip()
            if text:
                if not text.isdigit():
                    messagebox.showerror("GIF", f"'{text}' is not a whole number.", parent=self)
                    return
                extra += [f"--{key}", text]
        if self.gif_background.get():
            extra.append("--subtract-background")
        self._run("gif", extra)

    def _play_gif(self) -> None:
        selection = self.gif_list.curselection()
        if selection:
            gif = self.gifs[selection[0]]
            self.gif_player.open(gif, describe=lambda _: gif.name)
            self.gif_player.toggle()
