"""Desktop window for vame-motifs: ``vame-motifs-gui`` (or ``python -m vame_motifs.gui``).

The window adds nothing to the analysis. It does three things:

- edits the experiment YAML (the same file the CLI reads), as a form;
- runs the CLI commands on that file and shows their output;
- 'Explore results' (viewer.py): plays the input videos and motif clips, shows
  the motif tree and UMAP, lets the user pick the tree cut, name the
  communities (saved as *_labeled.csv) and make GIFs (vame.gif).

Each command runs as its own process (``python -m vame_motifs <command> -c
<yaml>``), exactly as it would from a terminal: same steps, same run records,
same exit codes. That keeps the window responsive during training and lets a
run be stopped.

With 'On GPU server' ticked, the same command runs on a server over SSH
instead, with the files copied there and the results copied back (remote.py).

Tkinter ships with Python, so no extra package is needed.
"""

from __future__ import annotations

import os
import queue
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import yaml

from vame_motifs import __version__, remote
from vame_motifs.cli import COMMANDS
from vame_motifs.config import METHODS, VIDEO_SUFFIXES

# Commands in pipeline order, as buttons. 'run' gets its own, larger button.
STEP_COMMANDS = ["validate", "init", "prepare", "train", "segment", "export", "communities", "videos"]

# Form defaults: the same values as the README's example experiment and config.py.
DEFAULTS = {
    "pose_files": "",
    "videos": "",
    "fps": "30",
    "align_center": "",
    "align_direction": "",
    "exclude": [],
    "min_confidence": "0.9",
    "n_clusters": "15",
    "method": "hmm",
    "output": "outputs",
    "max_epochs": "100",
    "time_window": "30",
    "zdims": "30",
    "seed": "42",
    "community_cut_tree": "3",
}


def read_bodyparts(path: Path) -> list[str]:
    """Body part names from a DLC CSV header (only the header rows are read)."""
    import pandas as pd

    columns = pd.read_csv(path, header=[0, 1, 2], index_col=0, nrows=0).columns
    return list(dict.fromkeys(columns.get_level_values(1)))


def yaml_to_form(raw: dict) -> dict:
    """Experiment YAML (as loaded) -> form values. Missing keys keep their defaults."""
    def section(name: str) -> dict:
        value = raw.get(name)
        return value if isinstance(value, dict) else {}

    inp, kp, motifs = section("input"), section("keypoints"), section("motifs")
    cleaning, adv = section("cleaning"), section("advanced")
    values = dict(DEFAULTS)
    sources = {
        "pose_files": inp, "videos": inp, "fps": inp,
        "align_center": kp, "align_direction": kp, "exclude": kp,
        "min_confidence": cleaning,
        "n_clusters": motifs, "method": motifs,
        "max_epochs": adv, "time_window": adv, "zdims": adv, "seed": adv, "community_cut_tree": adv,
    }
    for key, sec in sources.items():
        if sec.get(key) is not None:
            values[key] = sec[key] if key == "exclude" else str(sec[key])
    if raw.get("output") is not None:
        values["output"] = str(raw["output"])
    values["exclude"] = list(values["exclude"] or [])
    return values


def form_to_yaml(values: dict) -> dict:
    """Form values -> experiment YAML. Numbers are written as numbers when they parse."""
    def number(text: str):
        text = str(text).strip()
        for cast in (int, float):
            try:
                return cast(text)
            except ValueError:
                pass
        return text  # left as typed: load_config reports the wrong type

    inp = {"pose_files": values["pose_files"].strip()}
    if values["videos"].strip():
        inp["videos"] = values["videos"].strip()
    inp["fps"] = number(values["fps"])
    return {
        "input": inp,
        "keypoints": {
            "align_center": values["align_center"].strip(),
            "align_direction": values["align_direction"].strip(),
            "exclude": list(values["exclude"]),
        },
        "cleaning": {"min_confidence": number(values["min_confidence"])},
        "motifs": {"n_clusters": number(values["n_clusters"]), "method": values["method"]},
        "output": values["output"].strip() or "outputs",
        "advanced": {key: number(values[key]) for key in ("max_epochs", "time_window", "zdims", "seed", "community_cut_tree")},
    }


class App(ttk.Frame):
    def __init__(self, root: tk.Tk, yaml_path: Path | None = None):
        super().__init__(root, padding=8)
        self.root = root
        self.yaml_path: Path | None = None
        self.process: subprocess.Popen | None = None  # the running step of a job
        self.running = False                           # a job (one or more steps) is running
        self.stopping = False
        self.job_remote = False
        # ('line', text) | ('info', text) | ('done', exit code, failed step kind) | ('tunnel_closed',)
        self.output_queue: queue.Queue[tuple] = queue.Queue()
        self.vars: dict[str, tk.StringVar] = {}
        self.dirty = False
        self.settings = remote.load_settings()
        self.tunnel: subprocess.Popen | None = None    # the open SSH connection ('Connect')
        self.tunnel_settings: remote.RemoteSettings | None = None
        self.tunnel_ready = False
        self.askpass_dir: str | None = None
        self.on_done = None                            # called with the exit code when the job ends
        self.explorer = None                           # the 'Explore results' window, when open

        root.title("vame-motifs")
        root.minsize(980, 680)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.grid(sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        self._build_file_bar()
        self._build_form()
        self._build_steps()
        self._build_log()
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(2, weight=1)

        self.set_values(DEFAULTS)
        self._load_server_settings()
        if yaml_path:
            self.load(Path(yaml_path))
        self._update_title()
        self.after(100, self._drain_output)

    # ==================================================================
    # Layout
    # ==================================================================
    def _build_file_bar(self) -> None:
        bar = ttk.Frame(self)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        ttk.Label(bar, text="Experiment file:").pack(side="left")
        self.path_label = ttk.Label(bar, text="(not saved)", foreground="gray")
        self.path_label.pack(side="left", padx=6, fill="x", expand=True)
        for text, command in [("New", self.new), ("Open…", self.open), ("Save", self.save), ("Save as…", self.save_as)]:
            ttk.Button(bar, text=text, command=command).pack(side="left", padx=2)

    def _entry(self, parent, row: int, label: str, key: str, browse: str | None = None, hint: str = "", width: int = 40):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
        var = self.vars.setdefault(key, tk.StringVar())
        var.trace_add("write", lambda *_: self._mark_dirty())
        ttk.Entry(parent, textvariable=var, width=width).grid(row=row, column=1, sticky="ew", pady=2)
        col = 2
        if browse in ("file+folder", "folder"):
            if browse == "file+folder":
                ttk.Button(parent, text="File…", width=6, command=lambda: self._browse_file(key)).grid(row=row, column=col, padx=2)
                col += 1
            ttk.Button(parent, text="Folder…", width=7, command=lambda: self._browse_folder(key)).grid(row=row, column=col, padx=2)
            col += 1
        if hint:
            ttk.Label(parent, text=hint, foreground="gray").grid(row=row, column=4, sticky="w", padx=4)
        return var

    def _build_form(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.grid(row=1, column=0, sticky="nsew", padx=(0, 8))

        # --- Data ---------------------------------------------------------
        data = ttk.Frame(notebook, padding=10)
        data.columnconfigure(1, weight=1)
        notebook.add(data, text="Data")
        self._entry(data, 0, "Pose files", "pose_files", browse="file+folder")
        ttk.Label(data, text="A folder of DeepLabCut .csv files, or one .csv file.", foreground="gray").grid(
            row=1, column=1, columnspan=4, sticky="w")
        self._entry(data, 2, "Videos (optional)", "videos", browse="file+folder")
        ttk.Label(data, text="Raw videos, only needed for 'videos'. Leave empty to skip.", foreground="gray").grid(
            row=3, column=1, columnspan=4, sticky="w")
        self._entry(data, 4, "Frame rate (fps)", "fps", width=10)
        self._entry(data, 5, "Output folder", "output", browse="folder")
        ttk.Label(data, text="Paths are relative to the experiment file.", foreground="gray").grid(
            row=6, column=0, columnspan=5, sticky="w", pady=(10, 0))
        ttk.Button(data, text="Watch the videos…", command=lambda: self.open_explorer("recordings")).grid(
            row=7, column=0, sticky="w", pady=(10, 0))

        # --- Body parts ---------------------------------------------------
        kp = ttk.Frame(notebook, padding=10)
        kp.columnconfigure(1, weight=1)
        notebook.add(kp, text="Body parts")
        ttk.Button(kp, text="Read body parts from pose files", command=self.load_bodyparts).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self.combos: list[ttk.Combobox] = []
        for row, (label, key) in enumerate([("Align center", "align_center"), ("Align direction", "align_direction")], 1):
            ttk.Label(kp, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
            var = self.vars.setdefault(key, tk.StringVar())
            var.trace_add("write", lambda *_: self._mark_dirty())
            combo = ttk.Combobox(kp, textvariable=var, width=30)
            combo.grid(row=row, column=1, sticky="w", pady=2)
            self.combos.append(combo)
        ttk.Label(kp, text="Center is placed at (0, 0); direction sets the orientation.", foreground="gray").grid(
            row=3, column=1, sticky="w")
        ttk.Label(kp, text="Exclude").grid(row=4, column=0, sticky="nw", padx=(0, 6), pady=(10, 2))
        self.exclude_list = tk.Listbox(kp, selectmode="multiple", height=8, exportselection=False)
        self.exclude_list.grid(row=4, column=1, sticky="nsew", pady=(10, 2))
        self.exclude_list.bind("<<ListboxSelect>>", lambda _: self._mark_dirty())
        ttk.Label(kp, text="Selected body parts are left out of the model.", foreground="gray").grid(
            row=5, column=1, sticky="w")
        kp.rowconfigure(4, weight=1)

        # --- Motifs -------------------------------------------------------
        motifs = ttk.Frame(notebook, padding=10)
        motifs.columnconfigure(1, weight=1)
        notebook.add(motifs, text="Motifs")
        self._entry(motifs, 0, "Number of motifs", "n_clusters", width=10)
        ttk.Label(motifs, text="Method").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=2)
        var = self.vars.setdefault("method", tk.StringVar())
        var.trace_add("write", lambda *_: self._mark_dirty())
        ttk.Combobox(motifs, textvariable=var, values=list(METHODS), state="readonly", width=10).grid(
            row=1, column=1, sticky="w", pady=2)
        ttk.Label(motifs, text="hmm is slower but smoother; kmeans is much faster.", foreground="gray").grid(
            row=2, column=1, sticky="w")
        self._entry(motifs, 3, "Min. confidence", "min_confidence", width=10)
        ttk.Label(motifs, text="DLC likelihood below this: point discarded and filled in.", foreground="gray").grid(
            row=4, column=1, sticky="w")
        ttk.Label(motifs, text="New number of motifs or method: just 'Run all' again (the model is reused).",
                  foreground="gray").grid(row=5, column=0, columnspan=2, sticky="w", pady=(10, 0))

        # --- Advanced -----------------------------------------------------
        adv = ttk.Frame(notebook, padding=10)
        adv.columnconfigure(1, weight=1)
        notebook.add(adv, text="Advanced")
        for row, (label, key, hint) in enumerate([
            ("Max epochs", "max_epochs", "at least 10"),
            ("Time window", "time_window", "frames per window"),
            ("Latent dims (zdims)", "zdims", "size of the latent space"),
            ("Seed", "seed", "same seed + same data = same motifs"),
            ("Community cut tree", "community_cut_tree", "higher = more, smaller communities"),
        ]):
            self._entry(adv, row, label, key, hint=hint, width=10)
        ttk.Label(adv, text="Changing these, the alignment, exclusions or min. confidence needs retraining:\n"
                            "tick 'Force' and use 'Run all'.", foreground="gray").grid(
            row=6, column=0, columnspan=5, sticky="w", pady=(10, 0))

        self._build_server_tab(notebook)

    def _build_server_tab(self, notebook: ttk.Notebook) -> None:
        """Where to run on a GPU server. Saved per user (remote.SETTINGS_FILE), not in the experiment YAML."""
        srv = ttk.Frame(notebook, padding=10)
        srv.columnconfigure(1, weight=1)
        notebook.add(srv, text="GPU server")
        self.server_vars: dict[str, tk.StringVar] = {}
        for row, (label, key, hint) in enumerate([
            ("Server", "host", "user@host, or a name from ~/.ssh/config"),
            ("Port", "port", "empty = 22"),
            ("Key file", "key_file", "empty = your default SSH key"),
            ("Jump host", "jump_host", "optional: login node to go through"),
            ("Folder on server", "folder", "relative = in your home folder"),
            ("Python on server", "python", "with vame-motifs installed"),
            ("Run prefix", "prefix", "optional, e.g. srun --gres=gpu:1"),
        ]):
            ttk.Label(srv, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
            var = self.server_vars[key] = tk.StringVar()
            ttk.Entry(srv, textvariable=var, width=22).grid(row=row, column=1, sticky="ew", pady=2)
            ttk.Label(srv, text=hint, foreground="gray").grid(row=row, column=3, sticky="w", padx=4)
        ttk.Button(srv, text="File…", width=6, command=self._browse_key).grid(row=2, column=2, padx=2)

        self.sync_data = tk.BooleanVar(value=True)
        ttk.Checkbutton(srv, text="Copy pose files and videos to the server, and results back",
                        variable=self.sync_data).grid(row=7, column=0, columnspan=4, sticky="w", pady=(8, 0))

        bar = ttk.Frame(srv)
        bar.grid(row=8, column=0, columnspan=4, sticky="w", pady=(8, 0))
        self.connect_button = ttk.Button(bar, text="Connect", command=self.connect)
        self.connect_button.pack(side="left")
        self.disconnect_button = ttk.Button(bar, text="Disconnect", command=self.disconnect, state="disabled")
        self.disconnect_button.pack(side="left", padx=4)
        self.tunnel_label = ttk.Label(bar, text="Not connected", foreground="gray")
        self.tunnel_label.pack(side="left", padx=6)

        ttk.Label(srv, foreground="gray", wraplength=560, justify="left", text=(
            "Tick 'On GPU server' (Run box) to run the commands there. 'Connect' opens one SSH connection and "
            "keeps it open, so a password or code is asked only once; with an SSH key it is optional. "
            "Python example: ~/miniconda3/envs/vame/bin/python. Absolute paths in the experiment are "
            "taken as paths on the server.")).grid(row=9, column=0, columnspan=4, sticky="w", pady=(10, 0))

    def _build_steps(self) -> None:
        box = ttk.LabelFrame(self, text="Run", padding=10)
        box.grid(row=1, column=1, sticky="nsew")
        box.columnconfigure(0, weight=1)

        self.run_button = ttk.Button(box, text="Run all", command=lambda: self.run_command("run"))
        self.run_button.grid(row=0, column=0, columnspan=2, sticky="ew", ipady=4)
        ttk.Label(box, text=COMMANDS["run"][1], foreground="gray", wraplength=330).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(2, 8))

        self.step_buttons = [self.run_button]
        for i, name in enumerate(STEP_COMMANDS):
            button = ttk.Button(box, text=name, width=12, command=lambda n=name: self.run_command(n))
            button.grid(row=2 + i, column=0, sticky="w", pady=1)
            ttk.Label(box, text=COMMANDS[name][1], foreground="gray", wraplength=230).grid(
                row=2 + i, column=1, sticky="w", padx=6)
            self.step_buttons.append(button)

        row = 2 + len(STEP_COMMANDS)
        options = ttk.Frame(box)
        options.grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.force = tk.BooleanVar()
        self.verbose = tk.BooleanVar()
        self.run_remote = tk.BooleanVar()
        ttk.Checkbutton(options, text="Force (redo steps)", variable=self.force).pack(side="left")
        ttk.Checkbutton(options, text="Verbose", variable=self.verbose).pack(side="left", padx=10)
        ttk.Checkbutton(options, text="On GPU server", variable=self.run_remote).pack(side="left")

        actions = ttk.Frame(box)
        actions.grid(row=row + 1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        status = ttk.Button(actions, text="Status", command=lambda: self.run_command("status"))
        status.pack(side="left")
        self.step_buttons.append(status)
        self.stop_button = ttk.Button(actions, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=4)
        ttk.Button(actions, text="Open output folder", command=self.open_output).pack(side="right")
        ttk.Button(box, text="Explore results…", command=lambda: self.open_explorer("map")).grid(
            row=row + 2, column=0, columnspan=2, sticky="ew", pady=(10, 0), ipady=4)
        ttk.Label(box, text="Videos, motif tree and map, community names, GIFs", foreground="gray").grid(
            row=row + 3, column=0, columnspan=2, sticky="w")

    def _build_log(self) -> None:
        frame = ttk.LabelFrame(self, text="Output", padding=4)
        frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        self.log = tk.Text(frame, height=14, wrap="word", state="disabled", font=("Menlo", 11) if sys.platform == "darwin" else "TkFixedFont")
        self.log.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(frame, command=self.log.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scroll.set)
        self.log.tag_configure("error", foreground="#c62828")
        self.log.tag_configure("info", foreground="#1565c0")

        bottom = ttk.Frame(frame)
        bottom.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.status_label = ttk.Label(bottom, text="Ready")
        self.status_label.pack(side="left")
        ttk.Button(bottom, text="Clear", command=self.clear_log).pack(side="right")

    # ==================================================================
    # Form values
    # ==================================================================
    def set_values(self, values: dict) -> None:
        for key, value in values.items():
            if key != "exclude":
                self.vars.setdefault(key, tk.StringVar()).set(value)
        known = [name for name in (values["align_center"], values["align_direction"]) if name]
        self._set_bodyparts(known, selected=values["exclude"])
        self.dirty = False
        self._update_title()

    def get_values(self) -> dict:
        values = {key: var.get() for key, var in self.vars.items()}
        values["exclude"] = [self.exclude_list.get(i) for i in self.exclude_list.curselection()]
        return values

    def _set_bodyparts(self, bodyparts: list[str], selected: list[str]) -> None:
        """Fill the alignment choices and the exclude list. Names from the YAML stay listed even if unknown."""
        names = list(dict.fromkeys(bodyparts + list(selected)))
        for combo in self.combos:
            combo.configure(values=names)
        self.exclude_list.delete(0, "end")
        for i, name in enumerate(names):
            self.exclude_list.insert("end", name)
            if name in selected:
                self.exclude_list.selection_set(i)

    def load_bodyparts(self) -> None:
        pose = self.vars["pose_files"].get().strip()
        if not pose:
            messagebox.showinfo("Body parts", "Set the pose files first (Data tab).")
            return
        path = self._resolve(pose)
        files = [path] if path.is_file() else sorted(path.glob("*.csv")) if path.is_dir() else []
        if not files:
            messagebox.showerror("Body parts", f"No .csv files found at {path}")
            return
        try:
            bodyparts = read_bodyparts(files[0])
        except Exception as e:  # any unreadable file: show it, 'validate' gives the full check
            messagebox.showerror("Body parts", f"Cannot read {files[0].name}: {e}")
            return
        self._set_bodyparts(bodyparts, selected=self.get_values()["exclude"])
        self._write_log(f"Body parts in {files[0].name}: {', '.join(bodyparts)}\n", "info")

    # ==================================================================
    # Experiment file
    # ==================================================================
    def _base_dir(self) -> Path:
        return self.yaml_path.parent if self.yaml_path else Path.cwd()

    def _resolve(self, text: str) -> Path:
        path = Path(text).expanduser()
        return path if path.is_absolute() else self._base_dir() / path

    def _relative(self, path: str) -> str:
        """A chosen path, relative to the experiment file when it is inside its folder."""
        try:
            return str(Path(path).relative_to(self._base_dir()))
        except ValueError:
            return path

    def _browse_file(self, key: str) -> None:
        if key == "videos":
            types = [("Videos", " ".join(f"*{s}" for s in VIDEO_SUFFIXES)), ("All files", "*")]
        else:
            types = [("DeepLabCut CSV", "*.csv"), ("All files", "*")]
        path = filedialog.askopenfilename(initialdir=self._base_dir(), filetypes=types)
        if path:
            self.vars[key].set(self._relative(path))

    def _browse_folder(self, key: str) -> None:
        path = filedialog.askdirectory(initialdir=self._base_dir())
        if path:
            self.vars[key].set(self._relative(path))

    def _mark_dirty(self) -> None:
        if not self.dirty:
            self.dirty = True
            self._update_title()

    def _update_title(self) -> None:
        name = self.yaml_path.name if self.yaml_path else "untitled"
        self.root.title(f"vame-motifs {__version__} — {name}{' *' if self.dirty else ''}")
        self.path_label.configure(text=str(self.yaml_path) if self.yaml_path else "(not saved)",
                                  foreground="" if self.yaml_path else "gray")

    def _confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        answer = messagebox.askyesnocancel("Unsaved changes", "Save changes to the experiment file first?")
        if answer is None:
            return False
        return self.save() if answer else True

    def new(self) -> None:
        if self._confirm_discard():
            self.yaml_path = None
            self.set_values(DEFAULTS)
            self._set_server_folder()

    def open(self) -> None:
        if not self._confirm_discard():
            return
        path = filedialog.askopenfilename(filetypes=[("Experiment YAML", "*.yaml *.yml"), ("All files", "*")])
        if path:
            self.load(Path(path))

    def load(self, path: Path) -> None:
        try:
            with open(path) as fh:
                raw = yaml.safe_load(fh) or {}
            if not isinstance(raw, dict):
                raise ValueError("not a set of 'key: value' sections")
        except (OSError, yaml.YAMLError, ValueError) as e:
            messagebox.showerror("Open", f"Cannot read {path.name}: {e}")
            return
        self.yaml_path = path.resolve()
        self.set_values(yaml_to_form(raw))
        self._set_server_folder()
        self._write_log(f"Opened {self.yaml_path}\n", "info")

    def save(self) -> bool:
        if self.yaml_path is None:
            return self.save_as()
        with open(self.yaml_path, "w") as fh:
            yaml.safe_dump(form_to_yaml(self.get_values()), fh, sort_keys=False, default_flow_style=False)
        self.dirty = False
        self._update_title()
        return True

    def save_as(self) -> bool:
        path = filedialog.asksaveasfilename(defaultextension=".yaml", initialfile="experiment.yaml",
                                            filetypes=[("Experiment YAML", "*.yaml *.yml")])
        if not path:
            return False
        old_base = self._base_dir()
        self.yaml_path = Path(path).resolve()
        # Keep relative paths pointing at the same place from the new folder.
        for key in ("pose_files", "videos", "output"):
            text = self.vars[key].get().strip()
            if text and not Path(text).expanduser().is_absolute():
                self.vars[key].set(os.path.relpath(old_base / text, self.yaml_path.parent))
        self._set_server_folder(keep_typed=True)
        return self.save()

    # ==================================================================
    # Running commands
    # ==================================================================
    def run_command(self, command: str, extra: list[str] | None = None, on_done=None) -> None:
        """Run a vame-motifs command on the experiment. ``extra``: more command-line arguments;
        ``on_done(exit code)`` is called when it ends (the Explorer reloads its results)."""
        if self.running:
            return
        if self.dirty or self.yaml_path is None:
            if not self.save():
                return
        flags = list(extra or [])
        if self.force.get() and command in ("run", "segment"):
            flags.append("--force")
        if self.verbose.get():
            flags.append("-v")

        if self.run_remote.get():
            try:
                steps, notes = remote.remote_job(self._server_settings(), self.yaml_path, command, flags,
                                                 self.get_values())
            except remote.RemoteError as e:
                messagebox.showerror("GPU server", str(e))
                return
            self._save_server_settings()
            for note in notes:
                self._write_log(f"Note: {note}\n", "info")
        else:
            args = [sys.executable, "-m", "vame_motifs", command, "-c", str(self.yaml_path), *flags]
            steps = [remote.Step("run", f"vame-motifs {' '.join(args[3:])}", args, cwd=self.yaml_path.parent)]

        self.job_remote = self.run_remote.get()
        self.stopping = False
        self.on_done = on_done
        self._set_running(command)
        threading.Thread(target=self._run_job, args=(steps,), daemon=True).start()

    def _run_job(self, steps: list[remote.Step]) -> None:
        """Background thread: run the steps in order, forwarding their output to the window
        (Tk is not thread-safe). A failed upload ends the job; results are still copied
        back after a failed command, for its run record and log."""
        code, failed = 0, None
        for step in steps:
            if self.stopping or (step.kind == "download" and code == remote.SSH_FAILED):
                break
            self.output_queue.put(("info", f"\n$ {step.title}\n"))
            step_code = self._run_step(step)
            if step_code != 0 and failed is None:
                code, failed = step_code, step.kind
            if step_code != 0 and (step.kind == "upload" or step_code == remote.SSH_FAILED):
                break
        self.output_queue.put(("done", code, failed))

    def _run_step(self, step: remote.Step) -> int:
        if step.makedir:
            step.makedir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PYTHONUNBUFFERED="1")  # stream output as it is printed
        stdin = subprocess.PIPE if step.tty or step.stdin is not None else subprocess.DEVNULL
        try:
            process = subprocess.Popen(step.args, stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1, cwd=step.cwd, env=env)
        except OSError as e:
            self.output_queue.put(("line", f"Error: cannot start {step.args[0]}: {e}\n"))
            return 127
        if step.stdin is not None:
            try:
                process.stdin.write(step.stdin)
                process.stdin.close()  # end of file: the remote 'cat' finishes
            except BrokenPipeError:
                pass  # ssh failed before reading it: its exit code says why
        self.process = process
        for line in process.stdout:
            self.output_queue.put(("line", line))
        process.wait()
        if process.stdin:
            process.stdin.close()
        self.process = None
        return process.returncode

    def _drain_output(self) -> None:
        try:
            while True:
                item = self.output_queue.get_nowait()
                if item[0] == "done":
                    self._finished(*item[1:])
                elif item[0] == "tunnel_closed":
                    self._tunnel_closed()
                elif item[0] == "info":
                    self._write_log(item[1], "info")
                else:
                    line = item[1]
                    self._write_log(line, "error" if line.startswith(("Error:", "Traceback")) else None)
        except queue.Empty:
            pass
        self.after(100, self._drain_output)

    def _finished(self, code: int, failed: str | None) -> None:
        self.running = False
        messages = {0: "Finished", 1: "Failed: see the error above", 2: "Wrong command-line arguments",
                    127: "Program not found: see the error above"}
        if self.stopping:
            text = "Stopped"
        elif self.job_remote and code == remote.SSH_FAILED:
            text = "Cannot connect to the server: 'Connect' in the GPU server tab, or check its settings"
        elif failed == "upload":
            text = f"Copying the files to the server failed (exit code {code})"
        elif failed == "download":
            text = f"Finished on the server, but copying the results back failed (exit code {code})"
        else:
            text = messages.get(code, f"Stopped (exit code {code})")
        self._write_log(f"[{text}]\n", "info" if code == 0 else "error")
        self.status_label.configure(text=text)
        for button in self.step_buttons:
            button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        on_done, self.on_done = self.on_done, None
        if on_done is not None:
            on_done(-1 if self.stopping else code)

    def _set_running(self, command: str) -> None:
        self.running = True
        for button in self.step_buttons:
            button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        where = f" on {self.server_vars['host'].get().strip()}" if self.job_remote else ""
        self.status_label.configure(text=f"Running '{command}'{where}…")

    def stop(self) -> None:
        if self.running and messagebox.askyesno("Stop", "Stop the running command?"):
            self._stop_job()

    def _stop_job(self) -> None:
        self.stopping = True  # no further steps start
        process = self.process
        if process is not None:
            process.terminate()  # remote: closing ssh hangs up the command on the server

    # ==================================================================
    # GPU server
    # ==================================================================
    def _load_server_settings(self) -> None:
        saved = remote.settings_from_dict(self.settings["server"])
        for key, var in self.server_vars.items():
            var.set(getattr(saved, key))
        self.sync_data.set(saved.sync_data)
        self.run_remote.set(bool(self.settings["server"].get("run_on_server", False)))

    def _server_settings(self) -> remote.RemoteSettings:
        return remote.RemoteSettings(**{key: var.get() for key, var in self.server_vars.items()},
                                     sync_data=self.sync_data.get())

    def _set_server_folder(self, keep_typed: bool = False) -> None:
        """The experiment's folder on the server: the one used before, or one named after the local folder."""
        var = self.server_vars["folder"]
        if self.yaml_path is None:
            var.set("")
        elif str(self.yaml_path) in self.settings["folders"]:
            var.set(self.settings["folders"][str(self.yaml_path)])
        elif not (keep_typed and var.get().strip()):
            var.set(remote.default_folder(self.yaml_path))

    def _save_server_settings(self) -> None:
        server = {key: value for key, value in vars(self._server_settings()).items() if key != "folder"}
        server["run_on_server"] = self.run_remote.get()
        self.settings["server"] = server
        if self.yaml_path is not None and self.server_vars["folder"].get().strip():
            self.settings["folders"][str(self.yaml_path)] = self.server_vars["folder"].get().strip()
        try:
            remote.save_settings(self.settings)
        except OSError as e:
            self._write_log(f"Cannot save the server settings: {e}\n", "error")

    def _browse_key(self) -> None:
        path = filedialog.askopenfilename(initialdir=Path.home() / ".ssh")
        if path:
            self.server_vars["key_file"].set(path)

    def _askpass(self) -> str:
        """An executable that shows askpass.py's password window: SSH_ASKPASS takes a program, not a command."""
        if self.askpass_dir is None:
            self.askpass_dir = tempfile.mkdtemp(prefix="vame-motifs-")
        script = Path(self.askpass_dir) / "askpass"
        script.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -m vame_motifs.askpass "$@"\n')
        script.chmod(0o700)
        return str(script)

    def connect(self) -> None:
        """Open the SSH connection that later commands go through (see remote.py)."""
        if self.tunnel is not None:
            return
        if remote.CONTROL_PATH is None:
            messagebox.showinfo("GPU server", "Keeping a connection open is not available on Windows. "
                                              "Use an SSH key: commands then connect by themselves.")
            return
        settings = self._server_settings()
        if not settings.host.strip():
            messagebox.showerror("GPU server", "Set the server (user@host) first.")
            return
        self._save_server_settings()
        env = dict(os.environ, SSH_ASKPASS=self._askpass(), SSH_ASKPASS_REQUIRE="force")
        env.setdefault("DISPLAY", ":0")  # older OpenSSH only uses SSH_ASKPASS when DISPLAY is set
        args = remote.master_command(settings)
        self._write_log(f"\n$ {shlex.join(args)}\n", "info")
        try:
            # A new session: no terminal, so ssh asks through the password window.
            self.tunnel = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.PIPE, text=True, env=env, start_new_session=True)
        except OSError as e:
            self._write_log(f"Error: cannot start ssh: {e}\n", "error")
            return
        self.tunnel_settings, self.tunnel_ready = settings, False
        threading.Thread(target=self._read_tunnel, args=(self.tunnel,), daemon=True).start()
        self._set_tunnel_state(f"Connecting to {settings.host.strip()}…", connected=True)
        self.after(500, self._check_tunnel)

    def _read_tunnel(self, tunnel: subprocess.Popen) -> None:
        for line in tunnel.stderr:
            self.output_queue.put(("line", f"ssh: {line}"))
        tunnel.wait()
        self.output_queue.put(("tunnel_closed",))

    def _check_tunnel(self) -> None:
        """Poll until the connection is ready (after the password, if any)."""
        if self.tunnel is None or self.tunnel_ready or self.tunnel.poll() is not None:
            return
        try:
            ok = subprocess.run(remote.check_command(self.tunnel_settings), stdin=subprocess.DEVNULL,
                                capture_output=True, timeout=5, check=False).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            ok = False
        if ok:
            self.tunnel_ready = True
            self._set_tunnel_state(f"Connected to {self.tunnel_settings.host.strip()}", connected=True)
            self._write_log(f"[Connected to {self.tunnel_settings.host.strip()}]\n", "info")
        else:
            self.after(500, self._check_tunnel)

    def _tunnel_closed(self) -> None:
        if self.tunnel is None:
            return
        code = self.tunnel.wait()
        host = self.tunnel_settings.host.strip()
        if self.tunnel_ready or code in (0, -15):
            self._write_log(f"[Disconnected from {host}]\n", "info")
        else:
            self._write_log(f"[Cannot connect to {host} (exit code {code}): see the ssh lines above]\n", "error")
        self.tunnel, self.tunnel_ready = None, False
        self._set_tunnel_state("Not connected", connected=False)

    def _set_tunnel_state(self, text: str, connected: bool) -> None:
        self.tunnel_label.configure(text=text, foreground="" if connected else "gray")
        self.connect_button.configure(state="disabled" if connected else "normal")
        self.disconnect_button.configure(state="normal" if connected else "disabled")

    def disconnect(self, ask: bool = True) -> None:
        if self.tunnel is None:
            return
        if ask and self.running and self.job_remote and not messagebox.askyesno(
                "Disconnect", "A command is running on the server through this connection. Stop it and disconnect?"):
            return
        try:
            subprocess.run(remote.exit_command(self.tunnel_settings), stdin=subprocess.DEVNULL,
                           capture_output=True, timeout=5, check=False)
        except (OSError, subprocess.TimeoutExpired):
            pass
        if self.tunnel.poll() is None:
            self.tunnel.terminate()

    # ==================================================================
    # Explore results (viewer.py)
    # ==================================================================
    def open_explorer(self, tab: str = "map") -> None:
        """The 'Explore results' window, on one of its tabs (viewer.Explorer.TABS)."""
        if self.explorer is None:
            if (self.dirty or self.yaml_path is None) and not self.save():
                return
            from vame_motifs.config import load_config
            try:
                from vame_motifs import viewer  # matplotlib, OpenCV: loaded only when needed
                cfg = load_config(self.yaml_path)
            except ImportError as e:
                messagebox.showerror("Explore results", f"Missing package: {e}. Install VAME (section 1 of the README).")
                return
            except Exception as e:  # ConfigError and friends: what 'validate' would say
                messagebox.showerror("Explore results", f"Cannot read the experiment: {e}")
                return
            self.explorer = viewer.Explorer(self, cfg)
        self.explorer.show(tab)

    def open_output(self) -> None:
        folder = self._resolve(self.vars["output"].get().strip() or "outputs")
        if not folder.exists():
            messagebox.showinfo("Output", f"{folder} does not exist yet.")
            return
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        elif sys.platform == "win32":
            os.startfile(folder)  # noqa: S606  (Windows only)
        else:
            subprocess.Popen(["xdg-open", str(folder)])

    # ==================================================================
    # Log
    # ==================================================================
    def _write_log(self, text: str, tag: str | None = None) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text, (tag,) if tag else ())
        self.log.see("end")
        self.log.configure(state="disabled")

    def clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def on_close(self) -> None:
        if self.explorer is not None:
            self.explorer.close()
        if self.running:
            if not messagebox.askyesno("Quit", "A command is still running. Stop it and quit?"):
                return
            self._stop_job()
        if self._confirm_discard():
            self._save_server_settings()
            self.disconnect(ask=False)
            if self.askpass_dir:
                shutil.rmtree(self.askpass_dir, ignore_errors=True)
            self.root.destroy()


def main(argv: list[str] | None = None) -> int:
    """``vame-motifs-gui [experiment.yaml]``"""
    argv = sys.argv[1:] if argv is None else argv
    root = tk.Tk()
    App(root, Path(argv[0]) if argv else None)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
