"""The VAME steps, written against vame-py 0.14.4.

Our YAML holds the researcher's decisions; this module translates them into
VAME's config and calls VAME's functions in order:

    init     create the VAME project from the pose CSVs + write our settings
    prepare  preprocessing (cleaning, egocentric alignment, outliers, smoothing)
             + training set
    train    train + evaluate the model (the only slow, GPU-worthy step)
    segment  motif segmentation (hmm and/or kmeans)
    videos   cut a short .mp4 per motif per session from input.videos (optional),
             grouped into per-community folders (runs community analysis first)
    umap     2-D UMAP of the latent space, for the window's motif map
    gif      vame.gif: the animal next to its path through the UMAP, as a .gif

Whether a step is done is decided by the files it must produce, not by VAME's
states/states.json: VAME logs some failures and still records "success".
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from vame_motifs.config import ExperimentConfig

logger = logging.getLogger(__name__)

VAME_VERSION = "0.14.4"
SOURCE_SOFTWARE = "DeepLabCut"
MODEL_NAME = "VAME"  # VAME's default model_name; results are saved under it


class PipelineError(RuntimeError):
    """A VAME step cannot run or did not produce its output."""


def vame_settings(cfg: ExperimentConfig) -> dict:
    """Our YAML -> VAME config.yaml keys. Everything else keeps VAME's defaults."""
    return {
        "pose_confidence": cfg.min_confidence,
        "n_clusters": cfg.n_clusters,
        "segmentation_algorithms": cfg.algorithms,
        "max_epochs": cfg.max_epochs,
        "time_window": cfg.time_window,
        "zdims": cfg.zdims,
        "project_random_state": cfg.seed,
        "all_data": "yes",  # segment every session, never ask interactively
    }


def _import_vame():
    # No display on the server: VAME's plots are saved to files, not shown.
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import vame
    except ImportError as e:
        raise PipelineError(
            f"VAME is not installed in this environment ({e}). Install it with: python -m pip install vame-py=={VAME_VERSION}"
        ) from e
    return vame


def _pose_matrix(ds) -> np.ndarray:
    """Frames x (x, y, confidence) per keypoint, in keypoint order, from VAME's pose dataset."""
    position = ds["position"].isel(individuals=0).transpose("time", "keypoints", "space").values
    confidence = ds["confidence"].isel(individuals=0).transpose("time", "keypoints").values
    return np.concatenate([position, confidence[..., None]], axis=2).reshape(len(position), -1)


class VamePipeline:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.project_path: Path = cfg.output / cfg.project_name
        self._config: dict | None = None

    # --- files each step must produce -----------------------------------
    @property
    def trainset_path(self) -> Path:
        return self.project_path / "data" / "train" / "train_seq.npy"

    @property
    def model_path(self) -> Path:
        return self.project_path / "model" / "best_model" / f"{MODEL_NAME}_{self.cfg.project_name}.pkl"

    def label_file(self, session: str, algorithm: str) -> Path:
        """Where VAME saves a session's motif labels (one label per time window)."""
        n = self.cfg.n_clusters
        return (
            self.project_path / "results" / session / MODEL_NAME / f"{algorithm}-{n}" / f"{n}_{algorithm}_label_{session}.npy"
        )

    def missing_labels(self) -> list[Path]:
        return [
            self.label_file(p.stem, algorithm)
            for algorithm in self.cfg.algorithms
            for p in self.cfg.pose_files
            if not self.label_file(p.stem, algorithm).exists()
        ]

    def _log(self, name: str) -> Path:
        return self.project_path / "logs" / f"{name}.log"

    # --- step 0-1 -------------------------------------------------------
    def init(self) -> dict:
        """Create the VAME project (or open the existing one) and apply our settings.

        init_new_project returns an existing project untouched, ignoring
        config_kwargs, so the settings are written again with update_config.
        That makes YAML edits (e.g. a new n_clusters) take effect on re-runs.
        For the same reason, an existing project never gets new recordings:
        _check_project_sessions stops early if the recordings changed.
        """
        self._check_project_sessions()
        vame = _import_vame()
        from vame.util.auxiliary import read_config, update_config

        settings = vame_settings(self.cfg)
        config_path, _ = vame.init_new_project(
            project_name=self.cfg.project_name,
            poses_estimations=[str(p) for p in self.cfg.pose_files],
            source_software=SOURCE_SOFTWARE,
            working_directory=str(self.cfg.output),
            videos=[str(v) for v in self.cfg.videos] or None,  # paired positionally with pose_files
            fps=self.cfg.fps,
            config_kwargs=dict(settings),  # a copy: VAME adds its own keys to this dict
        )
        # Keyword arguments only: VAME's state-saving decorator misreads positional ones.
        self._config = update_config(config=read_config(config_path), config_update=settings)
        return self._config

    def _check_project_sessions(self) -> None:
        """Raise if an existing VAME project was made for other recordings than the YAML's pose files.

        VAME keeps the recordings it was created with: added pose files would be
        left out of training, and segmentation (which only checks the first
        recording's results) would silently skip them.
        """
        config_path = self.project_path / "config.yaml"
        if not config_path.exists():
            return  # a new project: VAME creates it with these pose files
        import yaml

        with open(config_path) as fh:
            in_project = set((yaml.safe_load(fh) or {}).get("session_names") or [])
        in_yaml = {p.stem for p in self.cfg.pose_files}
        if in_project == in_yaml:
            return
        changes = []
        if in_yaml - in_project:
            changes.append(f"new: {', '.join(sorted(in_yaml - in_project))}")
        if in_project - in_yaml:
            changes.append(f"no longer in the pose files: {', '.join(sorted(in_project - in_yaml))}")
        raise PipelineError(
            f"The VAME project in {self.project_path} was made for {len(in_project)} recording(s), but the "
            f"experiment now has {len(in_yaml)} ({'; '.join(changes)}). VAME cannot add or remove recordings "
            f"in an existing project: set a new output folder (or move {self.project_path.name} away) and run "
            "everything again."
        )

    @property
    def config(self) -> dict:
        return self._config if self._config is not None else self.init()

    # --- step 2-3 -------------------------------------------------------
    def prepare(self) -> None:
        vame = _import_vame()
        processed_variable = vame.preprocessing(
            config=self.config,
            centered_reference_keypoint=self.cfg.align_center,
            orientation_reference_keypoint=self.cfg.align_direction,
        )
        vame.create_trainset(
            config=self.config,
            read_from_variable=processed_variable,  # exactly what preprocessing produced
            keypoints_to_exclude=self.cfg.exclude or None,
        )
        if not self.trainset_path.exists():
            raise PipelineError(f"VAME did not create the training set. See {self._log('create_trainset')}")

    # --- step 4-5: the slow part ----------------------------------------
    def train(self) -> None:
        vame = _import_vame()
        if not self.trainset_path.exists():
            raise PipelineError("No training set yet. Run 'prepare' first.")
        vame.train_model(config=self.config)
        if not self.model_path.exists():
            raise PipelineError(
                f"Training finished without saving a model ({self.model_path}). "
                f"Increase advanced.max_epochs (now {self.cfg.max_epochs}) or see {self._log('train_model')}"
            )
        vame.evaluate_model(config=self.config)  # loss plots in model/evaluate/

    # --- step 6 ---------------------------------------------------------
    def segment(self, overwrite: bool = False) -> None:
        """Motif labels per session. VAME skips an algorithm/n_clusters pair it already has."""
        vame = _import_vame()
        if not self.model_path.exists():
            raise PipelineError("No trained model yet. Run 'train' first.")
        vame.segment_session(config=self.config, overwrite_segmentation=overwrite)
        missing = self.missing_labels()
        if missing:
            raise PipelineError(f"VAME segmentation did not produce {missing[0]}. See {self._log('pose_segmentation')}")

    # --- step 7: communities (groups of motifs with similar transitions) -
    def community_bag_file(self, algorithm: str) -> Path:
        """Where VAME saves the motif -> community grouping (one list of motif indices per community)."""
        n = self.cfg.n_clusters
        return self.project_path / "results" / "community_cohort" / f"{algorithm}-{n}" / "cohort_community_bag.npy"

    def has_community(self) -> bool:
        return all(self.community_bag_file(algorithm).exists() for algorithm in self.cfg.algorithms)

    def community(self) -> None:
        """Group motifs with similar transitions into higher-level communities, across all sessions."""
        if self.missing_labels():
            raise PipelineError("No motif labels yet. Run 'segment' first.")
        vame = _import_vame()
        vame.community(config=self.config, cut_tree=self.cfg.community_cut_tree)
        missing = [a for a in self.cfg.algorithms if not self.community_bag_file(a).exists()]
        if missing:
            raise PipelineError(f"VAME did not create a community grouping for '{missing[0]}'. See {self._log('community')}")
        for algorithm in self.cfg.algorithms:  # clips cut earlier: into the folders of the new communities
            self._group_videos_by_community(algorithm)

    def _motif_to_community(self, algorithm: str) -> dict[int, int]:
        bag = np.load(self.community_bag_file(algorithm), allow_pickle=True)
        return {int(motif): community for community, motifs in enumerate(bag) for motif in motifs}

    # --- step 8: motif videos (optional; needs input.videos) ------------
    def motif_videos_dir(self, session: str, algorithm: str) -> Path:
        """Where VAME saves a session's motif clips."""
        n = self.cfg.n_clusters
        return self.project_path / "results" / session / MODEL_NAME / f"{algorithm}-{n}" / "cluster_videos"

    def has_motif_videos(self) -> bool:
        return all(
            any(self.motif_videos_dir(p.stem, algorithm).glob("**/*.mp4"))
            for algorithm in self.cfg.algorithms
            for p in self.cfg.pose_files
        )

    def _group_videos_by_community(self, algorithm: str) -> None:
        """Move each session's motif clips into community_<i>/ subfolders of cluster_videos/.

        Clips already in a community folder are moved again if a new cut changed
        their community; folders left empty are removed.
        """
        motif_to_community = self._motif_to_community(algorithm)
        for pose_path in self.cfg.pose_files:
            session = pose_path.stem
            clips_dir = self.motif_videos_dir(session, algorithm)
            for clip in list(clips_dir.glob(f"**/{session}-motif_*.mp4")):
                motif = int(clip.stem.rsplit("_", 1)[-1])
                community = motif_to_community.get(motif)
                if community is None:
                    continue  # motif had no frames, so VAME never wrote a clip for it
                dest_dir = clips_dir / f"community_{community}"
                if clip.parent != dest_dir:
                    dest_dir.mkdir(exist_ok=True)
                    clip.rename(dest_dir / clip.name)
            for folder in clips_dir.glob("community_*"):
                if folder.is_dir() and not any(folder.iterdir()):
                    folder.rmdir()

    def motif_clips(self, session: str, algorithm: str) -> dict[int, Path]:
        """motif -> its clip for this session, wherever it is under cluster_videos/."""
        return {
            int(clip.stem.rsplit("_", 1)[-1]): clip
            for clip in self.motif_videos_dir(session, algorithm).glob(f"**/{session}-motif_*.mp4")
        }

    def motif_videos(self) -> None:
        """One short .mp4 per motif, per session, cut from input.videos and grouped by community."""
        if not self.cfg.videos:
            raise PipelineError("No raw videos configured (input.videos). Motif videos need the original recordings.")
        if self.missing_labels():
            raise PipelineError("No motif labels yet. Run 'segment' first.")
        if not self.has_community():
            self.community()
        vame = _import_vame()
        vame.motif_videos(config=self.config)
        for algorithm in self.cfg.algorithms:
            self._group_videos_by_community(algorithm)
        if not self.has_motif_videos():
            raise PipelineError(f"VAME did not create any motif videos. See {self._log('motif_videos')}")

    # --- motif map: UMAP of the latent space (for the window) ----------
    def latent_file(self, session: str) -> Path:
        """One latent vector per time window, saved by 'segment'."""
        return self.project_path / "results" / session / MODEL_NAME / "latent_vectors.npy"

    @property
    def umap_path(self) -> Path:
        return self.project_path / "results" / "umap_embedding.npz"

    def has_umap(self) -> bool:
        """The map exists and was made from the current model (retraining makes it stale)."""
        return (
            self.umap_path.exists()
            and self.model_path.exists()
            and self.umap_path.stat().st_mtime >= self.model_path.stat().st_mtime
        )

    def umap(self) -> Path:
        """2-D UMAP of the latent vectors of all sessions, sampled to VAME's num_points.

        Saved as umap_embedding.npz: ``embedding`` (points x 2) and, per point,
        ``session`` (index into ``sessions``) and ``window`` (index into that
        session's motif labels). Labels are not stored: motifs and communities
        change with n_clusters and the cut, the map does not.
        """
        missing = [self.latent_file(p.stem) for p in self.cfg.pose_files if not self.latent_file(p.stem).exists()]
        if missing:
            raise PipelineError(f"No latent vectors at {missing[0]}. Run 'segment' first.")
        import umap  # installed with VAME; imported here because it is slow to load

        sessions = [p.stem for p in self.cfg.pose_files]
        counts = [len(np.load(self.latent_file(s), mmap_mode="r")) for s in sessions]
        session_of = np.repeat(np.arange(len(sessions)), counts)
        window_of = np.concatenate([np.arange(n) for n in counts])
        n_points = min(int(self.config.get("num_points", 30_000)), len(session_of))
        rng = np.random.default_rng(self.cfg.seed)
        picked = np.sort(rng.choice(len(session_of), size=n_points, replace=False))
        latent = np.concatenate([np.load(self.latent_file(s)) for s in sessions])[picked]

        logger.info("UMAP of %d of %d time windows...", n_points, len(session_of))
        reducer = umap.UMAP(
            n_components=2,
            min_dist=self.config.get("min_dist", 0.1),
            n_neighbors=self.config.get("n_neighbors", 200),
            random_state=self.cfg.seed,
        )
        embedding = reducer.fit_transform(latent)
        np.savez(self.umap_path, embedding=embedding, session=session_of[picked], window=window_of[picked],
                 sessions=np.array(sessions))
        return self.umap_path

    # --- vame.gif: the animal and its path through the UMAP -------------
    def _well_tracked_start(self, confidence: np.ndarray, num_points: int, length: int) -> int:
        """A random first window (from the seed) among those whose frames have both alignment body
        parts tracked above min_confidence; VAME crops around them, so poor tracking shows no animal."""
        good = (confidence > self.cfg.min_confidence).all(axis=1).astype(float)
        lag = self.cfg.time_window // 2  # window i is shown on frame i + lag
        good = np.concatenate([[0.0], np.cumsum(good[lag:lag + num_points])])
        share = (good[length:] - good[:-length]) / length  # share of good frames from each start
        share = share[: num_points - length + 1]
        candidates = np.flatnonzero(share >= min(0.95, share.max()))
        return int(np.random.default_rng(self.cfg.seed).choice(candidates))

    def _link_raw_video(self, session: str) -> Path:
        """Make sure VAME's data/raw/<session><suffix> opens: it is where VAME reads the frames.

        VAME's init links it to input.videos. A project trained on the GPU server
        and copied back keeps a link to the server's path, which does not open
        here: it is pointed again at this computer's input.videos.
        """
        video = self.cfg.videos[[p.stem for p in self.cfg.pose_files].index(session)]
        raw = self.project_path / "data" / "raw" / f"{session}{video.suffix.lower()}"
        if not raw.exists():  # missing, or a link to a path that is not on this computer
            if not video.exists():
                raise PipelineError(f"input.videos: {video} does not exist")
            if raw.is_symlink():
                raw.unlink()
            raw.symlink_to(video.resolve())
        return video

    def gif(self, session: str | None = None, algorithm: str | None = None, start: int | None = None,
            length: int = 500, label: str = "community", subtract_background: bool = False,
            max_lag: int = 30, crop_size: tuple[int, int] = (300, 300)) -> Path:
        """What vame.gif() makes, for one session: per frame, the aligned, cropped animal next
        to the UMAP of its latent space with the last ``max_lag`` steps drawn as a path. The
        frames are joined into ``<output>/gifs/<session>_<algorithm>_<label>_<start>-<end>.gif``.

        vame.gif() in vame-py 0.14.4 cannot be called as is: it loads
        ``latent_vector_<session>.npy``, which segment_session no longer writes
        (it is ``latent_vectors.npy`` now), and it runs every session. So this
        writes the per-session embedding vame.gif expects, then calls the two
        functions vame.gif is made of (get_animal_frames, create_video).

        ``label``: colour the UMAP points by 'motif', 'community' or 'none'.
        ``start``: first time window; random (from the seed) if not given.
        """
        _import_vame()
        import matplotlib.pyplot as plt
        import umap
        from PIL import Image
        from vame.analysis.gif_creator import create_video
        from vame.io.load_poses import load_vame_dataset
        from vame.util import gif_pose_helper

        sessions = [p.stem for p in self.cfg.pose_files]
        session = session or sessions[0]
        if session not in sessions:
            raise PipelineError(f"Unknown session '{session}'. Sessions: {', '.join(sessions)}")
        algorithm = algorithm or self.cfg.algorithms[0]
        if algorithm not in self.cfg.algorithms:
            raise PipelineError(f"Segmentation '{algorithm}' is not in this experiment ({', '.join(self.cfg.algorithms)})")
        if label not in ("motif", "community", "none"):
            raise PipelineError(f"label must be motif, community or none (got '{label}')")
        if not self.cfg.videos:
            raise PipelineError("No raw videos configured (input.videos). The gif shows frames of the original recording.")
        if not self.label_file(session, algorithm).exists():
            raise PipelineError("No motif labels yet. Run 'segment' first.")
        if label == "community" and not self.has_community():
            self.community()

        results = self.label_file(session, algorithm).parent
        (results / "community").mkdir(exist_ok=True)
        embed_path = results / "community" / f"umap_embedding_{session}.npy"
        num_points = int(self.config.get("num_points", 30_000))
        if embed_path.exists():
            embed = np.load(embed_path)
        else:
            latent = np.load(self.latent_file(session))[:num_points]
            logger.info("UMAP of %d time windows of %s...", len(latent), session)
            embed = umap.UMAP(n_components=2, min_dist=self.config.get("min_dist", 0.1),
                              n_neighbors=self.config.get("n_neighbors", 200),
                              random_state=self.cfg.seed).fit_transform(latent)
            np.save(embed_path, embed)
        num_points = min(num_points, len(embed))

        if length < 2:
            raise PipelineError("The gif needs at least 2 frames (length)")
        if length > num_points:
            raise PipelineError(f"length {length} is longer than the {num_points} time windows of the map")
        # Alignment body parts, by their position in VAME's copy of the pose file.
        ds = load_vame_dataset(ds_path=str(self.project_path / "data" / "raw" / f"{session}.nc"))
        pose = _pose_matrix(ds)
        keypoints = [str(k) for k in ds.keypoints.values]
        pose_ref_index = [keypoints.index(self.cfg.align_center), keypoints.index(self.cfg.align_direction)]

        if start is None:
            start = self._well_tracked_start(pose[:, [3 * i + 2 for i in pose_ref_index]], num_points, length)
        if not 0 <= start <= num_points - length:
            raise PipelineError(f"start must be between 0 and {num_points - length} for length {length}")

        if label == "motif":
            colours = np.load(self.label_file(session, algorithm))
        elif label == "community":
            colours = np.load(results / "community" / f"cohort_community_label_{session}.npy")
        else:
            colours = None
        video = self._link_raw_video(session)

        frames_dir = results / "gif_frames"  # where create_video writes its PNGs
        frames_dir.mkdir(exist_ok=True)
        # get_animal_frames reads the poses with VAME's read_pose_estimation_file, whose matrix
        # has the columns mixed up (vame-py 0.14.4 flattens (space, keypoints) as (keypoints,
        # space)), so it would crop around the wrong points. It is given the right matrix.
        vame_reader = gif_pose_helper.read_pose_estimation_file
        gif_pose_helper.read_pose_estimation_file = lambda file_path, **_: (None, pose.copy(), ds)
        try:
            frames = gif_pose_helper.get_animal_frames(self.config, session, pose_ref_index, start, length,
                                                       subtract_background, video.suffix.lower(), crop_size)
            if len(frames) < length:
                raise PipelineError(f"Only {len(frames)} of {length} video frames could be read from {video.name}")
            create_video(str(results), session, embed, colours, frames, start, length, max_lag, num_points)
            plt.close("all")
            images = [Image.open(frames_dir / f"{session}gif_{i}.png").convert("RGB") for i in range(length)]
            out_dir = self.cfg.output / "gifs"
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / f"{session}_{algorithm}_{label}_{start}-{start + length}.gif"
            images[0].save(out, save_all=True, append_images=images[1:], loop=0,
                           duration=round(1000 / self.cfg.fps))
        finally:
            gif_pose_helper.read_pose_estimation_file = vame_reader
            for png in frames_dir.glob("*.png"):
                png.unlink()
            frames_dir.rmdir()
        return out

    # --- all steps ------------------------------------------------------
    def run(self, force: bool = False) -> None:
        """init -> prepare -> train -> segment -> videos, skipping finished steps unless force=True.

        Segmentation always runs: VAME itself skips results that already exist
        for the current n_clusters, so a new n_clusters is segmented without
        retraining. Motif videos (and the community grouping they need) only
        run when input.videos was set.
        """
        self.init()
        if force or not self.trainset_path.exists():
            self.prepare()
        else:
            logger.info("prepare: already done, skipping (use --force to redo)")
        if force or not self.model_path.exists():
            self.train()
        else:
            logger.info("train: already done, skipping (use --force to retrain)")
        self.segment(overwrite=force)
        if self.cfg.videos:
            if force or not self.has_motif_videos():
                self.motif_videos()
            else:
                logger.info("motif videos: already done, skipping (use --force to redo)")
