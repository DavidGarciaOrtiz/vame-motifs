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
        """
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
        """Move each session's flat motif clips into community_<i>/ subfolders of cluster_videos/."""
        motif_to_community = self._motif_to_community(algorithm)
        for pose_path in self.cfg.pose_files:
            session = pose_path.stem
            clips_dir = self.motif_videos_dir(session, algorithm)
            for clip in clips_dir.glob(f"{session}-motif_*.mp4"):
                motif = int(clip.stem.rsplit("_", 1)[-1])
                community = motif_to_community.get(motif)
                if community is None:
                    continue  # motif had no frames, so VAME never wrote a clip for it
                dest_dir = clips_dir / f"community_{community}"
                dest_dir.mkdir(exist_ok=True)
                clip.rename(dest_dir / clip.name)

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
