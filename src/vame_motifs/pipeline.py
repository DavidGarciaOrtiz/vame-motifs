"""The VAME steps, written against vame-py 0.14.4.

Our YAML holds the researcher's decisions; this module translates them into
VAME's config and calls VAME's functions in order:

    init     create the VAME project from the pose CSVs + write our settings
    prepare  preprocessing (cleaning, egocentric alignment, outliers, smoothing)
             + training set
    train    train + evaluate the model (the only slow, GPU-worthy step)
    segment  motif segmentation (hmm and/or kmeans)

Whether a step is done is decided by the files it must produce, not by VAME's
states/states.json: VAME logs some failures and still records "success".
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

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

    # --- all steps ------------------------------------------------------
    def run(self, force: bool = False) -> None:
        """init -> prepare -> train -> segment, skipping finished steps unless force=True.

        Segmentation always runs: VAME itself skips results that already exist
        for the current n_clusters, so a new n_clusters is segmented without
        retraining.
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
