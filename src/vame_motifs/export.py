"""VAME's motif labels -> CSV files a researcher can open.

VAME gives one label per sliding window of ``time_window`` frames, so a
recording of N frames gets N - time_window + 1 labels. VAME's own convention
(analysis/videowriter.py, ``cluster_start``) centres each label on its window:

    label i  <->  frame i + time_window // 2

With time_window = 30, the first label belongs to frame 15; the first 15 and
the last 14 frames have no motif and are left empty in the CSV. Following
VAME's convention keeps these CSVs aligned with any motif videos VAME makes.

Outputs, per segmentation algorithm (hmm and/or kmeans)::

    <output>/motifs/<algorithm>/<session>_motifs.csv   frame, time_s, motif
    <output>/motifs/<algorithm>/motif_usage.csv        share of time in each motif, per session
"""

from __future__ import annotations

import datetime as dt
import platform
import shutil
import socket
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from vame_motifs.config import ExperimentConfig
from vame_motifs.io import count_frames
from vame_motifs.pipeline import VamePipeline


class ExportError(RuntimeError):
    """Motif labels are missing or don't match their pose file."""


def labels_to_frames(labels: np.ndarray, n_frames: int, time_window: int) -> pd.Series:
    """Place window labels on the frames they belong to; frames without a label stay empty (<NA>)."""
    offset = time_window // 2
    if offset + len(labels) > n_frames:
        raise ExportError(
            f"{len(labels)} labels do not fit in {n_frames} frames with time_window {time_window}. "
            "Was the pose file changed after segmentation?"
        )
    motif = pd.Series(pd.NA, index=range(n_frames), dtype="Int64")
    motif.iloc[offset : offset + len(labels)] = labels
    return motif


def motif_table(labels: np.ndarray, n_frames: int, fps: float, time_window: int) -> pd.DataFrame:
    frames = np.arange(n_frames)
    return pd.DataFrame(
        {
            "frame": frames,
            "time_s": np.round(frames / fps, 4),
            "motif": labels_to_frames(labels, n_frames, time_window),  # Int64: whole numbers, empty where unlabelled
        }
    )


def motif_usage(labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """Share of labelled frames spent in each motif (0 .. n_clusters-1)."""
    counts = np.bincount(labels.astype(int), minlength=n_clusters)
    return counts / counts.sum()


def export_motifs(cfg: ExperimentConfig) -> list[Path]:
    """Write one motif CSV per session and algorithm, plus a usage summary. Returns the files written."""
    pipeline = VamePipeline(cfg)
    written: list[Path] = []
    for algorithm in cfg.algorithms:
        out_dir = cfg.motifs_dir / algorithm
        out_dir.mkdir(parents=True, exist_ok=True)
        usage_rows = {}
        for pose_path in cfg.pose_files:
            session = pose_path.stem
            label_path = pipeline.label_file(session, algorithm)
            if not label_path.exists():
                raise ExportError(
                    f"No {algorithm} labels for '{session}' with n_clusters={cfg.n_clusters} "
                    f"({label_path}). Run 'segment' first."
                )
            labels = np.load(label_path)
            table = motif_table(labels, count_frames(pose_path), cfg.fps, cfg.time_window)
            out = out_dir / f"{session}_motifs.csv"
            table.to_csv(out, index=False)
            written.append(out)
            usage_rows[session] = motif_usage(labels, cfg.n_clusters)

        usage = pd.DataFrame.from_dict(usage_rows, orient="index", columns=[f"motif_{i}" for i in range(cfg.n_clusters)])
        usage.index.name = "session"
        usage_path = out_dir / "motif_usage.csv"
        usage.round(4).to_csv(usage_path)
        written.append(usage_path)
    return written


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def write_run_record(cfg: ExperimentConfig, command: str) -> Path:
    """outputs/runs/<timestamp>_<command>/ with a copy of the YAML and the package versions.

    This is what makes a result traceable months later.
    """
    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    run_dir = cfg.output / "runs" / f"{stamp}_{command}"
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg.yaml_path, run_dir / "experiment.yaml")
    lines = [
        f"command: {' '.join(sys.argv)}",
        f"host: {socket.gethostname()}",
        f"platform: {platform.platform()}",
        f"python: {platform.python_version()}",
    ] + [f"{pkg}: {_package_version(pkg)}" for pkg in ("vame-motifs", "vame-py", "torch", "numpy", "pandas")]
    (run_dir / "versions.txt").write_text("\n".join(lines) + "\n")
    return run_dir
