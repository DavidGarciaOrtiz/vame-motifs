"""VAME's motif labels -> CSV files a researcher can open.

The problem this module solves
------------------------------
VAME does not label single frames. It looks at a *window* of ``time_window``
consecutive frames (30 by default, i.e. one second at 30 fps) and gives that
whole window one motif number. The window then slides forward one frame and
gets the next label, and so on. VAME saves these labels as a ``.npy`` file
(numpy's binary format), one number per window, with no frame numbers.

A researcher needs: "at frame 1234 (41.1 s), the animal is in motif 3". This
module makes that table.

VAME gives one label per sliding window of ``time_window`` frames, so a
recording of N frames gets N - time_window + 1 labels. VAME's own convention
(analysis/videowriter.py, ``cluster_start``) centres each label on its window:

    label i  <->  frame i + time_window // 2

(``//`` is whole-number division: 30 // 2 == 15.)

With time_window = 30, the first label belongs to frame 15; the first 15 and
the last 14 frames have no motif and are left empty in the CSV. Following
VAME's convention keeps these CSVs aligned with any motif videos VAME makes.

Outputs, per segmentation algorithm (hmm and/or kmeans)::

    <output>/motifs/<algorithm>/<session>_motifs.csv   frame, time_s, motif
    <output>/motifs/<algorithm>/motif_usage.csv        share of time in each motif, per session

It also writes the *run records*: a folder per command run, with a copy of
the settings and the software versions (write_run_record, at the end).

numpy (``np``) is the library for arrays of numbers; pandas (``pd``) for tables.
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
    """Place window labels on the frames they belong to; frames without a label stay empty (<NA>).

    ``labels``: one motif per window (VAME's .npy). ``n_frames``: length of the
    recording. Returns one value per frame: the motif, or <NA> (pandas' "no
    value") at the edges.

    Example with time_window 4 (offset 2) and 6 frames, labels [7, 8, 9]::

        frame:  0     1     2  3  4  5
        motif:  <NA>  <NA>  7  8  9  <NA>
    """
    offset = time_window // 2
    # More labels than fit means the pose file is not the one VAME segmented.
    if offset + len(labels) > n_frames:
        raise ExportError(
            f"{len(labels)} labels do not fit in {n_frames} frames with time_window {time_window}. "
            "Was the pose file changed after segmentation?"
        )
    # Start with every frame empty. "Int64" (capital I) is pandas' whole-number
    # type that can also hold <NA>; numpy's int cannot represent "missing".
    motif = pd.Series(pd.NA, index=range(n_frames), dtype="Int64")
    # .iloc[a : b] selects positions a to b-1; the labels are written there.
    motif.iloc[offset : offset + len(labels)] = labels
    return motif


def motif_table(labels: np.ndarray, n_frames: int, fps: float, time_window: int) -> pd.DataFrame:
    """The content of ``<session>_motifs.csv``: one row per frame with frame, time_s and motif."""
    frames = np.arange(n_frames)  # 0, 1, 2, ..., n_frames - 1
    return pd.DataFrame(
        {
            "frame": frames,
            "time_s": np.round(frames / fps, 4),  # seconds since the start, 4 decimals
            "motif": labels_to_frames(labels, n_frames, time_window),  # Int64: whole numbers, empty where unlabelled
        }
    )


def motif_usage(labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """Share of labelled frames spent in each motif (0 .. n_clusters-1)."""
    # bincount counts how often each whole number appears: [3, 0, 3, 1] ->
    # [1, 1, 0, 2] (one 0, one 1, no 2, two 3s). minlength makes sure motifs
    # that never occur still get a 0.
    counts = np.bincount(labels.astype(int), minlength=n_clusters)
    return counts / counts.sum()  # counts -> fractions that add up to 1


def export_motifs(cfg: ExperimentConfig) -> list[Path]:
    """Write one motif CSV per session and algorithm, plus a usage summary. Returns the files written."""
    pipeline = VamePipeline(cfg)  # only used to know where VAME saved the labels
    written: list[Path] = []
    for algorithm in cfg.algorithms:  # "hmm", "kmeans", or both
        out_dir = cfg.motifs_dir / algorithm
        # parents=True: also create missing parent folders; exist_ok: no error if it exists.
        out_dir.mkdir(parents=True, exist_ok=True)
        usage_rows = {}  # session name -> its usage fractions
        for pose_path in cfg.pose_files:
            session = pose_path.stem
            label_path = pipeline.label_file(session, algorithm)
            if not label_path.exists():
                raise ExportError(
                    f"No {algorithm} labels for '{session}' with n_clusters={cfg.n_clusters} "
                    f"({label_path}). Run 'segment' first."
                )
            labels = np.load(label_path)  # read VAME's .npy file into an array
            table = motif_table(labels, count_frames(pose_path), cfg.fps, cfg.time_window)
            out = out_dir / f"{session}_motifs.csv"
            table.to_csv(out, index=False)  # index=False: don't write pandas' row numbers as an extra column
            written.append(out)
            usage_rows[session] = motif_usage(labels, cfg.n_clusters)

        # One row per session, one column per motif ("motif_0", "motif_1", ...).
        # orient="index": the dictionary keys become the rows.
        usage = pd.DataFrame.from_dict(usage_rows, orient="index", columns=[f"motif_{i}" for i in range(cfg.n_clusters)])
        usage.index.name = "session"  # header of the first column
        usage_path = out_dir / "motif_usage.csv"
        usage.round(4).to_csv(usage_path)
        written.append(usage_path)
    return written


def _package_version(name: str) -> str:
    """The installed version of a Python package, or "not installed"."""
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def write_run_record(cfg: ExperimentConfig, command: str) -> Path:
    """outputs/runs/<timestamp>_<command>/ with a copy of the YAML and the package versions.

    This is what makes a result traceable months later.

    cli.py calls this before most commands, and also saves the command's log
    (run.log) in the same folder.
    """
    # Current local date and time as text, e.g. "20261009-103406".
    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    run_dir = cfg.output / "runs" / f"{stamp}_{command}"
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg.yaml_path, run_dir / "experiment.yaml")  # the exact settings used
    # versions.txt: the command line, the computer, and the versions of the
    # packages that affect the results.
    lines = [
        f"command: {' '.join(sys.argv)}",
        f"host: {socket.gethostname()}",
        f"platform: {platform.platform()}",
        f"python: {platform.python_version()}",
    ] + [f"{pkg}: {_package_version(pkg)}" for pkg in ("vame-motifs", "vame-py", "torch", "numpy", "pandas")]
    (run_dir / "versions.txt").write_text("\n".join(lines) + "\n")
    return run_dir
