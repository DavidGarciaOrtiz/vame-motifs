"""Reading DeepLabCut pose files.

A single-animal DeepLabCut CSV has three header rows, which pandas reads as
three-level column names (scorer, bodypart, coord)::

    scorer,    DLC_resnet50_..., DLC_resnet50_..., DLC_resnet50_..., ...
    bodyparts, snout,            snout,            snout,            ...
    coords,    x,                y,                likelihood,       ...
    0,         312.4,            201.7,            0.998,            ...

The body part names come from that header, so neither DeepLabCut nor its
project folder is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

HEADER_ROWS = ["scorer", "bodyparts", "coords"]
COORDS = {"x", "y", "likelihood"}


class PoseFileError(ValueError):
    """A pose file is missing, malformed, or not in the supported format."""


@dataclass
class PoseFile:
    path: Path
    data: pd.DataFrame
    bodyparts: list[str]

    @property
    def name(self) -> str:
        """Session name, as VAME will call it (the file name without .csv)."""
        return self.path.stem

    @property
    def n_frames(self) -> int:
        return len(self.data)

    def likelihood(self) -> pd.DataFrame:
        """One likelihood column per body part (frames x bodyparts)."""
        lk = self.data.xs("likelihood", axis=1, level=2)
        lk.columns = lk.columns.droplevel(0)  # drop the scorer level, keep body part names
        return lk

    def low_confidence_fraction(self, threshold: float) -> pd.Series:
        """Share of frames below the threshold, per body part.

        The mean of a True/False column is the fraction of True values, i.e.
        the share of points VAME will discard and fill in at that threshold.
        """
        return (self.likelihood() < threshold).mean()


def _header_labels(path: Path) -> list[str]:
    """First cell of the first rows: tells DLC single-animal, multi-animal and other CSVs apart."""
    try:
        first_col = pd.read_csv(path, header=None, nrows=4, usecols=[0])[0]
    except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as e:
        raise PoseFileError(f"{path.name}: cannot be read as CSV ({e})") from e
    return [str(v) for v in first_col.tolist()]


def read_dlc_csv(path: str | Path) -> PoseFile:
    """Read one single-animal DeepLabCut CSV into a PoseFile."""
    path = Path(path)
    if not path.is_file():
        raise PoseFileError(f"File not found: {path}")

    # Peek at the header rows before parsing the whole file, so a wrong file
    # gets a clear message instead of a confusing pandas error.
    labels = _header_labels(path)
    if "individuals" in labels:
        raise PoseFileError(f"{path.name}: multi-animal DeepLabCut files are not supported.")
    if labels[:3] != HEADER_ROWS:
        raise PoseFileError(f"{path.name}: does not look like a DeepLabCut CSV (header rows: {labels[:3]}).")

    data = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    if data.empty:
        raise PoseFileError(f"{path.name}: has a header but no frames.")
    unknown = set(data.columns.get_level_values(2)) - COORDS
    if unknown:
        raise PoseFileError(f"{path.name}: unexpected coordinate columns {sorted(unknown)} (expected x, y, likelihood).")
    if not all(pd.api.types.is_numeric_dtype(t) for t in data.dtypes):
        raise PoseFileError(f"{path.name}: contains non-numeric values.")

    bodyparts = list(dict.fromkeys(data.columns.get_level_values(1)))  # unique, original order
    return PoseFile(path=path, data=data, bodyparts=bodyparts)


def count_frames(path: str | Path) -> int:
    """Number of frames in a DLC CSV, without parsing it (rows minus the 3 header rows)."""
    with open(path, "rb") as fh:
        n_lines = sum(1 for line in fh if line.strip())
    return n_lines - len(HEADER_ROWS)
