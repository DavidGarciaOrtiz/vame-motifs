"""Reading DeepLabCut pose files.

What a DeepLabCut (DLC) pose file is
------------------------------------
DLC tracks body parts in a video and saves, for every frame, where each body
part is (x, y in pixels) and how sure it is (likelihood, 0 to 1). The ``.csv``
has one row per video frame.

A single-animal DeepLabCut CSV has three header rows, which pandas reads as
three-level column names (scorer, bodypart, coord)::

    scorer,    DLC_resnet50_..., DLC_resnet50_..., DLC_resnet50_..., ...
    bodyparts, snout,            snout,            snout,            ...
    coords,    x,                y,                likelihood,       ...
    0,         312.4,            201.7,            0.998,            ...

- "scorer" is the name of the DLC model that made the file (the same in every column);
- "bodyparts" says which body part a column belongs to;
- "coords" says which of the three values it is.

So one column is addressed with three names, e.g.
``("DLC_resnet50_...", "snout", "x")``. pandas calls this a *MultiIndex*.
The first column (0, 1, 2...) is the frame number.

The body part names come from that header, so neither DeepLabCut nor its
project folder is needed.

pandas (imported as ``pd``) is the standard Python library for tables; a
table is a ``DataFrame``, a single column is a ``Series``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# What the first cell of the three header rows must say in a single-animal file.
HEADER_ROWS = ["scorer", "bodyparts", "coords"]
# The only allowed values on the third header row. A set: order does not matter.
COORDS = {"x", "y", "likelihood"}


# Our own error type. Inheriting from ValueError makes it a normal Python
# error; having our own name lets cli.py recognise it and print it as a short
# message instead of a long traceback.
class PoseFileError(ValueError):
    """A pose file is missing, malformed, or not in the supported format."""


@dataclass
class PoseFile:
    """One pose file, read into memory.

    ``path``: where it was read from. ``data``: the whole table (frames x
    columns, with the three-level column names above). ``bodyparts``: the body
    part names, in the order they appear in the file.
    """

    path: Path
    data: pd.DataFrame
    bodyparts: list[str]

    # @property lets a method be used like a value: pose.name instead of pose.name()
    @property
    def name(self) -> str:
        """Session name, as VAME will call it (the file name without .csv)."""
        return self.path.stem  # Path("a/rat01.csv").stem == "rat01"

    @property
    def n_frames(self) -> int:
        """Number of video frames: one row per frame."""
        return len(self.data)

    def likelihood(self) -> pd.DataFrame:
        """One likelihood column per body part (frames x bodyparts)."""
        # xs ("cross-section") keeps only the columns whose 3rd header level
        # (level=2, counting from 0) is "likelihood".
        lk = self.data.xs("likelihood", axis=1, level=2)
        lk.columns = lk.columns.droplevel(0)  # drop the scorer level, keep body part names
        return lk

    def low_confidence_fraction(self, threshold: float) -> pd.Series:
        """Share of frames below the threshold, per body part.

        The mean of a True/False column is the fraction of True values, i.e.
        the share of points VAME will discard and fill in at that threshold.
        """
        # (table < threshold) gives a table of True/False; .mean() averages
        # each column, and True counts as 1, False as 0.
        return (self.likelihood() < threshold).mean()


def _header_labels(path: Path) -> list[str]:
    """First cell of the first rows: tells DLC single-animal, multi-animal and other CSVs apart.

    (A leading underscore in a name means "only used inside this module".)
    """
    try:
        # Read only the first column (usecols=[0]) of the first 4 rows: fast,
        # even for a huge file. header=None: treat every row as data.
        first_col = pd.read_csv(path, header=None, nrows=4, usecols=[0])[0]
    except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as e:
        # "raise ... from e" keeps the original error attached, for debugging.
        raise PoseFileError(f"{path.name}: cannot be read as CSV ({e})") from e
    return [str(v) for v in first_col.tolist()]


def read_dlc_csv(path: str | Path) -> PoseFile:
    """Read one single-animal DeepLabCut CSV into a PoseFile.

    Raises PoseFileError, with a message a researcher can act on, when the file
    is missing, is a multi-animal file, is not a DLC file, or has bad values.
    """
    path = Path(path)  # accept a plain string too
    if not path.is_file():
        raise PoseFileError(f"File not found: {path}")

    # Peek at the header rows before parsing the whole file, so a wrong file
    # gets a clear message instead of a confusing pandas error.
    labels = _header_labels(path)
    if "individuals" in labels:  # multi-animal DLC files have an extra "individuals" header row
        raise PoseFileError(f"{path.name}: multi-animal DeepLabCut files are not supported.")
    if labels[:3] != HEADER_ROWS:
        raise PoseFileError(f"{path.name}: does not look like a DeepLabCut CSV (header rows: {labels[:3]}).")

    # header=[0, 1, 2]: the first three rows together name each column.
    # index_col=0: the first column (frame number) labels the rows.
    data = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    if data.empty:
        raise PoseFileError(f"{path.name}: has a header but no frames.")
    # Values found on the third header row that are not x, y or likelihood.
    unknown = set(data.columns.get_level_values(2)) - COORDS
    if unknown:
        raise PoseFileError(f"{path.name}: unexpected coordinate columns {sorted(unknown)} (expected x, y, likelihood).")
    # Every column must hold numbers (a stray word would break VAME later).
    if not all(pd.api.types.is_numeric_dtype(t) for t in data.dtypes):
        raise PoseFileError(f"{path.name}: contains non-numeric values.")

    # Each body part name appears 3 times (x, y, likelihood). dict.fromkeys
    # removes repeats while keeping the first-seen order (a set would not).
    bodyparts = list(dict.fromkeys(data.columns.get_level_values(1)))  # unique, original order
    return PoseFile(path=path, data=data, bodyparts=bodyparts)


def count_frames(path: str | Path) -> int:
    """Number of frames in a DLC CSV, without parsing it (rows minus the 3 header rows).

    Much faster than read_dlc_csv when only the length is needed: the lines
    are counted, not converted to numbers.
    """
    # "rb": read as raw bytes, the fastest way to go through the lines.
    # "with" closes the file automatically at the end of the block.
    with open(path, "rb") as fh:
        # Count the non-empty lines (a trailing empty line is not a frame).
        n_lines = sum(1 for line in fh if line.strip())
    return n_lines - len(HEADER_ROWS)
