"""Checks on the input before any VAME step runs.

Why this exists: VAME's steps take minutes to hours. A typo in a body part
name, or one recording tracked with different body parts, would otherwise
only show up deep inside VAME, with a confusing message. These checks take
seconds and say exactly what is wrong.

Two kinds of problem are handled differently on purpose:

- errors (unreadable file, body parts that don't match, unknown name) raise,
  because the run cannot work;
- warnings (poorly tracked body parts, very short files) are returned, because
  the run can work but the researcher should know.

"Raise" means: stop with an exception (Python's way of signalling an error).
The command (cli.py) catches it and prints a one-line ``Error:`` message.

Nothing here prints: the CLI decides how to show the result, and tests can
check it directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vame_motifs.config import ConfigError, ExperimentConfig
from vame_motifs.io import PoseFile, PoseFileError, read_dlc_csv

WARN_FRACTION = 0.2  # warn when more than 20 % of a body part's frames are below the threshold


# @dataclass writes the boring parts of a class for us (the __init__ that
# stores each field, a readable print-out...). A FileReport is just a record
# of three values about one pose file.
@dataclass
class FileReport:
    """The result of checking one pose file: its name, its length and any warnings."""

    name: str  # file name, e.g. "rat01DLC_resnet50.csv"
    n_frames: int  # number of video frames (rows) in the file
    # field(default_factory=list) gives every report its *own* empty list;
    # a plain "= []" would make all reports share one list.
    warnings: list[str] = field(default_factory=list)


def validate(cfg: ExperimentConfig) -> tuple[list[PoseFile], list[FileReport]]:
    """Raise on errors that make the run impossible; return per-file warnings for everything else.

    ``cfg`` is the experiment settings (config.load_config). Returns two lists,
    in the order of ``cfg.pose_files``: the files as read (io.PoseFile), and one
    FileReport per file.
    """
    # 1. Read every file. read_dlc_csv raises PoseFileError for a file that is
    #    not a valid single-animal DeepLabCut CSV.
    poses = [read_dlc_csv(p) for p in cfg.pose_files]

    # 2. Session names (file names without .csv) must be unique: VAME stores
    #    each session's results in a folder named after it.
    names = [p.name for p in poses]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise PoseFileError(f"Several pose files share the name(s) {sorted(duplicates)}; VAME needs unique session names.")

    # 3. All files must track the same body parts, in the same order: VAME
    #    trains one model on all of them, so the columns must mean the same.
    reference = poses[0].bodyparts
    for pose in poses[1:]:
        if pose.bodyparts != reference:
            raise PoseFileError(
                f"{pose.path.name} has body parts {pose.bodyparts}, "
                f"but {poses[0].path.name} has {reference}. All files must match."
            )

    # 4. Every body part named in the YAML must exist in the files (catches
    #    typos such as "Snout" vs "snout").
    for name in (cfg.align_center, cfg.align_direction, *cfg.exclude):
        if name not in reference:
            raise ConfigError(f"Unknown body part '{name}'. Available: {', '.join(reference)}")
    # VAME needs a few points to describe a posture; set() difference = the
    # body parts that are kept after removing the excluded ones.
    if len(set(reference) - set(cfg.exclude)) < 3:
        raise ConfigError("At least 3 body parts must remain after keypoints.exclude.")

    # 5. Warnings, per file. These do not stop the run.
    reports = []
    for pose in poses:
        warnings = []
        # VAME labels windows of time_window frames; a shorter file has none.
        if pose.n_frames <= cfg.time_window:
            warnings.append(f"only {pose.n_frames} frames: shorter than time_window ({cfg.time_window}), no motifs possible")
        # Share of frames, per body part, that VAME will discard (likelihood
        # below min_confidence) and fill in by interpolation. frac is a pandas
        # Series: body part name -> fraction between 0 and 1.
        frac = pose.low_confidence_fraction(cfg.min_confidence)
        # frac[frac > WARN_FRACTION] keeps only the body parts above 20 %.
        # {value:.0%} formats 0.6 as "60%".
        for bodypart, value in frac[frac > WARN_FRACTION].items():
            warnings.append(f"{bodypart}: {value:.0%} of frames below likelihood {cfg.min_confidence}")
        reports.append(FileReport(pose.path.name, pose.n_frames, warnings))
    return poses, reports
