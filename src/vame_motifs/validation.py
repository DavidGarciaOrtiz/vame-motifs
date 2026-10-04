"""Checks on the input before any VAME step runs.

Two kinds of problem are handled differently on purpose:

- errors (unreadable file, body parts that don't match, unknown name) raise,
  because the run cannot work;
- warnings (poorly tracked body parts, very short files) are returned, because
  the run can work but the researcher should know.

Nothing here prints: the CLI decides how to show the result, and tests can
check it directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vame_motifs.config import ConfigError, ExperimentConfig
from vame_motifs.io import PoseFile, PoseFileError, read_dlc_csv

WARN_FRACTION = 0.2  # warn when more than 20 % of a body part's frames are below the threshold


@dataclass
class FileReport:
    name: str
    n_frames: int
    warnings: list[str] = field(default_factory=list)


def validate(cfg: ExperimentConfig) -> tuple[list[PoseFile], list[FileReport]]:
    """Raise on errors that make the run impossible; return per-file warnings for everything else."""
    poses = [read_dlc_csv(p) for p in cfg.pose_files]

    names = [p.name for p in poses]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise PoseFileError(f"Several pose files share the name(s) {sorted(duplicates)}; VAME needs unique session names.")

    reference = poses[0].bodyparts
    for pose in poses[1:]:
        if pose.bodyparts != reference:
            raise PoseFileError(
                f"{pose.path.name} has body parts {pose.bodyparts}, "
                f"but {poses[0].path.name} has {reference}. All files must match."
            )

    for name in (cfg.align_center, cfg.align_direction, *cfg.exclude):
        if name not in reference:
            raise ConfigError(f"Unknown body part '{name}'. Available: {', '.join(reference)}")
    if len(set(reference) - set(cfg.exclude)) < 3:
        raise ConfigError("At least 3 body parts must remain after keypoints.exclude.")

    reports = []
    for pose in poses:
        warnings = []
        if pose.n_frames <= cfg.time_window:
            warnings.append(f"only {pose.n_frames} frames: shorter than time_window ({cfg.time_window}), no motifs possible")
        frac = pose.low_confidence_fraction(cfg.min_confidence)
        for bodypart, value in frac[frac > WARN_FRACTION].items():
            warnings.append(f"{bodypart}: {value:.0%} of frames below likelihood {cfg.min_confidence}")
        reports.append(FileReport(pose.path.name, pose.n_frames, warnings))
    return poses, reports
