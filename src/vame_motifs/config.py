"""Loading and checking the experiment YAML.

One YAML file describes one experiment. It only holds the decisions that
belong to the researcher (which files, frame rate, reference body parts,
number of motifs). Everything VAME can work out by itself (body part names,
session names, number of features) is never asked for.

All paths in the YAML are relative to the YAML file itself, so the command
works from any folder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

MIN_EPOCHS = 10
METHODS = {"hmm": ["hmm"], "kmeans": ["kmeans"], "both": ["hmm", "kmeans"]}
VIDEO_SUFFIXES = (".mp4", ".avi")  # the formats VAME's own video step accepts


class ConfigError(ValueError):
    """The experiment file is missing something or has an invalid value."""


@dataclass
class ExperimentConfig:
    yaml_path: Path
    pose_files: list[Path]
    fps: float
    align_center: str
    align_direction: str
    n_clusters: int
    method: str = "hmm"
    min_confidence: float = 0.9
    exclude: list[str] = field(default_factory=list)
    videos: list[Path] = field(default_factory=list)
    output: Path = Path("outputs")
    project_name: str = "vame_project"
    # advanced: sensible defaults, most researchers never change them
    max_epochs: int = 100
    time_window: int = 30
    zdims: int = 30
    seed: int = 42

    @property
    def algorithms(self) -> list[str]:
        """VAME segmentation algorithms for the chosen method."""
        return METHODS[self.method]

    @property
    def motifs_dir(self) -> Path:
        return self.output / "motifs"


def _require(section: dict, key: str, where: str):
    if key not in section or section[key] is None:
        raise ConfigError(f"Missing required field '{where}.{key}'")
    return section[key]


def _section(raw: dict, name: str) -> dict:
    value = raw.get(name) or {}
    if not isinstance(value, dict):
        raise ConfigError(f"'{name}' must be a section (key: value lines), got {value!r}")
    return value


def _find_pose_files(pose_path: Path) -> list[Path]:
    if not pose_path.exists():
        raise ConfigError(f"input.pose_files: {pose_path} does not exist")
    if pose_path.is_file():
        return [pose_path]
    files = sorted(pose_path.glob("*.csv"))
    if not files:
        raise ConfigError(f"No .csv files found in {pose_path}")
    return files


def _find_videos(video_path: Path, pose_files: list[Path]) -> list[Path]:
    """Raw videos paired to ``pose_files``, one per file, in the same order.

    DeepLabCut names a pose CSV after the video it analysed plus its own
    suffix (``session1.mp4`` -> ``session1DLC_resnet50_....csv``), so each
    video is matched to the pose file whose name it is the longest prefix of.
    """
    if not video_path.exists():
        raise ConfigError(f"input.videos: {video_path} does not exist")
    if video_path.is_file():
        if video_path.suffix.lower() not in VIDEO_SUFFIXES:
            raise ConfigError(f"input.videos: {video_path.name} is not a supported video format ({', '.join(VIDEO_SUFFIXES)})")
        candidates = [video_path]
    else:
        candidates = sorted(p for p in video_path.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
        if not candidates:
            raise ConfigError(f"input.videos: no video files ({', '.join(VIDEO_SUFFIXES)}) found in {video_path}")

    videos = []
    for pose_path in pose_files:
        matches = [v for v in candidates if pose_path.stem.startswith(v.stem)]
        if not matches:
            raise ConfigError(
                f"input.videos: no video matches pose file '{pose_path.name}' "
                f"(expected a video whose file name is a prefix of '{pose_path.stem}', "
                "i.e. the one DeepLabCut analysed)"
            )
        videos.append(max(matches, key=lambda v: len(v.stem)))  # most specific prefix wins

    reused = {v for v in videos if videos.count(v) > 1}
    if reused:
        raise ConfigError(f"input.videos: {', '.join(sorted(v.name for v in reused))} each match more than one pose file")
    return videos


def load_config(path: str | Path) -> ExperimentConfig:
    """Read the experiment YAML into a checked ExperimentConfig."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ConfigError(f"Experiment file not found: {path}")
    try:
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}  # safe_load: never runs code embedded in the file
    except yaml.YAMLError as e:
        raise ConfigError(f"{path.name} is not valid YAML: {e}") from e
    base = path.parent

    inp, kp = _section(raw, "input"), _section(raw, "keypoints")
    motifs, adv = _section(raw, "motifs"), _section(raw, "advanced")

    try:
        pose_files = _find_pose_files(base / _require(inp, "pose_files", "input"))
        cfg = ExperimentConfig(
            yaml_path=path,
            pose_files=pose_files,
            fps=float(_require(inp, "fps", "input")),
            align_center=str(_require(kp, "align_center", "keypoints")),
            align_direction=str(_require(kp, "align_direction", "keypoints")),
            n_clusters=int(_require(motifs, "n_clusters", "motifs")),
            method=str(motifs.get("method", "hmm")),
            min_confidence=float(_section(raw, "cleaning").get("min_confidence", 0.9)),
            exclude=list(kp.get("exclude") or []),
            videos=_find_videos(base / inp["videos"], pose_files) if inp.get("videos") else [],
            output=base / raw.get("output", "outputs"),
            project_name=str(raw.get("project_name", "vame_project")),
            max_epochs=int(adv.get("max_epochs", 100)),
            time_window=int(adv.get("time_window", 30)),
            zdims=int(adv.get("zdims", 30)),
            seed=int(adv.get("seed", 42)),
        )
    except ConfigError:
        raise
    except (TypeError, ValueError) as e:  # e.g. fps: "thirty"
        raise ConfigError(f"{path.name}: a value has the wrong type ({e})") from e

    # Value checks: fail now, not minutes into training.
    if cfg.method not in METHODS:
        raise ConfigError(f"motifs.method must be one of {', '.join(METHODS)} (got '{cfg.method}')")
    if cfg.fps <= 0:
        raise ConfigError("input.fps must be positive")
    if not 0 <= cfg.min_confidence <= 1:
        raise ConfigError("cleaning.min_confidence must be between 0 and 1")
    if cfg.n_clusters < 2:
        raise ConfigError("motifs.n_clusters must be at least 2")
    if cfg.max_epochs < MIN_EPOCHS:
        # VAME only saves a model once KL annealing has finished (about epoch 7 with its defaults).
        raise ConfigError(f"advanced.max_epochs must be at least {MIN_EPOCHS}")
    if cfg.time_window < 2:
        raise ConfigError("advanced.time_window must be at least 2")
    if cfg.align_center == cfg.align_direction:
        raise ConfigError("keypoints.align_center and keypoints.align_direction must be different body parts")
    if {cfg.align_center, cfg.align_direction} & set(cfg.exclude):
        raise ConfigError("keypoints.exclude cannot contain the alignment body parts")
    return cfg
