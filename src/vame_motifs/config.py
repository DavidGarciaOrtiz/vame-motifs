"""Loading and checking the experiment YAML.

One YAML file describes one experiment. It only holds the decisions that
belong to the researcher (which files, frame rate, reference body parts,
number of motifs). Everything VAME can work out by itself (body part names,
session names, number of features) is never asked for.

All paths in the YAML are relative to the YAML file itself, so the command
works from any folder.

What YAML is
------------
YAML is a plain-text settings format: ``key: value`` lines, grouped into
sections by indentation. Python's ``yaml`` library turns it into nested
dictionaries, e.g. ``{"input": {"fps": 30, ...}, "motifs": {...}}``. An
example::

    input:
      pose_files: data/pose/     # a folder of DLC .csv files, or one file
      videos: data/videos/       # optional
      fps: 30
    keypoints:
      align_center: snout
      align_direction: tailbase
      exclude: []
    cleaning:
      min_confidence: 0.9
    motifs:
      n_clusters: 15
      method: hmm
    output: outputs
    advanced:
      max_epochs: 100
      ...

How this module works
---------------------
``load_config(path)`` reads the file, finds the pose files and videos on
disk, converts every value to the right type, checks the values make sense,
and returns one ``ExperimentConfig`` object. Every other module reads its
settings from that object (``cfg.fps``, ``cfg.n_clusters``...), never from the
YAML directly. Any problem raises ``ConfigError`` with a message naming the
YAML field to fix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

MIN_EPOCHS = 10
# The "method" the researcher chooses -> the VAME segmentation algorithms it runs.
# hmm (hidden Markov model): slower, smoother motifs. kmeans: much faster.
METHODS = {"hmm": ["hmm"], "kmeans": ["kmeans"], "both": ["hmm", "kmeans"]}
VIDEO_SUFFIXES = (".mp4", ".avi")  # the formats VAME's own video step accepts


class ConfigError(ValueError):
    """The experiment file is missing something or has an invalid value."""


@dataclass
class ExperimentConfig:
    """All settings of one experiment, checked and with the right types.

    Paths are absolute here (already joined to the YAML file's folder).
    Fields without a "=" must always be given; the others have defaults.
    """

    yaml_path: Path  # the experiment file itself
    pose_files: list[Path]  # the DLC .csv files, sorted by name
    fps: float  # video frame rate: frames per second
    align_center: str  # body part placed at (0, 0) when VAME aligns the poses
    align_direction: str  # body part that sets the animal's orientation
    n_clusters: int  # number of motifs VAME should find
    method: str = "hmm"  # a key of METHODS
    min_confidence: float = 0.75  # DLC likelihood below this: the point is discarded and filled in
    exclude: list[str] = field(default_factory=list)  # body parts left out of the model
    videos: list[Path] = field(default_factory=list)  # one per pose file, same order; empty if none
    output: Path = Path("outputs")  # where every result goes
    project_name: str = "vame_project"  # VAME's project folder, inside output

    # advanced: sensible defaults, most researchers never change them
    max_epochs: int = 100  # training rounds over the data
    time_window: int = 30  # frames VAME looks at together to label one moment
    zdims: int = 30  # size of VAME's compact description of movement (the "latent space")
    seed: int = 42  # starting value of the random numbers: same seed + same data = same results
    community_cut_tree: int = 3  # where the motif tree is cut into communities (communities.py)

    @property
    def algorithms(self) -> list[str]:
        """VAME segmentation algorithms for the chosen method."""
        return METHODS[self.method]

    @property
    def motifs_dir(self) -> Path:
        """Folder of the motif CSVs written by export.py."""
        return self.output / "motifs"  # "/" joins paths with pathlib


def _require(section: dict, key: str, where: str):
    """The value of a field that must be present. ``where``: the section name, for the message."""
    if key not in section or section[key] is None:
        raise ConfigError(f"Missing required field '{where}.{key}'")
    return section[key]


def _section(raw: dict, name: str) -> dict:
    """A section of the YAML (a dictionary); an empty one if it is missing."""
    value = raw.get(name) or {}  # "or {}": a section written but left empty gives None
    if not isinstance(value, dict):  # e.g. "input: 30" instead of indented lines
        raise ConfigError(f"'{name}' must be a section (key: value lines), got {value!r}")
    return value


def _find_pose_files(pose_path: Path) -> list[Path]:
    """input.pose_files can be one .csv file or a folder of them."""
    if not pose_path.exists():
        raise ConfigError(f"input.pose_files: {pose_path} does not exist")
    if pose_path.is_file():
        return [pose_path]
    # glob("*.csv"): every file of the folder ending in .csv. Sorted, so the
    # order (and so VAME's results) is the same on every computer.
    files = sorted(pose_path.glob("*.csv"))
    if not files:
        raise ConfigError(f"No .csv files found in {pose_path}")
    return files


def _find_videos(video_path: Path, pose_files: list[Path]) -> list[Path]:
    """Videos paired to ``pose_files``, one per file, in the same order.

    Two DeepLabCut naming patterns are supported:
    - raw video: the video name is a prefix of the pose CSV's stem
      (``session1.mp4`` -> ``session1DLC_resnet50_....csv``).

    - labeled video: the pose CSV's stem is a prefix of the video name,
      with DLC's own suffix appended
      (``session1DLC_resnet50_..._filtered.csv`` ->
      ``session1DLC_resnet50_..._filtered_p60_labeled.mp4``).

    Each pose file is matched to the video whose stem is closest to its
    own (smallest length difference), in whichever direction applies.

    (A "stem" is the file name without its extension; a "prefix" is the
    beginning of a text: "rat01" is a prefix of "rat01DLC".)
    """
    # Step 1: the candidate videos (one file, or every video in a folder).
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

    # Step 2: for each pose file, the videos whose name fits either pattern.
    videos = []
    for pose_path in pose_files:
        matches = [
            v for v in candidates
            if pose_path.stem.startswith(v.stem) or v.stem.startswith(pose_path.stem)
        ]
        if not matches:
            raise ConfigError(
                f"input.videos: no video matches pose file '{pose_path.name}' "
                f"(expected either the raw video DeepLabCut analysed, whose file name "
                f"is a prefix of '{pose_path.stem}', or a labeled video whose file name "
                f"starts with '{pose_path.stem}')"
            )
        # most specific match wins: smallest stem-length difference, in either direction
        # (min with key=...: the item for which that function gives the smallest number;
        # "lambda v: ..." is a one-line function without a name).
        videos.append(min(matches, key=lambda v: abs(len(v.stem) - len(pose_path.stem))))

    # Step 3: one video cannot belong to two recordings.
    reused = {v for v in videos if videos.count(v) > 1}
    if reused:
        raise ConfigError(f"input.videos: {', '.join(sorted(v.name for v in reused))} each match more than one pose file")
    return videos


def load_config(path: str | Path) -> ExperimentConfig:
    """Read the experiment YAML into a checked ExperimentConfig."""
    # expanduser: "~" -> the home folder. resolve: an absolute path.
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ConfigError(f"Experiment file not found: {path}")
    try:
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}  # safe_load: never runs code embedded in the file
    except yaml.YAMLError as e:
        raise ConfigError(f"{path.name} is not valid YAML: {e}") from e
    base = path.parent  # the folder relative paths start from

    inp, kp = _section(raw, "input"), _section(raw, "keypoints")
    motifs, adv = _section(raw, "motifs"), _section(raw, "advanced")

    # Build the config. float(...), int(...), str(...) convert each value to
    # its type; a wrong value (fps: "thirty") makes them raise TypeError or
    # ValueError, which is turned into a ConfigError below.
    # x.get("key", default): the value, or the default when the key is missing.
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
            community_cut_tree=int(adv.get("community_cut_tree", 3)),
        )
    except ConfigError:
        raise  # already a clear message: pass it on unchanged
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
    if cfg.community_cut_tree < 0:
        raise ConfigError("advanced.community_cut_tree must be a non-negative integer")
    if cfg.align_center == cfg.align_direction:
        raise ConfigError("keypoints.align_center and keypoints.align_direction must be different body parts")
    # "&" between two sets: the names in both. Alignment body parts cannot be left out.
    if {cfg.align_center, cfg.align_direction} & set(cfg.exclude):
        raise ConfigError("keypoints.exclude cannot contain the alignment body parts")
    return cfg
