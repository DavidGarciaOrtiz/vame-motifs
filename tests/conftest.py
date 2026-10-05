"""Small synthetic DeepLabCut files, written fresh for each test."""
from pathlib import Path

import numpy as np
import pytest

BODYPARTS = ["snout", "left_ear", "right_ear", "centre", "tailbase"]


def write_dlc_csv(path: Path, n_frames: int = 200, bodyparts=BODYPARTS, bad: str | None = None, seed: int = 0) -> Path:
    """A single-animal DLC CSV with a random walk per body part. `bad` gets low likelihoods."""
    rng = np.random.default_rng(seed)
    scorer = "DLC_resnet50_testOct4shuffle1_100"
    header = [
        ",".join(["scorer"] + [scorer] * 3 * len(bodyparts)),
        ",".join(["bodyparts"] + [bp for bp in bodyparts for _ in range(3)]),
        ",".join(["coords"] + ["x", "y", "likelihood"] * len(bodyparts)),
    ]
    rows = []
    xy = rng.normal(300, 20, size=(len(bodyparts), 2))
    for frame in range(n_frames):
        xy += rng.normal(0, 2, size=xy.shape)
        values = []
        for i, bp in enumerate(bodyparts):
            lk = 0.3 if bp == bad else 0.99
            values += [f"{xy[i, 0]:.2f}", f"{xy[i, 1]:.2f}", f"{lk:.3f}"]
        rows.append(",".join([str(frame)] + values))
    path.write_text("\n".join(header + rows) + "\n")
    return path


def write_video(path: Path) -> Path:
    """An empty stand-in for a raw video file; config-level tests only check names/extensions."""
    path.write_bytes(b"")
    return path


def write_yaml(folder: Path, **overrides) -> Path:
    settings = {"n_clusters": 5, "method": "hmm", "align_center": "snout", "exclude": "[]", "videos": ""} | overrides
    path = folder / "experiment.yaml"
    videos_line = f"  videos: {settings['videos']}\n" if settings["videos"] else ""
    path.write_text(
        f"""input:
  pose_files: pose/
{videos_line}  fps: 30
keypoints:
  align_center: {settings['align_center']}
  align_direction: tailbase
  exclude: {settings['exclude']}
motifs:
  n_clusters: {settings['n_clusters']}
  method: {settings['method']}
output: outputs
advanced:
  time_window: 30
"""
    )
    return path


@pytest.fixture
def experiment(tmp_path: Path) -> Path:
    """An experiment folder with two pose files and a valid YAML; returns the YAML path."""
    pose = tmp_path / "pose"
    pose.mkdir()
    write_dlc_csv(pose / "rat01DLC_resnet50.csv", seed=1)
    write_dlc_csv(pose / "rat02DLC_resnet50.csv", n_frames=150, seed=2)
    return write_yaml(tmp_path)
