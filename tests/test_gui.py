"""The GUI's form <-> YAML conversion; the window itself is not opened."""
import yaml

from vame_motifs.config import load_config
from vame_motifs.gui import DEFAULTS, form_to_yaml, read_bodyparts, yaml_to_form


def test_yaml_round_trip_is_read_by_the_cli(experiment):
    values = yaml_to_form(yaml.safe_load(experiment.read_text()))
    assert values["pose_files"] == "pose/" and values["n_clusters"] == "5"

    values["n_clusters"] = "7"
    values["exclude"] = ["left_ear"]
    experiment.write_text(yaml.safe_dump(form_to_yaml(values), sort_keys=False))

    cfg = load_config(experiment)
    assert cfg.n_clusters == 7 and cfg.exclude == ["left_ear"]
    assert cfg.time_window == 30 and cfg.videos == []


def test_missing_sections_keep_defaults():
    assert yaml_to_form({}) == DEFAULTS


def test_read_bodyparts(experiment):
    pose = sorted((experiment.parent / "pose").glob("*.csv"))[0]
    assert read_bodyparts(pose) == ["snout", "left_ear", "right_ear", "centre", "tailbase"]


def test_bouts_are_runs_of_frames():
    import numpy as np

    from vame_motifs.viewer import bouts

    assert bouts(np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=bool)).tolist() == [[1, 2], [5, 5], [7, 9]]
    assert bouts(np.zeros(4, dtype=bool)).tolist() == []
