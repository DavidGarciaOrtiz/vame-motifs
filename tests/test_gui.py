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


def test_short_names_are_the_last_digits_of_the_video_name():
    from vame_motifs.viewer import short_names

    long = "01022023_001_16236-{}DLC_Resnet50_Test-ExampleSep16shuffle1_snapshot_best-84_filtered"
    assert short_names([long.format(62755), long.format(62764)]) == {
        long.format(62755): "62755", long.format(62764): "62764"}
    # No digits, or the same digits twice: the full name.
    assert short_names(["mouseDLC_x", "a_7DLC", "b_7DLC", "rat01DLC"]) == {
        "mouseDLC_x": "mouseDLC_x", "a_7DLC": "a_7DLC", "b_7DLC": "b_7DLC", "rat01DLC": "01"}


def test_renamed_recordings_are_saved_and_read_back(experiment):
    import pytest

    from vame_motifs.viewer import Results

    cfg = load_config(experiment)  # rat01DLC_resnet50, rat02DLC_resnet50
    results = Results(cfg, "hmm")
    assert results.display == {"rat01DLC_resnet50": "01", "rat02DLC_resnet50": "02"}

    results.rename("rat02DLC_resnet50", "  control  ")
    assert Results(cfg, "hmm").display["rat02DLC_resnet50"] == "control"  # saved in the output folder
    with pytest.raises(ValueError, match="already the name"):
        results.rename("rat01DLC_resnet50", "control")
    results.rename("rat02DLC_resnet50", "")  # empty: back to the short name
    assert Results(cfg, "hmm").display["rat02DLC_resnet50"] == "02"


def test_usage_is_per_recording(experiment):
    import numpy as np

    from vame_motifs.pipeline import VamePipeline
    from vame_motifs.viewer import Results

    cfg = load_config(experiment)
    for pose_path, motif in zip(cfg.pose_files, (0, 3)):
        label_file = VamePipeline(cfg).label_file(pose_path.stem, "hmm")
        label_file.parent.mkdir(parents=True)
        np.save(label_file, np.full(100, motif))

    results = Results(cfg, "hmm")
    assert results.usage("rat01DLC_resnet50").tolist() == [1, 0, 0, 0, 0]
    assert results.usage("rat02DLC_resnet50").tolist() == [0, 0, 0, 1, 0]
    assert results.usage().tolist() == [0.5, 0, 0, 0.5, 0]  # all recordings together
