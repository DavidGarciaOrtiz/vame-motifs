import pytest
from conftest import write_dlc_csv, write_yaml

from vame_motifs.config import ConfigError, load_config
from vame_motifs.io import PoseFileError
from vame_motifs.validation import validate


def test_valid_experiment(experiment):
    _, reports = validate(load_config(experiment))
    assert [r.n_frames for r in reports] == [200, 150]
    assert all(not r.warnings for r in reports)


def test_warns_on_poor_tracking(experiment):
    write_dlc_csv(experiment.parent / "pose" / "rat01DLC_resnet50.csv", bad="left_ear")
    _, reports = validate(load_config(experiment))
    assert any("left_ear" in w for w in reports[0].warnings)


def test_unknown_bodypart_lists_available(experiment):
    write_yaml(experiment.parent, align_center="nose")
    with pytest.raises(ConfigError, match="Unknown body part 'nose'. Available: snout"):
        validate(load_config(experiment))


def test_mismatched_bodyparts(experiment):
    write_dlc_csv(experiment.parent / "pose" / "rat03.csv", bodyparts=["snout", "centre", "tailbase"])
    with pytest.raises(PoseFileError, match="All files must match"):
        validate(load_config(experiment))


def test_config_rejects_bad_values(experiment):
    write_yaml(experiment.parent, method="spectral")
    with pytest.raises(ConfigError, match="motifs.method"):
        load_config(experiment)
    write_yaml(experiment.parent, n_clusters='"many"')
    with pytest.raises(ConfigError, match="wrong type"):
        load_config(experiment)


def test_config_paths_are_relative_to_yaml(experiment):
    cfg = load_config(experiment)
    assert cfg.output == experiment.parent / "outputs"
    assert [p.name for p in cfg.pose_files] == ["rat01DLC_resnet50.csv", "rat02DLC_resnet50.csv"]
