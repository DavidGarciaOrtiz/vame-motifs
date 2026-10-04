import pytest
from conftest import BODYPARTS, write_dlc_csv

from vame_motifs.io import PoseFileError, count_frames, read_dlc_csv


def test_reads_bodyparts_in_order(tmp_path):
    pose = read_dlc_csv(write_dlc_csv(tmp_path / "a.csv", n_frames=50))
    assert pose.bodyparts == BODYPARTS
    assert pose.n_frames == 50
    assert pose.name == "a"


def test_count_frames_matches_parsed_rows(tmp_path):
    path = write_dlc_csv(tmp_path / "a.csv", n_frames=123)
    assert count_frames(path) == read_dlc_csv(path).n_frames == 123


def test_low_confidence_fraction_per_bodypart(tmp_path):
    pose = read_dlc_csv(write_dlc_csv(tmp_path / "a.csv", bad="tailbase"))
    frac = pose.low_confidence_fraction(0.9)
    assert frac["tailbase"] == 1.0
    assert frac["snout"] == 0.0


def test_rejects_non_dlc_csv(tmp_path):
    path = tmp_path / "plain.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(PoseFileError, match="does not look like a DeepLabCut CSV"):
        read_dlc_csv(path)


def test_rejects_multi_animal(tmp_path):
    path = tmp_path / "multi.csv"
    path.write_text("scorer,s,s,s\nindividuals,a,a,a\nbodyparts,n,n,n\ncoords,x,y,likelihood\n0,1,2,0.9\n")
    with pytest.raises(PoseFileError, match="multi-animal"):
        read_dlc_csv(path)


def test_missing_file(tmp_path):
    with pytest.raises(PoseFileError, match="not found"):
        read_dlc_csv(tmp_path / "nope.csv")
