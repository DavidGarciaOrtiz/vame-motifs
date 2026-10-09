import numpy as np
import pytest
from conftest import write_video, write_yaml

from vame_motifs.config import load_config
from vame_motifs.pipeline import PipelineError, VamePipeline


def test_motif_videos_without_input_videos_says_so(experiment):
    pipeline = VamePipeline(load_config(experiment))
    with pytest.raises(PipelineError, match="No raw videos configured"):
        pipeline.motif_videos()


def test_motif_videos_without_labels_says_what_to_run(experiment):
    videos = experiment.parent / "videos"
    videos.mkdir()
    write_video(videos / "rat01.mp4")
    write_video(videos / "rat02.mp4")
    write_yaml(experiment.parent, videos="videos/")

    pipeline = VamePipeline(load_config(experiment))
    with pytest.raises(PipelineError, match="Run 'segment' first"):
        pipeline.motif_videos()


def test_has_motif_videos_false_until_clips_exist(experiment):
    cfg = load_config(experiment)
    pipeline = VamePipeline(cfg)
    assert not pipeline.has_motif_videos()

    for pose_path in cfg.pose_files:
        video_dir = pipeline.motif_videos_dir(pose_path.stem, "hmm")
        video_dir.mkdir(parents=True)
        (video_dir / f"{pose_path.stem}-motif_0.mp4").write_bytes(b"")
    assert pipeline.has_motif_videos()


def _write_community_bag(pipeline: VamePipeline, algorithm: str, bag: list[list[int]]) -> None:
    bag_path = pipeline.community_bag_file(algorithm)
    bag_path.parent.mkdir(parents=True)
    np.save(bag_path, np.array(bag, dtype=object))


def test_has_community_false_until_bag_exists(experiment):
    cfg = load_config(experiment)
    pipeline = VamePipeline(cfg)
    assert not pipeline.has_community()

    _write_community_bag(pipeline, "hmm", [[0, 1], [2, 3, 4]])
    assert pipeline.has_community()


def test_motif_to_community_maps_each_motif_to_its_bag_index(experiment):
    pipeline = VamePipeline(load_config(experiment))
    _write_community_bag(pipeline, "hmm", [[0, 2], [1, 3, 4]])

    assert pipeline._motif_to_community("hmm") == {0: 0, 2: 0, 1: 1, 3: 1, 4: 1}


def test_group_videos_by_community_moves_clips_into_community_subfolders(experiment):
    cfg = load_config(experiment)
    pipeline = VamePipeline(cfg)
    _write_community_bag(pipeline, "hmm", [[0, 2], [1, 3, 4]])

    for pose_path in cfg.pose_files:
        video_dir = pipeline.motif_videos_dir(pose_path.stem, "hmm")
        video_dir.mkdir(parents=True)
        for motif in range(5):
            (video_dir / f"{pose_path.stem}-motif_{motif}.mp4").write_bytes(b"")

    pipeline._group_videos_by_community("hmm")

    for pose_path in cfg.pose_files:
        video_dir = pipeline.motif_videos_dir(pose_path.stem, "hmm")
        assert not list(video_dir.glob("*.mp4"))  # nothing left flat at the top level
        assert sorted(p.name for p in (video_dir / "community_0").glob("*.mp4")) == [
            f"{pose_path.stem}-motif_0.mp4",
            f"{pose_path.stem}-motif_2.mp4",
        ]
        assert sorted(p.name for p in (video_dir / "community_1").glob("*.mp4")) == [
            f"{pose_path.stem}-motif_1.mp4",
            f"{pose_path.stem}-motif_3.mp4",
            f"{pose_path.stem}-motif_4.mp4",
        ]


def test_new_cut_moves_grouped_clips_to_their_new_community(experiment):
    cfg = load_config(experiment)
    pipeline = VamePipeline(cfg)
    _write_community_bag(pipeline, "hmm", [[0, 2], [1, 3, 4]])
    session = cfg.pose_files[0].stem
    video_dir = pipeline.motif_videos_dir(session, "hmm")
    video_dir.mkdir(parents=True)
    for motif in range(5):
        (video_dir / f"{session}-motif_{motif}.mp4").write_bytes(b"")
    pipeline._group_videos_by_community("hmm")

    pipeline.community_bag_file("hmm").unlink()
    np.save(pipeline.community_bag_file("hmm"), np.array([[0, 1, 2, 3], [4]], dtype=object))
    pipeline._group_videos_by_community("hmm")

    assert sorted(p.name for p in video_dir.iterdir()) == ["community_0", "community_1"]
    assert len(list((video_dir / "community_0").glob("*.mp4"))) == 4
    assert {m: p.parent.name for m, p in pipeline.motif_clips(session, "hmm").items()}[4] == "community_1"


def test_pose_matrix_orders_columns_x_y_confidence_per_keypoint():
    import xarray as xr

    from vame_motifs.pipeline import _pose_matrix

    # Stored as VAME stores it: (time, space, keypoints, individuals).
    position = np.array([[[10, 20], [11, 21]]], dtype=float)[..., None]  # x: 10, 20; y: 11, 21
    confidence = np.array([[0.9, 0.8]])[..., None]
    ds = xr.Dataset({"position": (("time", "space", "keypoints", "individuals"), position),
                     "confidence": (("time", "keypoints", "individuals"), confidence)})

    assert _pose_matrix(ds).tolist() == [[10, 11, 0.9, 20, 21, 0.8]]


def test_gif_start_is_in_a_well_tracked_stretch(experiment):
    pipeline = VamePipeline(load_config(experiment))  # min_confidence 0.9, time_window 30
    confidence = np.full((1000, 2), 0.99)
    confidence[:600 + 15, 0] = 0.1  # poorly tracked until window 600 (frame 615)

    start = pipeline._well_tracked_start(confidence, num_points=900, length=100)

    assert 595 <= start <= 800  # at least 95% of its frames well tracked
