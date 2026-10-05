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
