import numpy as np
import pandas as pd
import pytest

from vame_motifs.config import load_config
from vame_motifs.export import ExportError, export_motifs, labels_to_frames
from vame_motifs.pipeline import VamePipeline


def test_labels_are_centred_on_their_window():
    n_frames, time_window = 100, 30
    labels = np.arange(n_frames - time_window + 1)  # 71 windows
    motif = labels_to_frames(labels, n_frames, time_window)
    assert motif.isna().sum() == 29
    assert motif.iloc[:15].isna().all() and motif.iloc[-14:].isna().all()
    assert motif.iloc[15] == 0 and motif.iloc[85] == 70


def test_too_many_labels_is_an_error():
    with pytest.raises(ExportError):
        labels_to_frames(np.zeros(90, dtype=int), 100, 30)


def test_export_writes_csvs_and_usage(experiment):
    cfg = load_config(experiment)
    pipeline = VamePipeline(cfg)
    rng = np.random.default_rng(0)
    for pose_path, n_frames in zip(cfg.pose_files, (200, 150)):
        label_path = pipeline.label_file(pose_path.stem, "hmm")
        label_path.parent.mkdir(parents=True)
        np.save(label_path, rng.integers(0, cfg.n_clusters, n_frames - cfg.time_window + 1))

    written = export_motifs(cfg)

    table = pd.read_csv(cfg.motifs_dir / "hmm" / "rat02DLC_resnet50_motifs.csv")
    assert list(table.columns) == ["frame", "time_s", "motif"]
    assert len(table) == 150 and table["motif"].isna().sum() == 29
    raw_line = (cfg.motifs_dir / "hmm" / "rat02DLC_resnet50_motifs.csv").read_text().splitlines()[16]
    assert raw_line.startswith("15,0.5,") and "." not in raw_line.split(",")[2]  # motif written as an integer
    assert table["time_s"].iloc[30] == 1.0
    usage = pd.read_csv(cfg.motifs_dir / "hmm" / "motif_usage.csv", index_col="session")
    assert usage.shape == (2, cfg.n_clusters)
    assert np.allclose(usage.sum(axis=1), 1, atol=1e-3)
    assert len(written) == 3


def test_export_without_segmentation_says_what_to_run(experiment):
    with pytest.raises(ExportError, match="Run 'segment' first"):
        export_motifs(load_config(experiment))
