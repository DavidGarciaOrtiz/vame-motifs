"""Cutting VAME's motif tree, and the community names saved as *_labeled.csv."""
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from vame_motifs import communities
from vame_motifs.config import load_config
from vame_motifs.pipeline import VamePipeline


def write_tree(path) -> None:
    """A tree like VAME's tree.graphml for 5 motifs:

                Root
              /      \\
          h_a          h_b
         /   \\       /   \\
        3    h_c     4     1
            /   \\
           0     2
    """
    tree = nx.Graph()
    tree.add_edges_from([("Root", "h_a"), ("Root", "h_b"), ("h_a", 3), ("h_a", "h_c"),
                         ("h_c", 0), ("h_c", 2), ("h_b", 4), ("h_b", 1)])
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(tree, path)


@pytest.fixture
def tree(tmp_path):
    write_tree(tmp_path / "tree.graphml")
    return communities.load_tree(tmp_path / "tree.graphml")


def test_cut_tree_groups_motifs_under_the_nodes_at_that_depth(tree):
    assert communities.cut_tree(tree, 0) == [[3, 0, 2, 4, 1]]
    assert communities.cut_tree(tree, 1) == [[3, 0, 2], [4, 1]]
    assert communities.cut_tree(tree, 2) == [[3], [0, 2], [4], [1]]
    # Deeper than a motif: that motif is a community on its own.
    assert communities.cut_tree(tree, 3) == [[3], [0], [2], [4], [1]]
    assert communities.max_cut(tree) == 3


def test_negative_cut_is_refused(tree):
    with pytest.raises(ValueError):
        communities.cut_tree(tree, -1)


def test_missing_tree_says_what_to_run(tmp_path):
    with pytest.raises(communities.CommunityError, match="communities"):
        communities.load_tree(tmp_path / "tree.graphml")


def test_layout_puts_motifs_left_to_right_and_root_on_top(tree):
    pos = communities.tree_layout(tree)
    assert [pos[str(m)][0] for m in communities.leaf_order(tree)] == [0, 1, 2, 3, 4]
    assert pos["Root"][1] == 0 and pos["0"][1] == -3
    assert pos["h_c"][0] == 1.5  # above the middle of its two motifs


def _write_labels(cfg) -> None:
    pipeline = VamePipeline(cfg)
    for pose_path, n_frames in zip(cfg.pose_files, (200, 150)):
        label_path = pipeline.label_file(pose_path.stem, "hmm")
        label_path.parent.mkdir(parents=True)
        np.save(label_path, np.arange(n_frames - cfg.time_window + 1) % cfg.n_clusters)


def test_write_labels_saves_names_and_labeled_csvs(experiment):
    cfg = load_config(experiment)
    _write_labels(cfg)
    bag = [[3, 0, 2], [4, 1]]

    written = communities.write_labels(cfg, "hmm", bag, {0: "grooming", 1: " rearing "})

    assert [p.name for p in written] == ["community_labels.csv", "rat01DLC_resnet50_motifs_labeled.csv",
                                         "rat02DLC_resnet50_motifs_labeled.csv"]
    names = pd.read_csv(cfg.motifs_dir / "hmm" / "community_labels.csv", dtype=str)
    assert names.to_dict("list") == {"community": ["0", "1"], "label": ["grooming", "rearing"],
                                     "motifs": ["0 2 3", "1 4"]}
    table = pd.read_csv(cfg.motifs_dir / "hmm" / "rat02DLC_resnet50_motifs_labeled.csv")
    assert list(table.columns) == ["frame", "time_s", "motif", "community", "label"]
    assert len(table) == 150 and table["community"].isna().sum() == 29  # the window edges, as in the motif CSV
    row = table.iloc[15]  # first labelled frame: window 0, motif 0
    assert (row["motif"], row["community"], row["label"]) == (0, 0, "grooming")
    assert table.loc[table["motif"] == 4, "label"].eq("rearing").all()


def test_names_follow_their_motifs_when_communities_are_renumbered(experiment):
    cfg = load_config(experiment)
    _write_labels(cfg)
    communities.write_labels(cfg, "hmm", [[3, 0, 2], [4, 1]], {0: "grooming", 1: "rearing"})

    # A new cut: [4, 1] is now community 0, and [3, 0, 2] was split, so its name does not apply.
    assert communities.read_labels(cfg, "hmm", [[4, 1], [3], [0, 2]]) == {0: "rearing"}


def test_write_labels_without_segmentation_says_what_to_run(experiment):
    cfg = load_config(experiment)
    with pytest.raises(communities.CommunityError, match="Run 'segment' first"):
        communities.write_labels(cfg, "hmm", [[0, 1, 2, 3, 4]], {})
