"""Communities: groups of motifs, cut from VAME's motif tree, and the user's names for them.

VAME's ``community`` step builds a tree of the motifs (motifs that often follow
each other are merged first) and cuts it at an integer depth, ``cut_tree``:
every node at that depth becomes one community holding the motifs below it.
VAME saves the tree it built (``tree.graphml``), so any cut can be previewed
here without VAME, in milliseconds, before running ``communities`` with it.

    load_tree(path)            tree.graphml -> networkx tree
    cut_tree(tree, cutline)    the communities VAME would make at that cut
    tree_layout(tree)          node positions for drawing the tree

Researchers' names for communities are kept next to the motif CSVs::

    <output>/motifs/<algorithm>/community_labels.csv               community, label, motifs
    <output>/motifs/<algorithm>/<session>_motifs_labeled.csv       frame, time_s, motif, community, label

Names are stored with the community's motifs, so they stay attached to the
right group when a new cut renumbers the communities.

Only numpy, pandas and networkx are needed (all installed with VAME), so the
window can use this module without loading VAME.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from vame_motifs.config import ExperimentConfig
from vame_motifs.export import motif_table
from vame_motifs.io import count_frames
from vame_motifs.pipeline import VamePipeline

ROOT = "Root"  # VAME's name for the tree's root
LABELS_FILE = "community_labels.csv"


class CommunityError(RuntimeError):
    """The motif tree or community grouping is missing."""


# ======================================================================
# The motif tree
# ======================================================================
def load_tree(path: Path) -> nx.Graph:
    """VAME's tree.graphml. Leaves are motifs (named by their number); inner nodes are merges."""
    if not Path(path).exists():
        raise CommunityError(f"No motif tree at {path}. Run 'communities' once first.")
    return nx.read_graphml(path)


def is_motif(node) -> bool:
    return str(node).isdigit()


def _children(tree: nx.Graph, node, parent) -> list:
    """Children in VAME's left-to-right drawing order (its hierarchy_pos: neighbour order)."""
    return [n for n in tree.neighbors(node) if n != parent]


def depths(tree: nx.Graph) -> dict:
    """Distance of every node from the root."""
    return nx.single_source_shortest_path_length(tree, ROOT)


def max_cut(tree: nx.Graph) -> int:
    """The deepest useful cut: below it every motif is its own community."""
    return max(depths(tree).values())


def leaf_order(tree: nx.Graph) -> list[int]:
    """Motifs from left to right, as VAME draws the tree."""
    order: list[int] = []

    def visit(node, parent):
        children = _children(tree, node, parent)
        if not children and is_motif(node):
            order.append(int(node))
        for child in children:
            visit(child, node)

    visit(ROOT, None)
    return order


def cut_tree(tree: nx.Graph, cutline: int) -> list[list[int]]:
    """The communities ``vame.community(cut_tree=cutline)`` makes from this tree.

    Same rule as VAME (bag_nodes_by_cutline): each motif goes to its ancestor at
    depth ``cutline``; a motif shallower than that is a community on its own.
    Same order too (sort_communities_by_position): communities and the motifs in
    them from left to right.
    """
    if cutline < 0:
        raise ValueError("The cut must be a non-negative integer")
    bags: dict = {}
    for motif in leaf_order(tree):
        path = nx.shortest_path(tree, ROOT, str(motif))
        ancestor = path[cutline] if len(path) - 1 >= cutline else str(motif)
        bags.setdefault(ancestor, []).append(motif)
    return list(bags.values())  # leaves were visited left to right


def motif_to_community(bag) -> dict[int, int]:
    return {int(motif): community for community, motifs in enumerate(bag) for motif in motifs}


def tree_layout(tree: nx.Graph) -> dict:
    """(x, y) per node: motifs at x = 0, 1, 2... from left to right, a merge above the middle
    of its children, y = minus the depth (root at the top)."""
    x_of = {str(m): float(i) for i, m in enumerate(leaf_order(tree))}
    depth = depths(tree)

    def place(node, parent) -> float:
        children = _children(tree, node, parent)
        if children:
            x_of[node] = float(np.mean([place(child, node) for child in children]))
        return x_of[node]

    place(ROOT, None)
    return {node: (x_of[node], -depth[node]) for node in tree.nodes}


# ======================================================================
# Community names
# ======================================================================
def labels_path(cfg: ExperimentConfig, algorithm: str) -> Path:
    return cfg.motifs_dir / algorithm / LABELS_FILE


def _motif_key(motifs) -> str:
    return " ".join(str(m) for m in sorted(int(m) for m in motifs))


def read_labels(cfg: ExperimentConfig, algorithm: str, bag) -> dict[int, str]:
    """Saved names for the communities in ``bag`` (community index -> name).

    Matched by the community's motifs, not its number: a name saved for
    motifs [4, 8, 10] is only given back to a community with exactly those.
    """
    path = labels_path(cfg, algorithm)
    if not path.exists():
        return {}
    saved = pd.read_csv(path, dtype=str, keep_default_na=False)
    by_motifs = dict(zip(saved["motifs"], saved["label"]))
    return {i: by_motifs[_motif_key(m)] for i, m in enumerate(bag) if by_motifs.get(_motif_key(m))}


def write_labels(cfg: ExperimentConfig, algorithm: str, bag, names: dict[int, str]) -> list[Path]:
    """Save the names, and one ``<session>_motifs_labeled.csv`` per recording. Returns the files written.

    The labeled CSV is the motif CSV (export.motif_table) with two more columns:
    the community of each frame's motif and that community's name. Frames
    without a motif (the window edges) have neither.
    """
    out_dir = cfg.motifs_dir / algorithm
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame({
        "community": range(len(bag)),
        "label": [names.get(i, "").strip() for i in range(len(bag))],
        "motifs": [_motif_key(m) for m in bag],
    })
    table.to_csv(labels_path(cfg, algorithm), index=False)
    written = [labels_path(cfg, algorithm)]

    pipeline = VamePipeline(cfg)
    community_of = motif_to_community(bag)
    for pose_path in cfg.pose_files:
        session = pose_path.stem
        label_file = pipeline.label_file(session, algorithm)
        if not label_file.exists():
            raise CommunityError(f"No {algorithm} labels for '{session}' ({label_file}). Run 'segment' first.")
        frames = motif_table(np.load(label_file), count_frames(pose_path), cfg.fps, cfg.time_window)
        frames["community"] = frames["motif"].map(community_of).astype("Int64")
        frames["label"] = frames["community"].map(dict(enumerate(table["label"])))
        out = out_dir / f"{session}_motifs_labeled.csv"
        frames.to_csv(out, index=False)
        written.append(out)
    return written
