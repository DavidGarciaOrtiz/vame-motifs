"""Communities: groups of motifs, cut from VAME's motif tree, and the user's names for them.

What a community is
-------------------
VAME may find, say, 15 motifs. Many are variations of one behaviour (three
kinds of "grooming", two kinds of "walking"). A *community* is a group of
motifs that belong together, giving fewer, broader categories.

How VAME builds them: it counts how often each motif is followed by each other
motif. Motifs that often follow each other are merged first, then the merged
groups are merged, and so on, until everything is one group. The result is a
*tree* (like a family tree, upside down):

- the *leaves* (bottom) are the motifs;
- each *inner node* is a merge of the nodes below it;
- the *root* (top) holds everything.

The *depth* of a node is how many steps it is below the root. Cutting the tree
at depth ``k`` means: each node at depth ``k`` becomes one community, holding
all the motifs below it. A small cut gives a few big communities; a large cut
gives many small ones.

VAME's ``community`` step builds a tree of the motifs (motifs that often follow
each other are merged first) and cuts it at an integer depth, ``cut_tree``:
every node at that depth becomes one community holding the motifs below it.
VAME saves the tree it built (``tree.graphml``), so any cut can be previewed
here without VAME, in milliseconds, before running ``communities`` with it.

    load_tree(path)            tree.graphml -> networkx tree
    cut_tree(tree, cutline)    the communities VAME would make at that cut
    tree_layout(tree)          node positions for drawing the tree

A list of communities is called a *bag* here (VAME's word): a list of lists of
motif numbers, e.g. ``[[6, 12, 14, 2], [4, 8, 10], [5]]`` means community 0
holds motifs 6, 12, 14 and 2; community 1 holds 4, 8 and 10; and so on.

Researchers' names for communities are kept next to the motif CSVs::

    <output>/motifs/<algorithm>/community_labels.csv               community, label, motifs
    <output>/motifs/<algorithm>/<session>_motifs_labeled.csv       frame, time_s, motif, community, label

Names are stored with the community's motifs, so they stay attached to the
right group when a new cut renumbers the communities.

Only numpy, pandas and networkx are needed (all installed with VAME), so the
window can use this module without loading VAME.

networkx (``nx``) is a library for graphs: things (nodes) connected by links
(edges). A tree is a graph without loops. Here the edges have no direction:
``tree.neighbors(node)`` gives a node's parent *and* its children.
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
    """VAME's tree.graphml. Leaves are motifs (named by their number); inner nodes are merges.

    In the file every node name is text: motifs are "0", "1", ... and merges
    have names like "h_6_28"; the top is "Root".
    """
    if not Path(path).exists():
        raise CommunityError(f"No motif tree at {path}. Run 'communities' once first.")
    return nx.read_graphml(path)


def is_motif(node) -> bool:
    """True for a motif (a leaf, named by its number), False for a merge or the root."""
    return str(node).isdigit()  # "12".isdigit() is True, "h_6_28".isdigit() is False


def _children(tree: nx.Graph, node, parent) -> list:
    """Children in VAME's left-to-right drawing order (its hierarchy_pos: neighbour order).

    The neighbours of a node are its parent and its children; leaving out the
    parent gives the children. (The root has no parent: pass None.)
    """
    return [n for n in tree.neighbors(node) if n != parent]


def depths(tree: nx.Graph) -> dict:
    """Distance of every node from the root."""
    # A dictionary node -> number of steps from the root (the root itself: 0).
    return nx.single_source_shortest_path_length(tree, ROOT)


def max_cut(tree: nx.Graph) -> int:
    """The deepest useful cut: below it every motif is its own community."""
    return max(depths(tree).values())


def leaf_order(tree: nx.Graph) -> list[int]:
    """Motifs from left to right, as VAME draws the tree."""
    order: list[int] = []

    # A *recursive* function: it calls itself for each child. Starting at the
    # root, it goes down the first child as far as possible, then the next...
    # ("depth-first"). Leaves are therefore reached from left to right.
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

    Example: the path from the root to motif 2 is Root -> h_a -> h_c -> 2.
    With cutline 1 motif 2 belongs to h_a's community; with cutline 2, to
    h_c's; with cutline 3 or more, to its own.
    """
    if cutline < 0:
        raise ValueError("The cut must be a non-negative integer")
    bags: dict = {}  # ancestor node -> the motifs below it, in order
    for motif in leaf_order(tree):
        # The nodes from the root down to this motif: path[0] is the root,
        # path[k] is the ancestor at depth k, the last one is the motif.
        path = nx.shortest_path(tree, ROOT, str(motif))
        ancestor = path[cutline] if len(path) - 1 >= cutline else str(motif)
        # setdefault: get the ancestor's list, creating an empty one the first time.
        bags.setdefault(ancestor, []).append(motif)
    # Dictionaries keep insertion order, so communities are in order of their
    # leftmost motif.
    return list(bags.values())  # leaves were visited left to right


def motif_to_community(bag) -> dict[int, int]:
    """Turn a bag ([[6, 12], [4]]) into a lookup: motif -> its community ({6: 0, 12: 0, 4: 1})."""
    # enumerate(bag) gives (0, first list), (1, second list), ...
    return {int(motif): community for community, motifs in enumerate(bag) for motif in motifs}


def tree_layout(tree: nx.Graph) -> dict:
    """(x, y) per node: motifs at x = 0, 1, 2... from left to right, a merge above the middle
    of its children, y = minus the depth (root at the top).

    Used by the window (viewer.py) to draw the tree.
    """
    # Leaves first: their x is their position from the left.
    x_of = {str(m): float(i) for i, m in enumerate(leaf_order(tree))}
    depth = depths(tree)

    # Recursive again: a node's x is the average x of its children, so the
    # children must be placed first. Returns the node's x.
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
    """Where the names are saved: <output>/motifs/<algorithm>/community_labels.csv."""
    return cfg.motifs_dir / algorithm / LABELS_FILE


def _motif_key(motifs) -> str:
    """A community's motifs as sorted text, e.g. [10, 4, 8] -> "4 8 10".

    Sorting makes the same group always give the same text, whatever the order
    of its motifs, so it can be used to recognise the group.
    """
    return " ".join(str(m) for m in sorted(int(m) for m in motifs))


def read_labels(cfg: ExperimentConfig, algorithm: str, bag) -> dict[int, str]:
    """Saved names for the communities in ``bag`` (community index -> name).

    Matched by the community's motifs, not its number: a name saved for
    motifs [4, 8, 10] is only given back to a community with exactly those.
    """
    path = labels_path(cfg, algorithm)
    if not path.exists():
        return {}  # nothing saved yet
    # dtype=str: read everything as text; keep_default_na=False: an empty name
    # stays "" instead of becoming pandas' "missing value".
    saved = pd.read_csv(path, dtype=str, keep_default_na=False)
    by_motifs = dict(zip(saved["motifs"], saved["label"]))  # "4 8 10" -> "rearing"
    # Keep only the communities that have a saved, non-empty name.
    return {i: by_motifs[_motif_key(m)] for i, m in enumerate(bag) if by_motifs.get(_motif_key(m))}


def write_labels(cfg: ExperimentConfig, algorithm: str, bag, names: dict[int, str]) -> list[Path]:
    """Save the names, and one ``<session>_motifs_labeled.csv`` per recording. Returns the files written.

    The labeled CSV is the motif CSV (export.motif_table) with two more columns:
    the community of each frame's motif and that community's name. Frames
    without a motif (the window edges) have neither.

    ``bag``: the communities (as VAME saved them); ``names``: community index ->
    name typed by the user (communities without a name get an empty one).
    """
    out_dir = cfg.motifs_dir / algorithm
    out_dir.mkdir(parents=True, exist_ok=True)
    # 1. community_labels.csv: one row per community.
    table = pd.DataFrame({
        "community": range(len(bag)),
        "label": [names.get(i, "").strip() for i in range(len(bag))],  # strip: remove spaces around the name
        "motifs": [_motif_key(m) for m in bag],
    })
    table.to_csv(labels_path(cfg, algorithm), index=False)
    written = [labels_path(cfg, algorithm)]

    # 2. One labeled CSV per recording.
    pipeline = VamePipeline(cfg)  # to find VAME's label files
    community_of = motif_to_community(bag)
    for pose_path in cfg.pose_files:
        session = pose_path.stem
        label_file = pipeline.label_file(session, algorithm)
        if not label_file.exists():
            raise CommunityError(f"No {algorithm} labels for '{session}' ({label_file}). Run 'segment' first.")
        # Start from the same table as <session>_motifs.csv (frame, time_s, motif)...
        frames = motif_table(np.load(label_file), count_frames(pose_path), cfg.fps, cfg.time_window)
        # ...and add two columns. .map(lookup) replaces every value by what the
        # lookup gives for it; empty motifs stay empty.
        frames["community"] = frames["motif"].map(community_of).astype("Int64")
        frames["label"] = frames["community"].map(dict(enumerate(table["label"])))
        out = out_dir / f"{session}_motifs_labeled.csv"
        frames.to_csv(out, index=False)
        written.append(out)
    return written
