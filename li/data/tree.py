import networkx as nx
import numpy as np

EPS = 1e-6


class Tree:
    """a lineage tree as flat arrays in dfs preorder.

    parent[i] < i, the root is 0, and the subtree of i is the range i:end[i].
    cell[i] is the bank row of an observed leaf, -1 otherwise.
    """

    def __init__(self, name, nodes, parent, time, cell):
        self.name = name
        self.nodes = np.asarray(nodes)
        self.parent = np.asarray(parent, dtype=np.int64)
        self.time = np.asarray(time, dtype=np.float64)
        self.cell = np.asarray(cell, dtype=np.int64)
        n = len(self.parent)
        size = np.ones(n, dtype=np.int64)
        for i in range(n - 1, 0, -1):
            size[self.parent[i]] += size[i]
        self.end = np.arange(n) + size
        self.leaf = size == 1
        self.leaf_time = round(float(self.time[self.leaf].max()), 3)
        self.index = {}

    def __len__(self):
        return len(self.parent)

    def __repr__(self):
        return f"Tree({self.name}, {len(self)} nodes, {self.leaf.sum()} leaves, leaf time {self.leaf_time})"

    def n_leaves(self):
        cs = np.concatenate([[0], np.cumsum(self.leaf)])
        return cs[self.end] - cs[np.arange(len(self))]

    def levels(self):
        """node indices grouped by depth, root first."""
        if not hasattr(self, "_levels"):
            depth = np.zeros(len(self), dtype=np.int64)
            for i in range(1, len(self)):
                depth[i] = depth[self.parent[i]] + 1
            order = np.argsort(depth, kind="stable")
            self._levels = np.split(order, np.flatnonzero(np.diff(depth[order])) + 1)
        return self._levels

    def children(self, i):
        return i + 1 + np.flatnonzero(self.parent[i + 1:self.end[i]] == i)

    def build_index(self, bins, max_leaves):
        """subtree roots per bin: the snapshot at the bin, capped by leaf count."""
        n_leaves = self.n_leaves()
        par_time = np.where(self.parent >= 0, self.time[np.maximum(self.parent, 0)], -np.inf)
        self.index = {}
        for b in bins:
            # nodes just past the cut hang off a snapshot node
            past = (self.time > b + EPS) & (par_time <= b + EPS) & (self.parent >= 0)
            size = np.zeros(len(self), dtype=np.int64)
            np.add.at(size, self.parent[past], n_leaves[past])
            size[self.leaf & (self.time <= b + EPS)] = 1
            self.index[b] = np.flatnonzero((size >= 1) & (size <= max_leaves))

    def subtree(self, r, b):
        """r plus the clades of its children past the cut at b."""
        if self.leaf[r]:
            return np.array([r])
        kids = self.children(r)
        kids = kids[self.time[kids] > b + EPS]
        return np.concatenate([[r]] + [np.arange(k, self.end[k]) for k in kids])


def from_nx(g, name, rows=None):
    """convert a networkx tree; rows maps leaf names to bank rows."""
    root = next(n for n in g if g.in_degree(n) == 0)
    nodes = list(nx.dfs_preorder_nodes(g, root))
    pos = {n: i for i, n in enumerate(nodes)}
    parent = np.full(len(nodes), -1, dtype=np.int64)
    for u, v in g.edges:
        parent[pos[v]] = pos[u]
    time = np.array([g.nodes[n]["time"] for n in nodes], dtype=np.float64)
    leaf = np.array([g.out_degree(n) == 0 for n in nodes])
    time[leaf] = np.round(time[leaf], 3)  # leaves carry float jitter
    rows = rows or {}
    cell = np.array([rows.get(n, -1) if lf else -1 for n, lf in zip(nodes, leaf)], dtype=np.int64)
    return Tree(name, np.array(nodes, dtype=object), parent, time, cell)


def to_nx(tree):
    g = nx.DiGraph()
    for i, n in enumerate(tree.nodes):
        g.add_node(n, time=tree.time[i])
    for i in range(1, len(tree)):
        g.add_edge(tree.nodes[tree.parent[i]], tree.nodes[i])
    return g


def prune(tree):
    """drop leaves with no state and any branch left without observed leaves."""
    obs = tree.leaf & (tree.cell >= 0)
    cs = np.concatenate([[0], np.cumsum(obs)])
    keep = (cs[tree.end] - cs[np.arange(len(tree))]) > 0
    new = np.cumsum(keep) - 1
    parent = np.where(tree.parent[keep] >= 0, new[np.maximum(tree.parent[keep], 0)], -1)
    return Tree(tree.name, tree.nodes[keep], parent, tree.time[keep], tree.cell[keep])


def scramble(tree, rng):
    """leaf cells permuted within the tree: same topology and composition, no lineage signal."""
    cell = tree.cell.copy()
    cell[tree.leaf] = rng.permutation(cell[tree.leaf])
    return Tree(tree.name, tree.nodes, tree.parent, tree.time, cell)
