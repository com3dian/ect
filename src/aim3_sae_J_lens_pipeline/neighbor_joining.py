"""Neighbor-Joining (Saitou–Nei) on a tropical distance matrix → Newick tree."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class NJNode:
    name: str
    children: list[tuple["NJNode", float]] = field(default_factory=list)

    def to_newick(self) -> str:
        if not self.children:
            return _escape_newick(self.name)
        inner = ",".join(
            f"{child.to_newick()}:{length:.6g}" for child, length in self.children
        )
        if self.name and not self.name.startswith("internal_"):
            return f"({inner}){_escape_newick(self.name)}"
        return f"({inner})"


def _escape_newick(name: str) -> str:
    if any(ch in name for ch in "():,;'\""):
        return "'" + name.replace("'", "''") + "'"
    return name.replace(" ", "_")


def neighbor_joining(labels: list[str], distances: np.ndarray) -> NJNode:
    """
    Build an unrooted NJ tree. ``distances`` is a symmetric n×n matrix with
    zeros on the diagonal. Branch lengths may be slightly negative if D is
    not a perfect additive tree; they are left as-is for diagnostics.
    """
    d = np.asarray(distances, dtype=np.float64)
    n = d.shape[0]
    if d.shape != (n, n):
        raise ValueError("distances must be square")
    if len(labels) != n:
        raise ValueError("labels length must match distance matrix")
    if n < 2:
        return NJNode(name=labels[0] if labels else "empty")

    nodes: list[NJNode] = [NJNode(name=lab) for lab in labels]
    d = d.copy()
    np.fill_diagonal(d, 0.0)
    next_internal = 0

    while n > 2:
        row_sum = d.sum(axis=1)
        q = (n - 2.0) * d - row_sum[:, None] - row_sum[None, :]
        np.fill_diagonal(q, np.inf)
        i, j = np.unravel_index(np.argmin(q), q.shape)
        if i > j:
            i, j = j, i

        delta = (row_sum[i] - row_sum[j]) / (n - 2.0)
        limb_i = 0.5 * d[i, j] + 0.5 * delta
        limb_j = d[i, j] - limb_i

        parent = NJNode(name=f"internal_{next_internal}")
        next_internal += 1
        parent.children = [(nodes[i], float(limb_i)), (nodes[j], float(limb_j))]

        mask = [k for k in range(n) if k != i and k != j]
        if not mask:
            return parent
        d_u = 0.5 * (d[i, mask] + d[j, mask] - d[i, j])
        new_d = np.zeros((n - 1, n - 1), dtype=np.float64)
        new_nodes: list[NJNode] = [parent] + [nodes[k] for k in mask]
        new_d[0, 1:] = d_u
        new_d[1:, 0] = d_u
        sub = d[np.ix_(mask, mask)]
        new_d[1:, 1:] = sub

        nodes = new_nodes
        d = new_d
        n = d.shape[0]

    parent = NJNode(name=f"internal_{next_internal}")
    parent.children = [(nodes[0], float(0.5 * d[0, 1])), (nodes[1], float(0.5 * d[0, 1]))]
    return parent


def newick_from_distances(labels: list[str], distances: np.ndarray) -> str:
    tree = neighbor_joining(labels, distances)
    return tree.to_newick() + ";"
