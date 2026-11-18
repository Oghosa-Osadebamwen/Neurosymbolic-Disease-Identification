"""
gng.py - Growing Neural Gas with a growth cap (Stage 1, outlier network)
=======================================================================

Fritzke (1995) Growing Neural Gas, configured as the design requires:

* INITIALISED FROM GHSOM: the GNG starts with one node per GHSOM leaf cluster (same number of
  nodes, prototypes copied from the GHSOM leaves, each node inherits the leaf's disease label).
* PRETRAINING: standard GNG adaptation over the training patients, with node insertion
  switched off - the nodes move and the edge topology is learnt, but the node count stays
  equal to GHSOM's.
      s1, s2 = two nearest nodes to x
      age(e) += 1 for every edge e of s1
      E(s1)  += ||x - w_s1||^2
      w_s1   += eps_b (x - w_s1);    w_n += eps_n (x - w_n) for topological neighbours n
      edge(s1, s2) created / age reset to 0;  edges older than a_max are removed
* GROWTH CAP = 1 at test time: at most ONE provisional node may exist at a time. When an
  outlier matches no node, the GNG proposes a node at x (cap -> 0). The symbolic stage then
  either COMMITS it (confirmed new disease: node kept, cap reset to 1) or REJECTS it
  (patient was a known disease: node removed, cap reset to 1). A committed node is never
  deleted, so later patients of the same new disease can be recognised (Case 3).
* Case 3 leaves the GNG unchanged: a registered node is never moved, no node is added and the
  cap stays at 1.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .ghsom import sq_dist


class CappedGNG:
    def __init__(self, eps_b=0.05, eps_n=0.005, a_max=50, err_decay=0.995, cap=1, seed=0):
        self.eps_b, self.eps_n, self.a_max, self.err_decay = eps_b, eps_n, a_max, err_decay
        self.cap0 = cap
        self.cap = cap
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ init + pretrain
    def init_from_ghsom(self, ghsom):
        self.W = ghsom.prototypes.astype(np.float64).copy()
        self.label = list(ghsom.leaf_label)            # disease name per node
        self.kind = ["known"] * len(self.W)            # "known" | "provisional" | "new"
        self.n_assigned = np.zeros(len(self.W))
        self.E = np.zeros(len(self.W))
        self.edges: dict[tuple, int] = {}
        self.n_init = len(self.W)
        return self

    def _nbrs(self, i):
        return [b if a == i else a for (a, b) in self.edges if i in (a, b)]

    def pretrain(self, X, y, epochs=3):
        X = X.astype(np.float64)
        for _ in range(epochs):
            for i in self.rng.permutation(len(X)):
                x = X[i]
                d = ((self.W - x) ** 2).sum(1)
                s1, s2 = np.argsort(d)[:2]
                for e in list(self.edges):
                    if s1 in e:
                        self.edges[e] += 1
                self.E[s1] += d[s1]
                self.W[s1] += self.eps_b * (x - self.W[s1])
                for n in self._nbrs(s1):
                    self.W[n] += self.eps_n * (x - self.W[n])
                self.edges[tuple(sorted((int(s1), int(s2))))] = 0
                self.edges = {e: a for e, a in self.edges.items() if a <= self.a_max}
                self.E *= self.err_decay
        # relabel nodes from the patients they now win (fallback: inherited GHSOM label)
        win = sq_dist(X, self.W).argmin(1)
        self.purity = np.zeros(len(self.W))
        for j in range(len(self.W)):
            yj = y[win == j]
            if len(yj):
                lab, c = Counter(yj).most_common(1)[0]
                self.label[j], self.purity[j] = lab, c / len(yj)
                self.n_assigned[j] = len(yj)
        return self

    # ------------------------------------------------------------------ queries
    def distances(self, x):
        """Squared Euclidean (used only for GNG's own learning and insertion edges)."""
        return ((self.W - x) ** 2).sum(1)

    def jaccard(self, x):
        """Weighted Jaccard distance from x to every node prototype (used by the XAI tests)."""
        from .xai import jaccard_dist
        return jaccard_dist(x > 0.5, self.W)[0]

    # ------------------------------------------------------------------ growth (cap = 1)
    def propose(self, x) -> int | None:
        """Insert a provisional node at x if the cap allows. Returns node id or None."""
        if self.cap <= 0:
            return None
        d = self.distances(x)
        s1, s2 = np.argsort(d)[:2]
        self.W = np.vstack([self.W, x[None].astype(np.float64)])
        self.label.append(None)
        self.kind.append("provisional")
        self.n_assigned = np.append(self.n_assigned, 1)
        self.E = np.append(self.E, 0.0)
        self.purity = np.append(self.purity, 1.0)
        q = len(self.W) - 1
        self.edges[(int(s1), q)] = 0
        self.edges[(int(s2), q)] = 0
        self.cap -= 1
        return q

    def commit(self, q, name):
        """Symbolic stage confirmed a new disease: keep node, register it, reset cap."""
        self.kind[q] = "new"
        self.label[q] = name
        self.cap = self.cap0

    def reject(self, q):
        """Symbolic stage said 'known disease': delete the provisional node, reset cap."""
        keep = [i for i in range(len(self.W)) if i != q]
        self.W = self.W[keep]
        self.label = [self.label[i] for i in keep]
        self.kind = [self.kind[i] for i in keep]
        self.n_assigned = self.n_assigned[keep]
        self.E = self.E[keep]
        self.purity = self.purity[keep]
        self.edges = {(a - (a > q), b - (b > q)): age for (a, b), age in self.edges.items()
                      if q not in (a, b)}
        self.cap = self.cap0

    def summary(self):
        k = Counter(self.kind)
        return {"nodes": len(self.W), "init_nodes": self.n_init, "new_nodes": k.get("new", 0),
                "edges": len(self.edges)}
