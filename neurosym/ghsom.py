"""
ghsom.py - Growing Hierarchical Self-Organising Map (Stage 1, bulk network)
==========================================================================

Implements GHSOM (Dittenbach, Merkl & Rauber, 2000; Rauber, Merkl & Dittenbach, 2002) with
BOTH growth directions, plus the label-aware expansion rule required by the design
("GHSOM is pretrained with labelled symptoms").

Notation
--------
x in {0,1}^m        a patient's symptom vector
w_i in [0,1]^m      weight (prototype) of map unit i
c(x) = argmin_i ||x - w_i||           best-matching unit (BMU)
qe_i  = (1/|C_i|) sum_{x in C_i} ||x - w_i||    mean quantisation error of unit i
MQE_map = mean_i qe_i (over units that map data)
qe_0  = (1/n) sum_x ||x - mean(X)||   error of the single layer-0 unit

Horizontal growth (breadth) of a map whose parent unit has error qe_p:
    while MQE_map >= tau1 * qe_p:
        e = argmax_i sum_{x in C_i} ||x - w_i||        (error unit)
        d = most dissimilar grid neighbour of e
        insert a row or column between e and d, new weights = mean of the two neighbours
        retrain the map (batch SOM)

Vertical growth (depth): a unit i spawns a child map, trained only on C_i, if
    qe_i >= tau2 * qe_0                        (standard GHSOM criterion)   OR
    purity_i < purity_target                   (label-aware criterion, pretraining with labels)
and |C_i| >= min_child_samples and depth < max_depth.

Batch SOM update (Kohonen batch algorithm) with Gaussian neighbourhood:
    h_ci(t) = exp(-||r_c - r_i||^2 / (2 sigma(t)^2)),  sigma decays geometrically
    w_i <- sum_x h_{c(x) i} x / sum_x h_{c(x) i}

The leaves of the hierarchy (units with data and no child map) are the GHSOM clusters.
Each leaf gets a numeric cluster code; the labelled pretraining gives each code a majority
disease, a purity and a label distribution - this is what Stage 3 looks up.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np


def sq_dist(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance between every row of X and every row of W."""
    d = (X * X).sum(1)[:, None] - 2.0 * X @ W.T + (W * W).sum(1)[None, :]
    return np.maximum(d, 0.0)


@dataclass
class SOMLayer:
    """One map of the hierarchy."""
    map_id: int
    depth: int
    W: np.ndarray                     # (rows, cols, m)
    parent: tuple | None              # (parent_map_id, parent_unit) or None for the root map
    children: dict = field(default_factory=dict)   # unit index -> child map_id
    live: np.ndarray | None = None    # boolean mask of units that map training data

    @property
    def shape(self):
        return self.W.shape[:2]

    @property
    def flat(self):
        return self.W.reshape(-1, self.W.shape[2])


class GHSOM:
    def __init__(self, tau1=0.6, tau2=0.05, purity_target=0.95, max_units=64, max_depth=3,
                 min_child_samples=6, epochs=30, sigma0=None, seed=0):
        self.tau1, self.tau2 = tau1, tau2
        self.purity_target = purity_target
        self.max_units, self.max_depth = max_units, max_depth
        self.min_child_samples = min_child_samples
        self.epochs, self.sigma0 = epochs, sigma0
        self.rng = np.random.default_rng(seed)
        self.maps: list[SOMLayer] = []
        self.growth_log: list[dict] = []

    # ------------------------------------------------------------------ batch SOM
    def _train_map(self, W, X, epochs):
        rows, cols, m = W.shape
        grid = np.array([(r, c) for r in range(rows) for c in range(cols)], dtype=float)
        G = sq_dist(grid, grid)
        s0 = self.sigma0 or max(rows, cols) / 2.0
        s1 = 0.5
        flat = W.reshape(-1, m).copy()
        for t in range(epochs):
            sigma = s0 * (s1 / s0) ** (t / max(1, epochs - 1))
            bmu = sq_dist(X, flat).argmin(1)
            H = np.exp(-G[bmu] / (2 * sigma ** 2))           # (n, units)
            num = H.T @ X
            den = H.sum(0)[:, None]
            ok = den[:, 0] > 1e-12
            flat[ok] = num[ok] / den[ok]
        return flat.reshape(rows, cols, m)

    def _unit_errors(self, flat, X):
        d = np.sqrt(sq_dist(X, flat))
        bmu = d.argmin(1)
        dist = d[np.arange(len(X)), bmu]
        U = flat.shape[0]
        n = np.bincount(bmu, minlength=U)
        s = np.bincount(bmu, weights=dist, minlength=U)
        qe = np.where(n > 0, s / np.maximum(n, 1), 0.0)
        return bmu, dist, n, s, qe

    def _grow_map(self, X, parent_qe):
        """Horizontal growth of one map, starting from 2x2."""
        m = X.shape[1]
        idx = self.rng.choice(len(X), size=min(4, len(X)), replace=False)
        init = X[idx] + self.rng.normal(0, 0.01, (len(idx), m))
        if len(idx) < 4:
            init = np.vstack([init] * 4)[:4]
        W = self._train_map(init.reshape(2, 2, m), X, self.epochs)
        while True:
            flat = W.reshape(-1, m)
            bmu, _, n, s, qe = self._unit_errors(flat, X)
            mqe = qe[n > 0].mean()
            rows, cols = W.shape[:2]
            if mqe < self.tau1 * parent_qe or rows * cols >= self.max_units:
                return W
            e = int(np.argmax(s))
            er, ec = divmod(e, cols)
            nbrs = [(er + dr, ec + dc) for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                    if 0 <= er + dr < rows and 0 <= ec + dc < cols]
            dr_, dc_ = max(nbrs, key=lambda rc: np.linalg.norm(W[er, ec] - W[rc]))
            if dc_ == ec:   # vertical neighbour -> insert a row
                r = min(er, dr_)
                new = (W[r] + W[r + 1]) / 2
                W = np.concatenate([W[:r + 1], new[None], W[r + 1:]], axis=0)
            else:           # horizontal neighbour -> insert a column
                c = min(ec, dc_)
                new = (W[:, c] + W[:, c + 1]) / 2
                W = np.concatenate([W[:, :c + 1], new[:, None], W[:, c + 1:]], axis=1)
            W = self._train_map(W, X, max(5, self.epochs // 2))

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, y: np.ndarray):
        X = X.astype(np.float64)
        self.m = X.shape[1]
        mean = X.mean(0)
        self.qe0 = float(np.sqrt(((X - mean) ** 2).sum(1)).mean())
        self.maps = []
        self._expand(X, y, parent=None, parent_qe=self.qe0, depth=1)
        self._build_leaves(X, y)
        return self

    def _expand(self, X, y, parent, parent_qe, depth):
        W = self._grow_map(X, parent_qe)
        layer = SOMLayer(len(self.maps), depth, W, parent)
        self.maps.append(layer)
        bmu, _, n, s, qe = self._unit_errors(layer.flat, X)
        layer.live = n > 0
        self.growth_log.append({"map": layer.map_id, "depth": depth, "rows": W.shape[0],
                                "cols": W.shape[1], "n": len(X), "mqe": float(qe[n > 0].mean())})
        for u in np.where(n > 0)[0]:
            members = bmu == u
            Xu, yu = X[members], y[members]
            purity = Counter(yu).most_common(1)[0][1] / len(yu)
            distinct = len(np.unique(Xu, axis=0))
            want = qe[u] >= self.tau2 * self.qe0 or purity < self.purity_target
            if (want and depth < self.max_depth and len(Xu) >= self.min_child_samples
                    and distinct >= 4 and int(n[n > 0].size) > 1):
                child_id = len(self.maps)
                self._expand(Xu, yu, parent=(layer.map_id, int(u)), parent_qe=qe[u], depth=depth + 1)
                layer.children[int(u)] = child_id

    # ------------------------------------------------------------------ leaves / mapping
    def _route(self, X):
        """Return (leaf_code, sq_distance_to_leaf_prototype) for each row of X."""
        X = X.astype(np.float64)
        leaf = np.full(len(X), -1)
        d2 = np.zeros(len(X))
        stack = [(0, np.arange(len(X)))]
        while stack:
            mid, idx = stack.pop()
            if len(idx) == 0:
                continue
            layer = self.maps[mid]
            D = sq_dist(X[idx], layer.flat)
            D[:, ~layer.live] = np.inf
            bmu = D.argmin(1)
            for u in np.unique(bmu):
                sel = idx[bmu == u]
                if int(u) in layer.children:
                    stack.append((layer.children[int(u)], sel))
                else:
                    leaf[sel] = self.leaf_code.get((mid, int(u)), -1)
                    d2[sel] = D[bmu == u, u]
        return leaf, d2

    def _build_leaves(self, X, y):
        self.leaf_code, self.leaf_key = {}, []
        for layer in self.maps:
            for u in np.where(layer.live)[0]:
                if int(u) not in layer.children:
                    self.leaf_code[(layer.map_id, int(u))] = len(self.leaf_key)
                    self.leaf_key.append((layer.map_id, int(u)))
        self.prototypes = np.array([self.maps[m].flat[u] for m, u in self.leaf_key])
        leaf, _ = self._route(X)
        self.leaf_dist, self.leaf_label, self.leaf_purity, self.leaf_n = [], [], [], []
        for c in range(len(self.leaf_key)):
            yc = y[leaf == c]
            cnt = Counter(yc)
            tot = max(1, len(yc))
            self.leaf_dist.append({k: v / tot for k, v in cnt.items()})
            lab = cnt.most_common(1)[0][0] if cnt else None
            self.leaf_label.append(lab)
            self.leaf_purity.append(cnt.most_common(1)[0][1] / tot if cnt else 0.0)
            self.leaf_n.append(len(yc))
        self.leaf_purity = np.array(self.leaf_purity)
        self.leaf_n = np.array(self.leaf_n)

    def transform(self, X):
        return self._route(X)

    # ------------------------------------------------------------------ summary
    def summary(self):
        depths = Counter(l.depth for l in self.maps)
        n = self.leaf_n
        return {"maps": len(self.maps), "max_depth": max(depths), "maps_per_depth": dict(depths),
                "leaf_clusters": len(self.leaf_key),
                "weighted_purity": float((self.leaf_purity * n).sum() / n.sum()),
                "root_grid": "x".join(map(str, self.maps[0].shape))}
