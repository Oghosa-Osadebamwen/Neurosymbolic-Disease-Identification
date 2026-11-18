"""
xai.py - Stage 2 explanation models
===================================

FIT EXPLANATIONS (GHSOM path - "why x belongs to cluster c and is not an outlier")
  1. Quantisation-error attribution. The squared QE decomposes exactly over symptoms:
         QE^2(x, c) = sum_j (x_j - w_cj)^2 = sum_j a_j
     a_j is symptom j's share of the misfit. Sign tells the clinician the direction:
         x_j = 1, w_cj small  -> "present but unusual for this cluster"
         x_j = 0, w_cj large  -> "expected for this cluster but absent"
     The QE is also turned into a conformal p-value (below) - the "error percentile".
  2. Shallow CART surrogate. A depth-limited decision tree fitted once to the pretrained GHSOM
     cluster codes (GHSOM is fixed between retrainings). It gives the human reader an if-then
     rule; fidelity is measured on held-out calibration patients. Its agreement is NOT used
     by the symbolic stage as evidence (it imitates GHSOM).

PROXIMITY EXPLANATIONS (GNG path - "how close is the outlier to each GNG cluster")
  3. Conformal outlier test (split / Mondrian conformal prediction, Vovk et al. 2005).
     Non-conformity score = weighted JACCARD distance between the patient's symptom vector and
     the cluster prototype (the design asks for Hamming or Jaccard on binary symptoms):
         s(x, j) = 1 - sum_k min(x_k, w_jk) / sum_k max(x_k, w_jk)
     (= ordinary Jaccard when the prototype is binary) With calibration scores S_j of cluster j,
         p_j(x) = (1 + #{s in S_j : s >= s(x, j)}) / (|S_j| + 1)
     Clusters with fewer than `min_mondrian` calibration cases use the pooled score
     distribution of all clusters (a newly created GNG node always does - weak statistics).
     Conformal set  Gamma(x) = { j : p_j(x) > alpha }.
         empty -> likely new disease,  one -> clear match,  several -> ambiguous.
  4. Symptom differential. For the nearest k nodes, with binarised prototype b_j = [w_j > 0.5]:
         shared = x AND b_j,   lacks = (NOT x) AND b_j,   excess = x AND (NOT b_j)
     Hamming distance = |lacks| + |excess|. This is the symptom-by-symptom comparison.

All explanation outputs are plain dictionaries (the "explanation record") - Stage 3 reads only
these records.
"""
from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeClassifier


class ConformalScorer:
    """Mondrian split-conformal p-values over clusters, with pooled fallback."""

    def __init__(self, min_mondrian=30):
        self.min_mondrian = min_mondrian

    def fit(self, scores: np.ndarray, groups: np.ndarray):
        self.pooled = np.sort(scores)
        self.by_group = {g: np.sort(scores[groups == g]) for g in np.unique(groups)
                         if (groups == g).sum() >= self.min_mondrian}
        return self

    def pvalue(self, score: float | np.ndarray, group=None):
        ref = self.by_group.get(group, self.pooled)
        n = len(ref)
        ge = n - np.searchsorted(ref, score, side="left")
        return (1 + ge) / (n + 1)

    def uses_pooled(self, group):
        return group not in self.by_group


def binarise(W):
    return np.asarray(W) > 0.5


def jaccard_dist(xb: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Weighted (Ruzicka) Jaccard distance between binary patient vector(s) and prototype rows W:
           d(x, w) = 1 - sum_j min(x_j, w_j) / sum_j max(x_j, w_j)
    For binary x: sum min = x.w and sum max = |x| + sum(w) - x.w. When w is binary this is exactly
    the ordinary Jaccard distance; unlike binarising first, it never collapses to 1 for a cluster
    whose prototype has no symptom above 0.5."""
    X = np.atleast_2d(xb).astype(np.float64)
    W = np.atleast_2d(W).astype(np.float64)
    inter = X @ W.T
    union = X.sum(1)[:, None] + W.sum(1)[None, :] - inter
    return np.where(union > 1e-12, 1.0 - inter / np.maximum(union, 1e-12), 0.0)


def qe_attribution(x, w, names, top=5):
    """Exact decomposition QE^2 = sum_j (x_j - w_j)^2, reported two ways:
    * top      : the `top` symptoms with the largest share of the misfit (for the human reader)
    * deviations from the cluster's typical profile b = [w > 0.5], i.e. every symptom whose
      term exceeds 0.25 (|x_j - w_j| > 0.5): present-but-unusual and expected-but-absent.
    Together with the cluster profile this is a complete account of how X differs from its cluster."""
    a = (x - w) ** 2
    order = np.argsort(-a)[:top]
    top_list = []
    for j in order:
        if a[j] < 1e-6:
            break
        top_list.append({"symptom": names[j], "share": float(a[j] / max(a.sum(), 1e-12)),
                         "direction": "present, unusual for cluster" if x[j] > w[j]
                         else "expected for cluster, absent"})
    dev = np.abs(x - w) > 0.5
    return {"top": top_list,
            "present_unusual": [names[j] for j in np.where(dev & (x > 0.5))[0]],
            "expected_absent": [names[j] for j in np.where(dev & (x <= 0.5))[0]]}


class CARTSurrogate:
    def __init__(self, max_depth=8, seed=0):
        self.tree = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)

    def fit(self, X, codes, X_hold, codes_hold):
        self.tree.fit(X, codes)
        self.fidelity = float((self.tree.predict(X_hold) == codes_hold).mean()) if len(X_hold) else np.nan
        return self

    def explain(self, x, names):
        t = self.tree.tree_
        node, conds, used = 0, [], set()
        while t.children_left[node] != -1:
            f = t.feature[node]
            if x[f] <= t.threshold[node]:
                conds.append(f"NOT {names[f]}")
                node = t.children_left[node]
            else:
                conds.append(names[f])
                used.add(names[f])
                node = t.children_right[node]
            used.add(names[f])
        pred = int(self.tree.classes_[np.argmax(t.value[node])])
        return {"rule": " AND ".join(conds) or "(root)", "predicted_cluster": pred,
                "rule_symptoms": sorted(used)}


def symptom_differential(x, w, names):
    b = w > 0.5
    xb = x > 0.5
    shared = np.where(xb & b)[0]
    lacks = np.where(~xb & b)[0]
    excess = np.where(xb & ~b)[0]
    return {"shared": [names[j] for j in shared], "lacks": [names[j] for j in lacks],
            "excess": [names[j] for j in excess], "hamming": int(len(lacks) + len(excess))}
