"""
pipeline.py - the GHSOM + GNG neuro-symbolic pipeline (Kautz Type 3, Neuro;Symbolic)
====================================================================================

Strict sequence, each stage reads only the previous stage's output:

  Stage 1 NEURAL     GHSOM (pretrained on labelled symptoms) -> leaf cluster code + QE
                     GNG   (initialised with GHSOM's leaves, growth cap 1) -> outliers
  Stage 2 XAI        fit record       : QE attribution + conformal p-value + shallow CART rule
                     proximity record : conformal outlier test over GNG nodes + symptom differential
  Stage 3 SYMBOLIC   reads ONLY the explanation record (never the raw vector) and resolves:

  Case 1  GHSOM accepts (p > alpha_fit) and cluster purity >= purity_min
          -> ontology looks up the cluster code -> KNOWN disease
          (low purity -> ontology reasoning over the cluster's candidate diseases)
  Case 2  GHSOM rejects -> GNG finds no accepting node -> proposes a node (cap 1 -> 0)
          -> ontology finds no consistent known disease -> NEW disease registered,
             node kept, cap reset to 1
  Case 3  GHSOM rejects -> a node registered in an earlier Case 2 accepts x and the symptom
          differential is small -> the ontology checks x against the profile registered in Case 2:
          confirmed -> OLD (earlier new) disease, GNG unchanged; not confirmed -> Case 2 path
  Other   GNG accepts a known node, or explanations disagree -> ontology reasons over the
          prefetched candidate diseases (then the full ontology) -> KNOWN, or NEW if nothing fits.
          A provisional node for a patient the ontology calls known is rejected (deleted).

Stage 3 never receives the raw vector or a raw symptom list. Its evidence is rebuilt from the
explanation fields: the fit record's cluster profile corrected by the QE-attribution deviations,
or the proximity record's symptom differential (shared + excess symptoms of the nearest node).
The conformal tests use weighted Jaccard distance to the prototypes; the QE attribution keeps the
squared-error decomposition the design specifies; the differential is Hamming.

`mode="neural"` runs the same Stage 1 + Stage 2 routing WITHOUT the ontology (neural-alone baseline):
the GHSOM leaf label or the nearest accepting GNG node label is the answer, and every empty
conformal set becomes a new disease.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .ghsom import GHSOM, sq_dist
from .gng import CappedGNG
from .ontology import Ontology
from .xai import (CARTSurrogate, ConformalScorer, binarise, jaccard_dist, qe_attribution,
                  symptom_differential)


class NeuroSymbolicPipeline:
    # True = current version. False = the earlier version, which registered a new disease from a patient in a
    # mixed cluster without first checking the new diseases registered earlier (kept only for comparison).
    recheck_registered_on_mixed = True

    def __init__(self, ghsom_params=None, alpha_fit=0.05, alpha_gng=0.01, purity_min=0.9,
                 ontology_q=0.05, cart_depth=8, min_mondrian=30, k_nearest=3, gng_epochs=2, seed=0):
        self.ghsom_params = dict(ghsom_params or {})
        self.alpha_fit, self.alpha_gng = alpha_fit, alpha_gng
        self.purity_min, self.ontology_q = purity_min, ontology_q
        self.cart_depth, self.min_mondrian, self.k = cart_depth, min_mondrian, k_nearest
        self.gng_epochs, self.seed = gng_epochs, seed

    # ================================================================== TRAINING
    def fit(self, X_train, y_train, X_cal, y_cal, rules, feature_names, known, demographic=()):
        self.names = list(feature_names)
        Xtr, Xca = X_train.astype(np.float64), X_cal.astype(np.float64)

        # ---- Stage 1a: GHSOM pretrained on labelled symptoms
        self.ghsom = GHSOM(seed=self.seed, **self.ghsom_params).fit(Xtr, y_train)
        # ---- Stage 1b: GNG initialised with the same number of nodes, then pretrained (no growth)
        self.gng = CappedGNG(seed=self.seed).init_from_ghsom(self.ghsom).pretrain(Xtr, y_train, self.gng_epochs)
        self.gng_snapshot = self._gng_state()

        # ---- Stage 2 calibration (held-out calibration patients, known diseases only)
        self.leaf_bin = binarise(self.ghsom.prototypes)
        leaf_ca, _ = self.ghsom.transform(Xca)
        s_fit = jaccard_dist(Xca > 0.5, self.ghsom.prototypes)[np.arange(len(Xca)), leaf_ca]
        self.fit_conformal = ConformalScorer(self.min_mondrian).fit(s_fit, leaf_ca)
        D = jaccard_dist(Xca > 0.5, self.gng.W)
        nn = D.argmin(1)
        self.gng_conformal = ConformalScorer(self.min_mondrian).fit(D[np.arange(len(Xca)), nn], nn)
        Bp = self.gng.W[nn] > 0.5
        ham = ((Xca > 0.5) != Bp).sum(1)
        self.h_max = float(np.quantile(ham, 0.95))          # "small differential" for Case 3
        leaf_tr, _ = self.ghsom.transform(Xtr)
        self.cart = CARTSurrogate(self.cart_depth, self.seed).fit(Xtr, leaf_tr, Xca, leaf_ca)

        # ---- Stage 3: ontology restricted to known diseases, thresholds from labelled data
        self.ontology = Ontology(rules, self.names, known, demographic, q=self.ontology_q)
        self.ontology.calibrate(np.vstack([Xtr, Xca]), np.concatenate([y_train, y_cal]))
        self.ontology_snapshot = dict(self.ontology.registry)
        self.n_new = 0
        return self

    def _gng_state(self):
        g = self.gng
        return (g.W.copy(), list(g.label), list(g.kind), g.n_assigned.copy(), g.E.copy(),
                dict(g.edges), g.purity.copy(), g.cap)

    def reset_stream(self):
        """Restore the post-training state (so several test runs start identically)."""
        g = self.gng
        (g.W, g.label, g.kind, g.n_assigned, g.E, g.edges, g.purity, g.cap) = [
            v.copy() if hasattr(v, "copy") else v for v in self.gng_snapshot]
        self.ontology.registry = {}
        self.n_new = 0

    # ================================================================== STAGE 2
    def explain(self, x):
        """Stage 1 forward pass + Stage 2 explanation record for one patient."""
        x = x.astype(np.float64)
        leaf, d2 = self.ghsom.transform(x[None])
        c, d2 = int(leaf[0]), float(d2[0])
        w = self.ghsom.prototypes[c]
        s_fit = float(jaccard_dist(x > 0.5, w)[0, 0])
        p_fit = float(self.fit_conformal.pvalue(s_fit, c))
        if p_fit > self.alpha_fit:
            cart = self.cart.explain(x, self.names)
            attr = qe_attribution(x, w, self.names)
            top = {a["symptom"] for a in attr["top"]}
            rs = set(cart["rule_symptoms"])
            return {
                "path": "fit",
                "cluster_profile": [self.names[j] for j in np.where(self.leaf_bin[c])[0]],
                "jaccard_to_cluster": s_fit,
                "cluster_code": c, "cluster_disease": self.ghsom.leaf_label[c],
                "cluster_purity": float(self.ghsom.leaf_purity[c]),
                "cluster_label_dist": self.ghsom.leaf_dist[c],
                "qe": float(np.sqrt(d2)), "p_value": p_fit,
                "p_value_pooled": self.fit_conformal.uses_pooled(c),
                "qe_attribution": attr,
                "cart": {**cart, "fidelity": self.cart.fidelity},
                "cart_agrees": cart["predicted_cluster"] == c,
                "concordance": len(top & rs) / len(top | rs) if top | rs else 1.0,
            }
        # ---------- outlier -> GNG proximity explanation
        g = self.gng
        D = g.jaccard(x)
        order = np.argsort(D, kind="stable")
        pvals = np.array([self.gng_conformal.pvalue(D[j], j if g.kind[j] == "known" else None)
                          for j in range(len(D))])
        cset = [int(j) for j in order if pvals[j] > self.alpha_gng]
        proposed = None
        if not cset:
            proposed = g.propose(x)                       # cap 1 -> 0
        nearest = []
        for j in order[: self.k]:
            j = int(j)
            nearest.append({"node": j, "disease": g.label[j], "kind": g.kind[j],
                            "jaccard": float(D[j]), "p": float(pvals[j]),
                            "differential": symptom_differential(x, g.W[j], self.names)})
        return {
            "path": "proximity",
            "ghsom_cluster": c, "ghsom_p_value": p_fit,
            "conformal_set": [{"node": j, "disease": g.label[j], "kind": g.kind[j], "p": float(pvals[j])}
                              for j in cset[:10]],
            "nearest": nearest, "proposed_node": proposed,
            "nearest_registered": self._nearest_registered(x, D, pvals),
        }

    def _nearest_registered(self, x, D, pvals):
        g = self.gng
        reg = [j for j in range(len(D)) if g.kind[j] == "new"]
        if not reg:
            return None
        j = min(reg, key=lambda r: D[r])
        return {"node": j, "disease": g.label[j], "p": float(pvals[j]),
                "differential": symptom_differential(x, g.W[j], self.names)}

    # ================================================================== STAGE 3
    @staticmethod
    def evidence(rec) -> set:
        """Symptoms Stage 3 reasons over, rebuilt ONLY from explanation fields."""
        if rec["path"] == "fit":
            a = rec["qe_attribution"]
            return (set(rec["cluster_profile"]) - set(a["expected_absent"])) | set(a["present_unusual"])
        d = rec["nearest"][0]["differential"]
        return set(d["shared"]) | set(d["excess"])

    def resolve(self, rec):
        """Symbolic stage. Reads ONLY the explanation record."""
        F = self.evidence(rec)
        onto = self.ontology
        if rec["path"] == "fit":
            if rec["cluster_purity"] >= self.purity_min:
                return {"case": "Case 1", "outcome": "known", "disease": rec["cluster_disease"],
                        "reason": f"cluster {rec['cluster_code']} accepted (p={rec['p_value']:.2f}), "
                                  f"purity {rec['cluster_purity']:.2f} -> ontology lookup"}
            cands = [d for d, s in rec["cluster_label_dist"].items() if s >= 0.1]
            best, ranked = onto.best_consistent(F, cands)
            if best is None:
                best, ranked = onto.best_consistent(F)
            if best is not None:
                return {"case": "Ontology reasoning (low purity)", "outcome": "known",
                        "disease": best["disease"], "reason": f"ontology score {best['score']:.2f} "
                        f">= theta {best['theta']:.2f}", "ontology_top": ranked}
            # Before registering a new disease, check the new diseases registered earlier in the run:
            # a patient of a returning new disease can land in a mixed cluster rather than among the outliers.
            checks = ([onto.confirm_registered(F, n) for n in onto.registry]
                      if self.recheck_registered_on_mixed else [])
            ok = [c for c in checks if c["confirmed"]]
            if ok:
                c = max(ok, key=lambda c: c["score"])
                return {"case": "Case 3", "outcome": "old_new", "disease": c["disease"],
                        "reason": f"low-purity cluster; ontology confirms earlier addition {c['disease']} "
                                  f"(score {c['score']:.2f} >= {c['theta']:.2f})", "ontology_check": c,
                        "ontology_top": ranked}
            return {"case": "Case 2", "outcome": "new", "reason": "low-purity cluster, no ontology match",
                    "ontology_top": ranked, "needs_node": True}

        # ---------------- proximity path (outlier)
        reg = rec["nearest_registered"]
        case3_check = None
        if (reg is not None and reg["p"] > self.alpha_gng and reg["differential"]["hamming"] <= self.h_max):
            case3_check = onto.confirm_registered(F, reg["disease"])     # "Ontology confirms earlier disease?"
            if case3_check["confirmed"]:
                return {"case": "Case 3", "outcome": "old_new", "disease": reg["disease"], "node": reg["node"],
                        "reason": f"registered node {reg['node']} accepts (p={reg['p']:.2f}), differential "
                                  f"{reg['differential']['hamming']} <= {self.h_max:.0f}; ontology confirms "
                                  f"profile (score {case3_check['score']:.2f} >= {case3_check['theta']:.2f})",
                        "ontology_check": case3_check}
            # not confirmed -> Case 2 path (check known diseases, else register a new one)
        cands = list(dict.fromkeys([c["disease"] for c in rec["conformal_set"] if c["kind"] == "known"] +
                                   [n["disease"] for n in rec["nearest"] if n["kind"] == "known"]))
        best, ranked = onto.best_consistent(F, cands)
        where = "prefetched candidates"
        if best is None:
            best, ranked = onto.best_consistent(F)
            where = "full ontology"
        if best is not None:
            gng_ok = any(c["disease"] == best["disease"] for c in rec["conformal_set"])
            return {"case": "GNG -> known" if gng_ok else "Ontology override -> known",
                    "outcome": "known", "disease": best["disease"],
                    "reason": f"{where}: score {best['score']:.2f} >= theta {best['theta']:.2f}",
                    "reject_node": rec["proposed_node"], "ontology_top": ranked,
                    "case3_check": case3_check}
        return {"case": "Case 2", "outcome": "new", "reason": "no known disease or earlier addition fits",
                "node": rec["proposed_node"], "needs_node": rec["proposed_node"] is None,
                "ontology_top": ranked, "case3_check": case3_check}

    def resolve_neural(self, rec):
        """Neural-alone baseline: same Stage 1/2 routing, no ontology."""
        if rec["path"] == "fit":
            return {"case": "Case 1", "outcome": "known", "disease": rec["cluster_disease"]}
        if rec["conformal_set"]:
            top = rec["conformal_set"][0]
            if top["kind"] == "new":
                return {"case": "Case 3", "outcome": "old_new", "disease": top["disease"], "node": top["node"]}
            return {"case": "GNG -> known", "outcome": "known", "disease": top["disease"]}
        return {"case": "Case 2", "outcome": "new", "node": rec["proposed_node"],
                "needs_node": rec["proposed_node"] is None}

    # ================================================================== DIAGNOSE (stream)
    def diagnose(self, x, mode="neurosymbolic", keep_record=False):
        rec = self.explain(x)
        out = self.resolve(rec) if mode == "neurosymbolic" else self.resolve_neural(rec)
        g = self.gng
        if out["outcome"] == "new":
            q = out.get("node")
            if q is None:                       # cap was used / fit-path novelty -> make the node now
                if g.cap <= 0:                  # cannot happen with immediate resolution; safety
                    g.cap = g.cap0
                q = g.propose(x.astype(np.float64))
            self.n_new += 1
            name = f"NEW-{self.n_new}"
            g.commit(q, name)
            if mode == "neurosymbolic":
                prof = [self.names[j] for j in np.where(g.W[q] > 0.5)[0]]   # node prototype = profile
                self.ontology.register(name, prof, q)
            out["disease"] = name
        elif out["outcome"] == "old_new":            # GNG unchanged: no node moved or added
            if out["disease"] in self.ontology.registry:
                self.ontology.registry[out["disease"]]["cases"] += 1
        elif out.get("reject_node") is not None:
            g.reject(out["reject_node"])
        if keep_record:
            out["record"] = rec
        return out

    def summary(self):
        return {**{f"ghsom_{k}": v for k, v in self.ghsom.summary().items()},
                "gng_init_nodes": self.gng.n_init, "cart_fidelity": self.cart.fidelity,
                "cart_depth": self.cart_depth, "h_max": self.h_max}


# ====================================================================== SYMBOLIC-ALONE
def symbolic_alone(ontology: Ontology, X, names):
    out = []
    for x in X:
        F = {names[j] for j in np.where(x > 0.5)[0]}
        best, _ = ontology.best_consistent(F)
        out.append(best["disease"] if best else "UNKNOWN")
    return np.array(out, dtype=object)
