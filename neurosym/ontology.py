"""
ontology.py - Stage 3 symbolic knowledge
========================================

The ontology is the rule file shipped with each dataset (independent_rules.csv for Kaggle,
ddxplus_rules.csv for DDXPlus). Each disease d has criteria I_d; a criterion is satisfied if
ANY of its listed findings is present. Criteria are "symptom" (weight 1) or "risk" (weight 0.5).

For a set of observed findings F (taken from the Stage 2 explanation record):

    coverage_d  = (sum_{i in I_d, symptom} [i sat] + 0.5 sum_{i in I_d, risk} [i sat])
                  / (|I_d^symptom| + 0.5 |I_d^risk|)          - required findings present
    explained_d = |F_clin intersect U_d| / |F_clin|           - observed findings that d accounts for
                  (U_d = all findings named by d's criteria; F_clin excludes sex/age)
    score_d     = 2 * coverage_d * explained_d / (coverage_d + explained_d)

Consistency: d is CONSISTENT with F if score_d >= theta_d, where theta_d is the q-quantile of
score_d over the known training patients of d (q = 0.05 by default; data-driven, per disease).

Held-out ("novel") diseases are REMOVED from the ontology - a new disease is, by definition,
one the ontology does not know. Confirmed new diseases are added to a provisional registry
with the GNG prototype's symptom profile.
"""
from __future__ import annotations

import numpy as np


class Ontology:
    def __init__(self, rules: dict, feature_names: list, known: list, demographic=(), q=0.05):
        self.names = feature_names
        self.q = q
        self.demo = set(demographic)
        self.rules = {}
        for d in known:
            items = rules[d]
            self.rules[d] = [(frozenset(feature_names[i] for i in it.features), it.kind, it.phrase)
                             for it in items]
        self.universe = {d: frozenset().union(*[f for f, _, _ in items]) for d, items in self.rules.items()}
        self.registry: dict[str, dict] = {}      # provisional new diseases
        self.theta: dict[str, float] = {}

    # ------------------------------------------------------------------ scoring
    def score(self, findings: set, d: str) -> dict:
        items = self.rules[d]
        ws = wt = 0.0
        sat, unsat = [], []
        for feats, kind, phrase in items:
            w = 1.0 if kind == "symptom" else 0.5
            wt += w
            if feats & findings:
                ws += w
                sat.append(phrase)
            else:
                unsat.append(phrase)
        cov = ws / wt if wt else 0.0
        clin = findings - self.demo
        exp = len(clin & self.universe[d]) / len(clin) if clin else 0.0
        s = 2 * cov * exp / (cov + exp) if cov + exp > 0 else 0.0
        return {"disease": d, "score": s, "coverage": cov, "explained": exp,
                "satisfied": sat, "unsatisfied": unsat}

    def calibrate(self, X: np.ndarray, y: np.ndarray):
        for d in self.rules:
            rows = X[y == d]
            sc = [self.score(self._findings(r), d)["score"] for r in rows]
            self.theta[d] = float(np.quantile(sc, self.q)) if sc else 1.0
        return self

    def _findings(self, row):
        return {self.names[j] for j in np.where(row > 0.5)[0]}

    # ------------------------------------------------------------------ reasoning
    def best_consistent(self, findings: set, candidates=None):
        """The TOP-ranked candidate, if it clears its own threshold; plus the ranking.

        Only the ontology's preferred diagnosis may be accepted. Accepting *any* disease that
        clears its threshold would make the union of 40+ acceptance regions cover almost every
        patient, so nothing could ever be called new.
        """
        cands = [c for c in (candidates or self.rules) if c in self.rules]
        if not cands:
            return None, []
        ranked = sorted((self.score(findings, d) for d in cands), key=lambda r: -r["score"])
        for r in ranked:
            r["theta"] = self.theta[r["disease"]]
            r["consistent"] = r["score"] >= r["theta"]
        top = ranked[0]
        return (top if top["consistent"] else None), ranked[:3]

    def register(self, name: str, profile: list, source_node: int):
        self.registry[name] = {"profile": sorted(profile), "gng_node": source_node, "cases": 1}

    def confirm_registered(self, findings: set, name: str) -> dict:
        """Case 3 check: does the patient fit the profile registered for an earlier new disease?
        Same score as for known diseases, with each profile symptom treated as one criterion.
        Threshold theta_reg = median of the known diseases' thresholds (a typical known-disease bar)."""
        P = set(self.registry[name]["profile"]) - self.demo
        clin = findings - self.demo
        cov = len(clin & P) / len(P) if P else 0.0
        exp = len(clin & P) / len(clin) if clin else 0.0
        s = 2 * cov * exp / (cov + exp) if cov + exp > 0 else 0.0
        theta = float(np.median(list(self.theta.values())))
        return {"disease": name, "score": s, "coverage": cov, "explained": exp, "theta": theta,
                "confirmed": s >= theta}
