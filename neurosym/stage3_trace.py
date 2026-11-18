"""
stage3_trace.py - a step-by-step account of how Stage 3 reached its conclusion
==============================================================================

`trace(pipe, onto, rec)` walks the same decision procedure as `NeuroSymbolicPipeline.resolve`
and writes down every test it makes, with the numbers, so the app can show the reasoning:

    1. the evidence Stage 3 reads (rebuilt from the Stage 2 explanation only)
    2. which route the patient is on and why
    3. any check against new diseases registered earlier in the session
    4. the ontology ranking: for each candidate, the rule items met and missed, coverage,
       explained share, score and the disease's own threshold
    5. the decision rule applied, and the conclusion in plain words

`onto` must be the ontology as it stood BEFORE the patient was decided (the registry can grow
when a new disease is registered). The trace also returns its own conclusion, so tests can check
that it reproduces the pipeline's decision exactly.
"""
from __future__ import annotations


def _row(r):
    return {"disease": r["disease"], "score": r["score"], "coverage": r["coverage"],
            "explained": r["explained"], "theta": r.get("theta"), "passes": r["score"] >= r.get("theta", 1.0),
            "met": list(r.get("satisfied", [])), "missed": list(r.get("unsatisfied", []))}


def _ranking(onto, F, cands=None):
    best, ranked = onto.best_consistent(F, cands)
    return best, [_row(r) for r in ranked]


def trace(pipe, onto, rec):
    F = pipe.evidence(rec)
    steps = []
    T = {"evidence": sorted(F), "steps": steps, "outcome": None, "disease": None, "conclusion": ""}

    # ---------------------------------------------------------------- 1. evidence
    if rec["path"] == "fit":
        a = rec["qe_attribution"]
        ev = ("Stage 3 never sees the raw patient. It rebuilds the evidence from the Stage 2 fit explanation: "
              f"the symptoms of cluster {rec['cluster_code']}'s profile, minus those the explanation marks as "
              "expected but absent, plus those it marks as present but unusual.")
        extra = {"profile": rec["cluster_profile"], "absent": a["expected_absent"], "unusual": a["present_unusual"]}
    else:
        d = rec["nearest"][0]["differential"]
        ev = ("Stage 3 never sees the raw patient. It rebuilds the evidence from the Stage 2 proximity explanation: "
              f"the symptoms the patient shares with the nearest GNG node (#{rec['nearest'][0]['node']}) plus the "
              "extra symptoms the patient has that the node lacks.")
        extra = {"shared": d["shared"], "extra": d["excess"]}
    steps.append({"key": "evidence", "title": "Read the evidence", "text": ev, "extra": extra})

    def done(outcome, disease, text):
        T["outcome"], T["disease"], T["conclusion"] = outcome, disease, text
        steps.append({"key": "decision", "title": "Conclude", "text": text, "outcome": outcome})
        return T

    def rank_step(title, text, rows, where):
        steps.append({"key": "rank", "title": title, "text": text, "rows": rows, "where": where})

    rule_txt = ("Each candidate is scored against its rules from published sources. Only the top-ranked "
                "disease may be accepted, and only if its score reaches that disease's own threshold.")

    # ---------------------------------------------------------------- fit path
    if rec["path"] == "fit":
        pur = rec["cluster_purity"]
        if pur >= pipe.purity_min:
            steps.append({"key": "route", "title": "Choose the route",
                          "text": f"Stage 1 accepted the patient into GHSOM cluster {rec['cluster_code']} "
                                  f"(p = {rec['p_value']:.2f}). The cluster is **pure**: {pur:.0%} of its training "
                                  f"patients had {rec['cluster_disease']} (needs {pipe.purity_min:.0%}). So the "
                                  "cluster code is looked up directly in the ontology; no scoring is needed."})
            d = rec["cluster_disease"]
            if d in onto.rules:
                r = onto.score(F, d)
                r["theta"] = onto.theta[d]
                rank_step("Show the published rules for the looked-up disease",
                          f"For transparency only: a pure cluster is trusted without scoring, so this match "
                          f"does not affect the decision. It shows which published signs of {d} the evidence "
                          "contains and which it lacks.", [_row(r)], "lookup")
            return done("known", d, f"Cluster {rec['cluster_code']} is the {d} cluster → **known disease: {d}**.")

        cands = [d for d, s in rec["cluster_label_dist"].items() if s >= 0.1]
        mix = ", ".join(f"{d} {s:.0%}" for d, s in sorted(rec["cluster_label_dist"].items(), key=lambda kv: -kv[1])
                        if s >= 0.1)
        steps.append({"key": "route", "title": "Choose the route",
                      "text": f"Stage 1 accepted the patient into GHSOM cluster {rec['cluster_code']}, but the cluster "
                              f"is **mixed** (purity {pur:.0%}, below {pipe.purity_min:.0%}): {mix}. A lookup would "
                              "be unsafe, so the ontology reasons over the evidence instead, starting with the "
                              "diseases that make up at least 10% of the cluster."})
        best, rows = _ranking(onto, F, cands)
        rank_step("Score the cluster's diseases", rule_txt, rows, "cluster diseases")
        if best is None:
            best, rows = _ranking(onto, F)
            rank_step("None passed: score every known disease", "The top-ranked cluster disease did not reach its "
                      "threshold, so every disease in the ontology is scored.", rows, "full ontology")
        if best is not None:
            return done("known", best["disease"],
                        f"{best['disease']} ranks first and its score {best['score']:.2f} reaches its threshold "
                        f"{best['theta']:.2f} → **known disease: {best['disease']}**.")
        checks = ([onto.confirm_registered(F, n) for n in onto.registry]
                  if getattr(pipe, "recheck_registered_on_mixed", False) else [])
        if checks:
            steps.append({"key": "registered", "title": "Check the new diseases registered earlier",
                          "text": "No known disease fits. Before declaring a new disease, the patient is compared "
                                  "with the profile of each new disease registered earlier in this session.",
                          "rows": [{"disease": c["disease"], "score": c["score"], "coverage": c["coverage"],
                                    "explained": c["explained"], "theta": c["theta"], "passes": c["confirmed"],
                                    "met": [], "missed": []} for c in checks]})
        ok = [c for c in checks if c["confirmed"]]
        if ok:
            c = max(ok, key=lambda c: c["score"])
            return done("old_new", c["disease"],
                        f"No known disease fits, but the registered profile {c['disease']} does (score "
                        f"{c['score']:.2f} ≥ {c['theta']:.2f}) → **old disease (added earlier): {c['disease']}**.")
        return done("new", None, "No known disease reaches its threshold and no earlier new disease fits → "
                                 "**new disease**: it is registered with the GNG node's symptom profile.")

    # ---------------------------------------------------------------- proximity path
    cs = rec["conformal_set"]
    steps.append({"key": "route", "title": "Choose the route",
                  "text": f"Stage 1 rejected the patient as an **outlier** for every GHSOM cluster (p = "
                          f"{rec['ghsom_p_value']:.3f}), so the patient went to the GNG. "
                          + (f"{len(cs)} GNG node(s) accept the patient." if cs else
                             "**No GNG node accepts the patient**, so the GNG proposed one new node.")})
    reg = rec["nearest_registered"]
    if reg is not None:
        ok_p = reg["p"] > pipe.alpha_gng
        ok_h = reg["differential"]["hamming"] <= pipe.h_max
        txt = (f"The nearest node registered earlier is #{reg['node']} ({reg['disease']}): it "
               f"{'accepts' if ok_p else 'does not accept'} the patient (p = {reg['p']:.2f}, needs above "
               f"{pipe.alpha_gng}) and the symptom difference is {reg['differential']['hamming']} "
               f"(allowed {pipe.h_max:.0f}).")
        row = None
        if ok_p and ok_h:
            c = onto.confirm_registered(F, reg["disease"])
            row = [{"disease": c["disease"], "score": c["score"], "coverage": c["coverage"],
                    "explained": c["explained"], "theta": c["theta"], "passes": c["confirmed"], "met": [], "missed": []}]
            txt += " The ontology then checks the evidence against that registered profile."
            steps.append({"key": "registered", "title": "Check the new diseases registered earlier", "text": txt,
                          "rows": row})
            if c["confirmed"]:
                return done("old_new", reg["disease"],
                            f"The registered profile {reg['disease']} is confirmed (score {c['score']:.2f} ≥ "
                            f"{c['theta']:.2f}) → **old disease (added earlier): {reg['disease']}**. The GNG is "
                            "unchanged.")
        else:
            txt += " So it is not considered further."
            steps.append({"key": "registered", "title": "Check the new diseases registered earlier", "text": txt})
    cands = list(dict.fromkeys([c["disease"] for c in cs if c["kind"] == "known"] +
                               [n["disease"] for n in rec["nearest"] if n["kind"] == "known"]))
    best, rows = _ranking(onto, F, cands)
    rank_step("Score the diseases of the nearest GNG nodes",
              "Candidates are the known diseases of the GNG nodes that accept the patient or lie nearest. " + rule_txt,
              rows, "prefetched candidates")
    where = "nearest nodes"
    if best is None:
        best, rows = _ranking(onto, F)
        where = "full ontology"
        rank_step("None passed: score every known disease", "The top-ranked candidate did not reach its threshold, "
                  "so every disease in the ontology is scored.", rows, "full ontology")
    if best is not None:
        gng_ok = any(c["disease"] == best["disease"] for c in cs)
        how = ("the GNG also accepts it" if gng_ok else
               "the GNG did not accept it, so the ontology overrides the neural stage")
        return done("known", best["disease"],
                    f"{best['disease']} ranks first in the {where} and its score {best['score']:.2f} reaches its "
                    f"threshold {best['theta']:.2f}; {how} → **known disease: {best['disease']}**.")
    return done("new", None, "No known disease reaches its threshold, and no earlier new disease is confirmed → "
                             "**new disease**: the GNG keeps the proposed node and the disease is registered with "
                             "that node's symptom profile.")
