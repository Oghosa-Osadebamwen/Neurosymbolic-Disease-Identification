"""
test.py - evaluate the trained pipelines
========================================

For every trained model (models/<dataset>_seed<k>.pkl) the test patients - known-disease test
patients mixed with ALL patients of the held-out diseases - are streamed one at a time, in a
fixed random order, through three systems:

  * Symbolic-alone : the ontology applied to the raw symptoms (no networks). Unmatched
                     patients go into a single "UNKNOWN" bucket.
  * Neural-alone   : GHSOM + GNG with the same Stage 2 routing, no ontology.
  * Neural-alone, strict GNG : as above but the GNG is as strict as GHSOM (alpha_gng = alpha_fit),
                     so the networks alone can declare new diseases (a fairer neural baseline).
  * Neuro-symbolic : the full Stage 1 -> Stage 2 -> Stage 3 pipeline.

Outputs (results/):
  per_run.csv           every metric, every dataset x seed x method
  summary.csv           mean and standard deviation over seeds
  case_counts.csv       how often each Case / route fired (neuro-symbolic)
  example_records.json  one full explanation record + decision for Case 1, Case 2 and Case 3
  sweep_alpha.csv       sensitivity to the GHSOM conformal rejection level alpha_fit
  sweep_q.csv           sensitivity to the ontology threshold quantile q

Usage:  python test.py [--datasets kaggle ddxplus] [--no-sweep]
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import config
from neurosym.evaluate import open_set_metrics
from neurosym.pipeline import symbolic_alone
from train import load


def run_stream(pipe, X, mode, keep_examples=False, details=None):
    """Stream the test patients one at a time. If `details` is a list, one small dict per patient is
    appended to it (route, case, explanation statistics) for the XAI and per-disease analyses."""
    pipe.reset_stream()
    preds, cases, examples = [], [], {}
    for i, x in enumerate(X):
        out = pipe.diagnose(x, mode=mode, keep_record=keep_examples or details is not None)
        preds.append(out["disease"])
        cases.append(out["case"])
        if details is not None:
            rec = out["record"]
            d = {"case": out["case"], "outcome": out["outcome"], "path": rec["path"]}
            if rec["path"] == "fit":
                d.update(fit_p=rec["p_value"], cart_agrees=rec["cart_agrees"], concordance=rec["concordance"],
                         purity=rec["cluster_purity"])
            chk = out.get("ontology_check") or out.get("case3_check")
            if chk is not None:
                d["case3_checked"] = True
                d["case3_confirmed"] = bool(chk["confirmed"])
            details.append(d)
        if keep_examples and out["case"] not in examples:
            examples[out["case"]] = {"patient_index_in_stream": i, **out}
    return np.array(preds, dtype=object), cases, examples


def supervised_rows(ds, split, base):
    """Label-trained references inside the same open-set protocol (trained on the known diseases)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    Xtr, ytr = ds.X[split.train], ds.y[split.train]
    Xca = ds.X[split.calib]
    X, y = ds.X[split.test], ds.y[split.test]
    rows = []
    lr = LogisticRegression(max_iter=3000).fit(Xtr, ytr)
    rows.append({**base, "method": "Supervised: logistic regression",
                 **open_set_metrics(y, lr.predict(X), split.novel, split.known)})
    rf = RandomForestClassifier(n_estimators=300, random_state=base["seed"]).fit(Xtr, ytr)
    rows.append({**base, "method": "Supervised: random forest",
                 **open_set_metrics(y, rf.predict(X), split.novel, split.known)})
    # open-set version: flag 'new' when the top class probability is below the 5th percentile of the
    # calibration patients' top probabilities (same 5% false-alarm target as the GHSOM conformal test)
    thr = np.quantile(rf.predict_proba(Xca).max(1), 0.05)
    P = rf.predict_proba(X)
    pred = np.where(P.max(1) < thr, "UNKNOWN", rf.classes_[P.argmax(1)]).astype(object)
    rows.append({**base, "method": "Supervised: random forest + confidence threshold",
                 **open_set_metrics(y, pred, split.novel, split.known)})
    return rows


def evaluate_model(path, ds, sweep=False):
    blob = pickle.load(open(path, "rb"))
    pipe, split = blob["pipeline"], blob["split"]
    X, y = ds.X[split.test], ds.y[split.test]
    rows, case_rows, sweep_a, sweep_q, patient_rows = [], [], [], [], []
    base = {"dataset": ds.name, "seed": blob["info"]["seed"]}

    pred = symbolic_alone(pipe.ontology, X, pipe.names)
    rows.append({**base, "method": "Symbolic-alone", **open_set_metrics(y, pred, split.novel, split.known)})

    pred, cases, _ = run_stream(pipe, X, "neural")
    rows.append({**base, "method": "Neural-alone (GHSOM+GNG)", **open_set_metrics(y, pred, split.novel, split.known)})
    # fairer neural baseline: GNG as strict as GHSOM (alpha_gng = alpha_fit), so it can say "new"
    ag = pipe.alpha_gng
    pipe.alpha_gng = pipe.alpha_fit
    pred, cases, _ = run_stream(pipe, X, "neural")
    pipe.alpha_gng = ag
    rows.append({**base, "method": "Neural-alone, strict GNG", **open_set_metrics(y, pred, split.novel, split.known)})

    det = []
    pred, cases, examples = run_stream(pipe, X, "neurosymbolic", keep_examples=True, details=det)
    m = open_set_metrics(y, pred, split.novel, split.known)
    # XAI statistics (Stage 2), on the test stream
    nov = np.isin(y, split.novel)
    fit = [i for i, d in enumerate(det) if d["path"] == "fit"]
    fit_known = [i for i in fit if not nov[i]]
    m["cart_fidelity_calib"] = pipe.cart.fidelity
    m["cart_agreement_test"] = float(np.mean([det[i]["cart_agrees"] for i in fit])) if fit else np.nan
    m["concordance_test"] = float(np.mean([det[i]["concordance"] for i in fit])) if fit else np.nan
    m["known_rejected_by_ghsom"] = float(np.mean([det[i]["path"] != "fit" for i in np.where(~nov)[0]]))
    m["novel_rejected_by_ghsom"] = float(np.mean([det[i]["path"] != "fit" for i in np.where(nov)[0]]))
    c3 = [d for d in det if d.get("case3_checked")]
    m["case3_checks"] = len(c3)
    m["case3_confirmed_share"] = float(np.mean([d["case3_confirmed"] for d in c3])) if c3 else np.nan
    for i, d in enumerate(det):
        patient_rows.append({**base, "true": y[i], "pred": pred[i], "novel": bool(nov[i]),
                             "case": d["case"], "outcome": d["outcome"]})
    m["gng_nodes_after_test"] = len(pipe.gng.W)
    rows.append({**base, "method": "Neuro-symbolic (full)", **m})
    rows += supervised_rows(ds, split, base)
    for c, n in Counter(cases).items():
        idx = np.array([k == c for k in cases])
        case_rows.append({**base, "case": c, "n": n, "share_novel_patients": float(nov[idx].mean()),
                          "accuracy_on_known": float((pred[idx & ~nov] == y[idx & ~nov]).mean())
                          if (idx & ~nov).any() else np.nan})
    for ex in examples.values():
        ex["true_disease"] = str(y[ex["patient_index_in_stream"]])

    if sweep:
        a0 = pipe.alpha_fit
        for a in [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]:
            pipe.alpha_fit = a
            p, _, _ = run_stream(pipe, X, "neurosymbolic")
            sweep_a.append({**base, "alpha_fit": a, **open_set_metrics(y, p, split.novel, split.known)})
        pipe.alpha_fit = a0
        tr = np.concatenate([split.train, split.calib])
        for q in [0.01, 0.05, 0.10, 0.20, 0.30]:
            pipe.ontology.q = q
            pipe.ontology.calibrate(ds.X[tr], ds.y[tr])
            p, _, _ = run_stream(pipe, X, "neurosymbolic")
            sweep_q.append({**base, "ontology_q": q, **open_set_metrics(y, p, split.novel, split.known)})
    stage1 = {**base, **{k: v for k, v in blob["info"].items() if k.startswith(("ghsom", "gng", "cart"))},
              "novel_diseases": "; ".join(split.novel), "n_train": len(split.train),
              "n_calib": len(split.calib), "n_test": len(split.test)}
    return rows, case_rows, examples, sweep_a, sweep_q, patient_rows, stage1


MAIN_COLS = ["known_accuracy", "known_macro_f1", "known_false_new_rate", "novel_detection_rate",
             "open_set_accuracy", "new_groups_created", "novel_group_purity", "novel_group_ARI",
             "novel_diseases_recovered", "false_new_groups", "supported_new_groups",
             "supported_false_groups", "supported_novel_recovered"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(config.DATASETS))
    ap.add_argument("--no-sweep", action="store_true")
    args = ap.parse_args()
    out = Path(config.RESULTS_DIR)
    out.mkdir(exist_ok=True)
    allrows, allcases, allex, sa, sq, pat, st1 = [], [], {}, [], [], [], []
    for name in args.datasets:
        ds = load(name)
        for path in sorted(Path(config.MODEL_DIR).glob(f"{name}_seed*.pkl")):
            r, c, e, a, q, pr, s1 = evaluate_model(path, ds, sweep=not args.no_sweep)
            allrows += r; allcases += c; sa += a; sq += q; pat += pr; st1.append(s1)
            allex.setdefault(name, e)                 # examples from the first seed
            print(f"evaluated {path.name}")
    df = pd.DataFrame(allrows)
    df.to_csv(out / "per_run.csv", index=False)
    summ = df.groupby(["dataset", "method"])[MAIN_COLS].agg(["mean", "std"])
    summ.to_csv(out / "summary.csv")
    pd.DataFrame(allcases).to_csv(out / "case_counts.csv", index=False)
    pd.DataFrame(pat).to_csv(out / "per_patient.csv", index=False)
    pd.DataFrame(st1).to_csv(out / "stage1_models.csv", index=False)
    xai = ["cart_fidelity_calib", "cart_agreement_test", "concordance_test", "known_rejected_by_ghsom",
           "novel_rejected_by_ghsom", "case3_checks", "case3_confirmed_share"]
    df[df.method == "Neuro-symbolic (full)"].groupby("dataset")[xai].agg(["mean", "std"]).to_csv(out / "xai_stats.csv")
    if sa:
        pd.DataFrame(sa).to_csv(out / "sweep_alpha.csv", index=False)
        pd.DataFrame(sq).to_csv(out / "sweep_q.csv", index=False)
    with open(out / "example_records.json", "w") as fh:
        json.dump(allex, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    pd.set_option("display.width", 220)
    fmt = df.groupby(["dataset", "method"])[MAIN_COLS].agg(lambda s: f"{s.mean():.3f} ± {s.std():.3f}")
    print("\nRESULTS (mean ± sd over seeds)\n")
    print(fmt.T.to_string())


if __name__ == "__main__":
    main()
