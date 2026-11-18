"""
diagnose.py - try the trained pipeline on your own patients, one at a time
=========================================================================

Run train.py first (it creates models/<dataset>_seed<k>.pkl).

Examples
--------
  # what can I type? (symptom names, known diseases, diseases held out as "new")
  python diagnose.py --list-symptoms
  python diagnose.py --list-diseases

  # one patient, symptoms separated by commas
  python diagnose.py --patient "itching,skin_rash,nodal_skin_eruptions,dischromic_patches"

  # several patients in a row: the GNG and the new-disease registry carry over between them,
  # so a second patient of the same unknown disease can be recognised (Case 3)
  python diagnose.py --patient "a,b,c" --patient "a,b,d"

  # automatic demo: a known-disease patient, then two patients of a held-out (unknown) disease
  python diagnose.py --demo
  python diagnose.py --demo --novel "Diabetes"      # pick the held-out disease yourself

Options: --dataset kaggle|ddxplus (default kaggle), --seed 0..4 (default 0), --record (print full record)
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
from pathlib import Path

import numpy as np

import config
from train import load


def load_model(dataset, seed):
    path = Path(config.MODEL_DIR) / f"{dataset}_seed{seed}.pkl"
    if not path.exists():
        sys.exit(f"Model {path} not found - run:  python train.py --datasets {dataset} --seeds {seed}")
    blob = pickle.load(open(path, "rb"))
    return blob["pipeline"], blob["split"]


def to_vector(pipe, symptoms):
    idx = {n: i for i, n in enumerate(pipe.names)}
    x = np.zeros(len(pipe.names))
    unknown = []
    for s in symptoms:
        s = s.strip()
        if not s:
            continue
        if s in idx:
            x[idx[s]] = 1
        else:
            unknown.append(s)
    if unknown:
        sys.exit(f"Unknown symptom name(s): {unknown}\nUse --list-symptoms to see valid names.")
    return x


def show(n, x, out, pipe, full_record=False, true_label=None):
    rec = out["record"]
    present = [pipe.names[j] for j in np.where(x > 0.5)[0]]
    print("=" * 78)
    print(f"PATIENT {n}: {', '.join(present)}" + (f"   [true disease: {true_label}]" if true_label else ""))
    print("-" * 78)
    if rec["path"] == "fit":
        print(f"STAGE 1  GHSOM -> cluster {rec['cluster_code']} (trained label: {rec['cluster_disease']}, "
              f"purity {rec['cluster_purity']:.2f})")
        print(f"         error within cluster range? YES (conformal p = {rec['p_value']:.3f} > "
              f"alpha {pipe.alpha_fit})")
        print("STAGE 2  FIT EXPLANATION")
        print("         top deviating symptoms: " + (", ".join(
            f"{a['symptom']} ({a['direction']})" for a in rec["qe_attribution"]["top"][:3]) or "none"))
        print(f"         CART rule (for human readers, fidelity {rec['cart']['fidelity']:.2f}): "
              f"{rec['cart']['rule'][:150]}")
    else:
        print(f"STAGE 1  GHSOM nearest cluster {rec['ghsom_cluster']} -> error OUT of range "
              f"(p = {rec['ghsom_p_value']:.3f} <= {pipe.alpha_fit}) -> OUTLIER, sent to GNG")
        if rec["proposed_node"] is not None:
            print(f"         GNG: no node accepts X -> adds one new node (#{rec['proposed_node']}, cap 1 -> 0)")
        else:
            print("         GNG: accepted by node(s): " + ", ".join(
                f"#{c['node']} {c['disease']} (p={c['p']:.2f})" for c in rec["conformal_set"][:3]))
        print("STAGE 2  PROXIMITY EXPLANATION")
        for nb in rec["nearest"][:2]:
            d = nb["differential"]
            print(f"         vs node #{nb['node']} {nb['disease']}: shares {len(d['shared'])}, "
                  f"lacks {d['lacks'][:4]}, extra {d['excess'][:4]} (Hamming {d['hamming']})")
    print(f"STAGE 3  ONTOLOGY -> {out['case']}: {out['reason'] if 'reason' in out else ''}")
    if out.get("ontology_top"):
        print("         ontology ranking: " + ", ".join(
            f"{r['disease']} {r['score']:.2f} (needs {r['theta']:.2f})" for r in out["ontology_top"]))
    label = {"known": "KNOWN disease", "old_new": "OLD disease (added earlier)", "new": "NEW disease"}[out["outcome"]]
    print(f"\n>>> RESULT: {label}: {out['disease']}")
    if full_record:
        print(json.dumps(rec, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="kaggle", choices=list(config.DATASETS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patient", action="append", help="comma-separated symptom names (repeatable)")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--novel", help="with --demo: which held-out disease to use (see --list-diseases)")
    ap.add_argument("--list-symptoms", action="store_true")
    ap.add_argument("--list-diseases", action="store_true")
    ap.add_argument("--record", action="store_true", help="also print the full Stage 2 record")
    a = ap.parse_args()

    pipe, split = load_model(a.dataset, a.seed)
    if a.list_symptoms:
        print("\n".join(pipe.names))
        return
    if a.list_diseases:
        print("KNOWN (trained, in the ontology):\n  " + "\n  ".join(split.known))
        print("\nHELD OUT (the pipeline has never seen these - they should come out as NEW):\n  "
              + "\n  ".join(split.novel))
        return

    pipe.reset_stream()
    if a.demo:
        ds = load(a.dataset)
        rng = np.random.default_rng(a.seed)
        known_idx = [i for i in split.test if ds.y[i] in split.known]
        candidates = [a.novel] if a.novel else split.novel
        chosen, picks = None, None
        for nov in candidates:                       # find a held-out disease that shows Cases 2 and 3
            idx = [i for i in split.test if ds.y[i] == nov]
            if len(idx) < 2:
                continue
            trial = copy.deepcopy(pipe)
            trial.reset_stream()
            o1 = trial.diagnose(ds.X[idx[0]])
            if o1["outcome"] == "new" or a.novel:
                chosen, picks = nov, [idx[0]] + [i for i in idx[1:]][:2]
                break
        if chosen is None:
            nov = split.novel[0]
            chosen, picks = nov, [i for i in split.test if ds.y[i] == nov][:3]
        picks = [rng.choice(known_idx)] + picks
        print(f"DEMO: one known-disease patient, then patients of '{chosen}', a disease the pipeline never saw "
              f"(held out of training and of the ontology).")
        if not a.novel:
            print("(This held-out disease was picked because its first patient is flagged as new, to show Cases 2 "
                  "and 3. Not every new-disease patient is caught - see results/ for the rates. "
                  "Use --novel \"<name>\" to try another.)")
        print()
        for n, i in enumerate(picks, 1):
            out = pipe.diagnose(ds.X[i], keep_record=True)
            show(n, ds.X[i], out, pipe, a.record, true_label=ds.y[i])
        return
    if not a.patient:
        ap.print_help()
        return
    for n, p in enumerate(a.patient, 1):
        x = to_vector(pipe, p.split(","))
        out = pipe.diagnose(x, keep_record=True)
        show(n, x, out, pipe, a.record)


if __name__ == "__main__":
    main()
