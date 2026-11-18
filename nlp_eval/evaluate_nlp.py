"""evaluate_nlp.py - check the plain-English symptom extractor on 30 complaints written independently of its
phrase list (nlp_eval/complaints.csv), and on complaints written for every Kaggle patient.
Writes results/nlp_eval.csv and results/nlp_eval_summary.json."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from neurosym.data import load_kaggle
from neurosym.symptom_extractor import SymptomExtractor

ds = load_kaggle(config.DATA_DIR)
ex = SymptomExtractor(list(ds.features))
df = pd.read_csv(Path(__file__).parent / "complaints.csv", dtype=str).fillna("")
rows, tp, fp, fn, ntp, nfp, nfn = [], 0, 0, 0, 0, 0, 0
for r in df.itertuples():
    gold_p = set(filter(None, r.expected_present.split(";"))); gold_n = set(filter(None, r.expected_denied.split(";")))
    res = ex.extract(r.complaint)
    got_p, got_n = set(res.present), set(res.absent)
    tp += len(got_p & gold_p); fp += len(got_p - gold_p); fn += len(gold_p - got_p)
    ntp += len(got_n & gold_n); nfp += len(got_n - gold_n); nfn += len(gold_n - got_n)
    rows.append({"id": r.id, "complaint": r.complaint, "missed": ";".join(sorted(gold_p - got_p)),
                 "wrongly_added": ";".join(sorted(got_p - gold_p)), "denials_missed": ";".join(sorted(gold_n - got_n)),
                 "exact": got_p == gold_p and got_n == gold_n})
out = pd.DataFrame(rows)
out.to_csv(Path(config.RESULTS_DIR) / "nlp_eval.csv", index=False)
# round trip: a complaint written for every patient must be read back exactly
rt = sum(set(ex.extract(ex.complaint_from([ds.features[j] for j in np.where(ds.X[i] > 0.5)[0]], seed=i)).present)
         == {ds.features[j] for j in np.where(ds.X[i] > 0.5)[0]} for i in range(len(ds.y)))
summ = {"complaints": len(df), "symptom_mentions": tp + fn, "precision": tp / (tp + fp), "recall": tp / (tp + fn),
        "f1": 2 * tp / (2 * tp + fp + fn), "denials": ntp + nfn, "denials_found": ntp, "denials_false": nfp,
        "complaints_exactly_right": int(out.exact.sum()), "round_trip_patients": len(ds.y), "round_trip_exact": int(rt)}
json.dump(summ, open(Path(config.RESULTS_DIR) / "nlp_eval_summary.json", "w"), indent=1)
print(json.dumps(summ, indent=1)); print(out[~out.exact][["id", "missed", "wrongly_added", "denials_missed"]].to_string())
