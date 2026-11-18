"""compare_fix.py - the same patients through the earlier and the current pipeline.

    python compare_fix.py                      (Kaggle model of run 0)
    python compare_fix.py --patient "..." --patient "..."   (your own complaints, in order)

The only difference between the two: in the earlier version, a patient who lands in a mixed (low-purity) GHSOM
cluster that the rules cannot place is always registered as a NEW disease. In the current version, that patient is
first checked against the new diseases registered earlier in the session (Case 3).
"""
import argparse
import copy
import pickle

import numpy as np

import config
from neurosym.symptom_extractor import SymptomExtractor

EXAMPLES = {
    "Paralysis (brain haemorrhage), a disease the model was never trained on": [
        "He suddenly had a terrible headache and started vomiting. Now one side of his body is weak and he is confused.",
        "My father has a bad headache, he's been throwing up, he's disoriented and he can't move one side of his body.",
        "She has a severe headache and keeps vomiting. She is confused and one side of her body is weak.",
    ],
    "Hepatitis C, a disease the model was never trained on": [
        "My eyes have turned yellow and my skin is yellow too. I feel like vomiting, I've lost my appetite and "
        "I'm always tired. It runs in my family.",
        "My skin has turned yellow and my eyes are yellow. I'm nauseous, I have no appetite and I'm tired. "
        "It runs in the family.",
    ],
}


def run(base, complaints, recheck):
    p = copy.deepcopy(base)
    p.reset_stream()
    p.recheck_registered_on_mixed = recheck
    ex = SymptomExtractor(p.names)
    out = []
    for c in complaints:
        x = np.zeros(len(p.names))
        for s in ex.extract(c).present:
            x[p.names.index(s)] = 1
        o = p.diagnose(x, keep_record=True)
        path = "mixed cluster" if o["record"]["path"] == "fit" else "outlier"
        out.append(f"{o['case'].split(':')[0]:<34} -> {o['disease']:<22} ({path})")
    return out, len(p.ontology.registry)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", action="append", help="a complaint in plain English (repeat, in order)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    base = pickle.load(open(f"{config.MODEL_DIR}/kaggle_seed{a.seed}.pkl", "rb"))["pipeline"]
    groups = {"Your patients": a.patient} if a.patient else EXAMPLES
    for title, complaints in groups.items():
        print("=" * 100 + f"\n{title}\n" + "=" * 100)
        old, n_old = run(base, complaints, recheck=False)
        new, n_new = run(base, complaints, recheck=True)
        for i, c in enumerate(complaints):
            print(f"\nPatient {i + 1}: \"{c}\"")
            print(f"   earlier pipeline : {old[i]}")
            print(f"   current pipeline : {new[i]}")
        print(f"\nNew diseases registered: earlier pipeline {n_old}, current pipeline {n_new}\n")


if __name__ == "__main__":
    main()
