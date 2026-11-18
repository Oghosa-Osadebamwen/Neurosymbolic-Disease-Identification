"""evaluate.py - open-set metrics for known-disease diagnosis and new-disease identification."""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.metrics import adjusted_rand_score, f1_score


def is_new_label(p) -> bool:
    return str(p).startswith("NEW-") or p == "UNKNOWN"


def open_set_metrics(y_true, y_pred, novel: list, known: list, min_support: int = 5) -> dict:
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    nov = np.isin(y_true, novel)
    kn = ~nov
    flagged = np.array([is_new_label(p) for p in y_pred])
    m = {}
    m["n_test"] = int(len(y_true))
    m["n_known_test"] = int(kn.sum())
    m["n_novel_test"] = int(nov.sum())
    m["known_accuracy"] = float((y_pred[kn] == y_true[kn]).mean())
    m["known_macro_f1"] = float(f1_score(y_true[kn], y_pred[kn], labels=known, average="macro", zero_division=0))
    m["known_false_new_rate"] = float(flagged[kn].mean())
    m["novel_detection_rate"] = float(flagged[nov].mean()) if nov.any() else np.nan
    m["novel_called_known_rate"] = 1 - m["novel_detection_rate"]
    m["open_set_accuracy"] = float(((y_pred == y_true) & kn | flagged & nov).mean())
    # ---- grouping of the novel patients that were flagged new
    sel = nov & flagged
    groups = sorted({p for p in y_pred[flagged]})
    m["new_groups_created"] = len(groups)
    m["true_novel_diseases"] = len(novel)
    if sel.sum() > 1:
        m["novel_group_ARI"] = float(adjusted_rand_score(y_true[sel], y_pred[sel]))
        pur = 0
        majority_of = set()
        for g in set(y_pred[sel]):
            c = Counter(y_true[sel][y_pred[sel] == g]).most_common(1)[0]
            pur += c[1]
            majority_of.add(c[0])
        m["novel_group_purity"] = float(pur / sel.sum())
        m["novel_diseases_recovered"] = len(majority_of)
    else:
        m["novel_group_ARI"] = m["novel_group_purity"] = np.nan
        m["novel_diseases_recovered"] = 0
    # share of new groups that consist mostly of known-disease patients (false discoveries)
    false_groups = 0
    for g in groups:
        members = y_true[y_pred == g]
        if np.isin(members, known).mean() > 0.5:
            false_groups += 1
    m["false_new_groups"] = false_groups
    # "supported" new diseases: groups that collected >= min_support patients
    sizes = Counter(p for p in y_pred if str(p).startswith("NEW-"))
    sup = [g for g, n in sizes.items() if n >= min_support]
    m["supported_new_groups"] = len(sup)
    m["supported_false_groups"] = sum(np.isin(y_true[y_pred == g], known).mean() > 0.5 for g in sup)
    m["supported_novel_recovered"] = len({Counter(y_true[y_pred == g]).most_common(1)[0][0] for g in sup}
                                         & set(novel))
    return m
