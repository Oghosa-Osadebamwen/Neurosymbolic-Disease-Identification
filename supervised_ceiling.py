"""
supervised_ceiling.py - label-trained ceiling on all diseases (closed set), as in the dissertation
==================================================================================================
Kaggle: 5-fold stratified cross-validation on the 304 distinct patients.
DDXPlus: trained on the separate pool of 18,757 patients (Section 3.2.3), scored on the 1,943-patient sample.
Writes results/supervised_ceiling.csv.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, top_k_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import config
from neurosym.data import ddxplus_supervised_pool, load_ddxplus, load_kaggle


def scores(y, pred, proba, classes):
    y, pred, classes = (np.asarray(a, dtype=str) for a in (y, pred, classes))
    # a disease absent from the training data (DDXPlus: Bronchiolitis has no patients left for the pool)
    # gets a zero-probability column, so it is scored as always missed rather than dropped
    missing = sorted(set(y) - set(classes))
    if missing:
        proba = np.hstack([proba, np.zeros((len(proba), len(missing)))])
        classes = np.concatenate([classes, missing])
        order = np.argsort(classes)
        proba, classes = proba[:, order], classes[order]
    return {"accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, average="macro", zero_division=0),
            "recall": recall_score(y, pred, average="macro", zero_division=0),
            "f1": f1_score(y, pred, average="macro", zero_division=0),
            "top3": top_k_accuracy_score(y, proba, k=3, labels=classes)}


rows = []
models = {"Logistic regression": lambda: LogisticRegression(max_iter=3000),
          "Random forest (300 trees)": lambda: RandomForestClassifier(n_estimators=300, random_state=0)}
k = load_kaggle(config.DATA_DIR)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
for name, mk in models.items():
    proba = cross_val_predict(mk(), k.X, k.y, cv=cv, method="predict_proba")
    classes = np.unique(k.y)
    pred = classes[proba.argmax(1)]
    rows.append({"dataset": "kaggle", "model": name, "scored_by": "5-fold cross-validation",
                 **scores(k.y, pred, proba, classes)})
d = load_ddxplus(config.DATA_DIR)
Xp, yp = ddxplus_supervised_pool(config.DATA_DIR, d)
for name, mk in models.items():
    m = mk().fit(Xp, yp)
    rows.append({"dataset": "ddxplus", "model": name, "scored_by": "trained on 18,757-patient pool, tested on sample",
                 **scores(d.y, m.predict(d.X), m.predict_proba(d.X), m.classes_)})
out = pd.DataFrame(rows)
out.to_csv(f"{config.RESULTS_DIR}/supervised_ceiling.csv", index=False)
print(out.round(3).to_string())
