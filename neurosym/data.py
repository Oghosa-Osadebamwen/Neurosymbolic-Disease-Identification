"""
data.py - loading and encoding the two disease datasets
=======================================================

Both datasets are turned into the same shape:

    X        : (n_patients, n_features) binary matrix (1 = symptom/evidence present)
    y        : (n_patients,) disease name per patient
    features : list of feature names (column order of X)
    rules    : the symbolic ontology for this dataset, read from the rule CSV
               (disease -> list of rule items; each item is "any of these features")

Datasets
--------
1. Kaggle disease-symptom dataset (Training.csv): 4,920 rows, 41 diseases, 132 binary symptoms.
   It contains only ~304 distinct (disease, symptom-set) patterns; the rest are exact duplicates.
   We therefore DEDUPLICATE before splitting, so that no symptom pattern appears in both the
   training and the test set (otherwise test accuracy just measures memorisation).
   Ontology: rules/independent_rules.csv (sourced from NHS / CDC / Mayo / MedlinePlus pages).

2. DDXPlus (release_test_patients.csv.gz): 134,529 synthetic patients, 49 pathologies.
   Evidences are binary (E_91), categorical or multi-choice (E_55_@_V_29 -> feature "E_55=V_29").
   Sex and age are encoded as sex_male / sex_female and six age bins, exactly as in the rule file.
   Ontology: rules/ddxplus_rules.csv (built from public health pages, NOT from
   release_conditions.json).

Only the data and rule files are reused; no code from the earlier implementation.
"""
from __future__ import annotations

import ast
import re
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Kaggle label spellings -> spellings used in independent_rules.csv
KAGGLE_NAME_FIX = {
    "Peptic ulcer diseae": "Peptic ulcer disease",
    "hepatitis A": "Hepatitis A",
    "Osteoarthristis": "Osteoarthritis",
    "Dimorphic hemmorhoids(piles)": "Hemorrhoids (piles)",
    "(vertigo) Paroymsal  Positional Vertigo": "Paroxysmal positional vertigo",
}

DEMOGRAPHIC = ("sex_male", "sex_female", "age_under_2", "age_2_11", "age_12_17",
               "age_18_39", "age_40_64", "age_65_plus")


@dataclass
class RuleItem:
    """One ontology criterion: TRUE if ANY of `features` is present in the patient."""
    features: frozenset          # feature indices
    kind: str                    # "symptom" or "risk"
    phrase: str                  # the source wording (for explanations)


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    features: list
    rules: dict                  # disease -> list[RuleItem]
    demographic_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    weights: np.ndarray | None = None   # duplicate counts (Kaggle) - informational only

    @property
    def diseases(self):
        return sorted(set(self.y))


def _read_rules(path: Path, feat_index: dict) -> tuple[dict, list]:
    df = pd.read_csv(path)
    rules: dict[str, list[RuleItem]] = {}
    unknown = []
    for _, r in df.iterrows():
        idx = []
        for f in str(r["dataset_symptoms_any_of"]).split("|"):
            f = f.strip()
            if f in feat_index:
                idx.append(feat_index[f])
            else:
                unknown.append(f)
        if idx:
            rules.setdefault(r["disease"].strip(), []).append(
                RuleItem(frozenset(idx), r["item_type"], str(r["source_phrase"])))
    return rules, unknown


# --------------------------------------------------------------------------- Kaggle
def load_kaggle(data_dir: str | Path) -> Dataset:
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "Training.csv")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    y = df.pop("prognosis").str.strip().replace(KAGGLE_NAME_FIX).to_numpy()
    # step 3: "fluid_overload" appears twice (pandas reads the 2nd as "fluid_overload.1"): merge with OR
    dup_cols = [c for c in df.columns if c.endswith(".1") and c[:-2] in df.columns]
    for c in dup_cols:
        df[c[:-2]] = df[[c[:-2], c]].max(axis=1)
        df = df.drop(columns=c)
    # fix stray spaces: "spotting_ urination" -> "spotting_urination", "foul_smell_of urine" -> "foul_smell_of_urine"
    features = [re.sub(r"_+", "_", c.strip().replace(" ", "_")) for c in df.columns]
    X = df.to_numpy(dtype=np.float32)

    # ---- deduplicate (disease, symptom-set) patterns, keep the count as a weight
    key = pd.DataFrame(X).astype(int).astype(str).agg("".join, axis=1) + "|" + y
    uniq = pd.Series(range(len(key))).groupby(key.values).agg(["first", "size"])
    first, counts = uniq["first"].to_numpy(), uniq["size"].to_numpy()
    X, y = X[first], y[first]

    feat_index = {f: i for i, f in enumerate(features)}
    rules, unknown = _read_rules(data_dir / "rules" / "independent_rules.csv", feat_index)
    missing = set(y) - set(rules)
    if missing:
        raise ValueError(f"Ontology has no rules for: {missing}")
    if unknown:
        raise ValueError(f"Rule file names features not in Training.csv: {sorted(set(unknown))[:10]}")
    return Dataset("kaggle", X, y, features, rules, weights=counts)


# --------------------------------------------------------------------------- DDXPlus
def ddxplus_feature_names(data_dir: Path) -> list[str]:
    """Same feature space the rule file was built against."""
    return pd.read_csv(data_dir / "rules" / "ddxplus_feature_descriptions.csv")["feature"].tolist()


def _age_bin(a: int) -> str:
    if a < 2:
        return "age_under_2"
    if a < 12:
        return "age_2_11"
    if a < 18:
        return "age_12_17"
    if a < 40:
        return "age_18_39"
    if a < 65:
        return "age_40_64"
    return "age_65_plus"


def _encode_ddxplus(df, features, evid):
    """Section 3.2.3 conversion rules: yes/no -> one column; location-type evidence and pain character ->
    one column per value (default not a column); other categorical -> one column, 1 if non-default;
    0-10 scales E_56/E_58/E_59 dropped (not in the feature list); sex -> 2 columns; age -> 6 bands."""
    fi = {f: i for i, f in enumerate(features)}
    X = np.zeros((len(df), len(features)), dtype=np.float32)
    for r, (ev, sex, age) in enumerate(zip(df["EVIDENCES"], df["SEX"], df["AGE"])):
        for tok in ast.literal_eval(ev):
            if "_@_" in tok:
                code, val = tok.split("_@_")
                f = f"{code}={val}"
                if f in fi:
                    X[r, fi[f]] = 1
                if code in fi and str(val) != str(evid[code]["default_value"]):
                    X[r, fi[code]] = 1
            elif tok in fi:
                X[r, fi[tok]] = 1
        X[r, fi["sex_male" if sex == "M" else "sex_female"]] = 1
        X[r, fi[_age_bin(int(age))]] = 1
    return X


def _ddxplus_raw(data_dir: Path):
    df = pd.read_csv(data_dir / "ddxplus" / "release_test_patients.csv.gz")
    n_raw = len(df)
    # step 1: exact duplicate patients (same evidence, sex, age and disease)
    df = df.drop_duplicates(subset=["EVIDENCES", "SEX", "AGE", "PATHOLOGY"]).reset_index(drop=True)
    return df, n_raw


def load_ddxplus(data_dir: str | Path, per_disease_cap: int = 40, seed: int = 0) -> Dataset:
    """DDXPlus exactly as in the dissertation, Table 3.2: remove exact duplicates, sample up to 40 patients
    per disease (seed 0), convert to 0/1 columns, drop columns never present in the sample, and drop rows
    that became identical to another row of the same disease. Result: 1,943 patients, 49 diseases."""
    data_dir = Path(data_dir)
    evid = json.loads((data_dir / "ddxplus" / "release_evidences.json").read_text(encoding="utf-8"))
    features = ddxplus_feature_names(data_dir)
    df, n_raw = _ddxplus_raw(data_dir)
    n_dedup = len(df)
    # step 2: class-balanced sample, up to `per_disease_cap` per disease, fixed seed
    sample = pd.concat([g.sample(min(len(g), per_disease_cap), random_state=seed)
                        for _, g in df.groupby("PATHOLOGY")])
    X = _encode_ddxplus(sample, features, evid)                       # step 3
    y = sample["PATHOLOGY"].to_numpy()
    fi = {f: i for i, f in enumerate(features)}
    rules, unknown = _read_rules(data_dir / "rules" / "ddxplus_rules.csv", fi)
    if set(y) - set(rules):
        raise ValueError(f"Ontology has no rules for: {set(y) - set(rules)}")
    if unknown:
        raise ValueError(f"Rule file names unknown features: {sorted(set(unknown))[:10]}")
    # step 4a: drop columns that are never 1 in the sample
    keep = np.where(X.sum(0) > 0)[0]
    remap = {old: new for new, old in enumerate(keep)}
    X = X[:, keep]
    features = [features[i] for i in keep]
    # items whose columns never occur stay in the rule base (they count, and can never fire)
    rules = {d: [RuleItem(frozenset(remap[i] for i in it.features if i in remap), it.kind, it.phrase)
                 for it in items] for d, items in rules.items()}
    # step 4b: drop rows that became identical to another row of the same disease
    key = pd.Series([r.tobytes() for r in X.astype(np.int8)]).astype(str) + "|" + y
    dup = key.duplicated().to_numpy()
    X, y = X[~dup], y[~dup]
    demo = np.array([i for i, f in enumerate(features) if f in DEMOGRAPHIC])
    ds = Dataset("ddxplus", X, y, features, rules, demographic_idx=demo)
    ds.cleaning = {"rows_raw": n_raw, "exact_duplicates_removed": n_raw - n_dedup,
                   "sampled": len(sample), "columns_kept": len(features),
                   "identical_rows_removed": int(dup.sum()), "rows_clean": len(y),
                   "never_firing_rule_items": sum(1 for it in sum(rules.values(), []) if not it.features)}
    ds._sample_ids = set(sample.index)
    return ds


def ddxplus_supervised_pool(data_dir: str | Path, ds: Dataset, per_disease_cap: int = 400, seed: int = 0):
    """Separate training pool for the supervised reference (Section 3.2.3): patients of the test file that
    are not in the sample, up to 400 per disease, encoded on the sample's columns."""
    data_dir = Path(data_dir)
    evid = json.loads((data_dir / "ddxplus" / "release_evidences.json").read_text(encoding="utf-8"))
    df, _ = _ddxplus_raw(data_dir)
    rest = df.drop(index=list(ds._sample_ids))
    pool = pd.concat([g.sample(min(len(g), per_disease_cap), random_state=seed)
                      for _, g in rest.groupby("PATHOLOGY")])
    allf = ddxplus_feature_names(data_dir)
    Xp = _encode_ddxplus(pool, allf, evid)
    col = [allf.index(f) for f in ds.features]
    return Xp[:, col], pool["PATHOLOGY"].to_numpy()


# --------------------------------------------------------------------------- splitting
@dataclass
class Split:
    known: list            # diseases the model is trained on (and that the ontology knows)
    novel: list            # held-out diseases: never seen in training, removed from the ontology
    train: np.ndarray      # indices
    calib: np.ndarray      # calibration set (conformal tests, CART fidelity), known diseases only
    test: np.ndarray       # known-disease test patients + ALL novel-disease patients


def make_split(ds: Dataset, n_novel: int, seed: int, frac_calib=0.2, frac_test=0.3,
               max_novel_per_disease: int | None = None) -> Split:
    rng = np.random.default_rng(seed)
    diseases = np.array(ds.diseases)
    novel = sorted(rng.choice(diseases, n_novel, replace=False).tolist())
    known = sorted(set(diseases) - set(novel))
    tr, ca, te = [], [], []
    for d in known:
        idx = rng.permutation(np.where(ds.y == d)[0])
        n = len(idx)
        n_te = max(1, int(round(frac_test * n)))
        n_ca = max(1, int(round(frac_calib * n)))
        if n - n_te - n_ca < 2:            # keep at least 2 training patterns per disease
            n_ca = max(0, n - n_te - 2)
        te += idx[:n_te].tolist()
        ca += idx[n_te:n_te + n_ca].tolist()
        tr += idx[n_te + n_ca:].tolist()
    for d in novel:
        idx = rng.permutation(np.where(ds.y == d)[0])
        if max_novel_per_disease:
            idx = idx[:max_novel_per_disease]
        te += idx.tolist()
    return Split(known, novel, np.array(tr), np.array(ca), rng.permutation(np.array(te)))
