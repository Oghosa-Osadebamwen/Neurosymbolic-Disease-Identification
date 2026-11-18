"""
train.py - train the GHSOM + GNG neuro-symbolic pipeline
========================================================

For each dataset and seed:
  1. hold out `n_novel` diseases as unseen "new" diseases (removed from training AND ontology)
  2. split the known diseases into train / calibration / test
  3. Stage 1: pretrain GHSOM on labelled training symptoms; initialise GNG from GHSOM's leaves
     and pretrain it (no growth)
  4. Stage 2: calibrate conformal tests and fit the shallow CART surrogate on held-out data
  5. Stage 3: build the ontology for the known diseases and calibrate its thresholds
  6. save the trained model to models/<dataset>_seed<k>.pkl

Usage:  python train.py [--datasets kaggle ddxplus] [--seeds 0 1 2 3 4]
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import config
from neurosym.data import load_ddxplus, load_kaggle, make_split
from neurosym.pipeline import NeuroSymbolicPipeline


def load(name):
    cfg = config.DATASETS[name]
    if name == "kaggle":
        return load_kaggle(config.DATA_DIR, **cfg["loader"])
    return load_ddxplus(config.DATA_DIR, **cfg["loader"])


def train_one(ds, name, seed):
    cfg = config.DATASETS[name]
    split = make_split(ds, cfg["n_novel"], seed, max_novel_per_disease=cfg["max_novel_per_disease"])
    t0 = time.time()
    pipe = NeuroSymbolicPipeline(seed=seed, **cfg["pipeline"])
    pipe.fit(ds.X[split.train], ds.y[split.train], ds.X[split.calib], ds.y[split.calib],
             ds.rules, ds.features, split.known, demographic=[ds.features[i] for i in ds.demographic_idx])
    info = {"dataset": name, "seed": seed, "train_seconds": round(time.time() - t0, 1),
            "n_train": len(split.train), "n_calib": len(split.calib), "n_test": len(split.test),
            "n_features": ds.X.shape[1], "known_diseases": len(split.known), "novel_diseases": split.novel,
            **pipe.summary()}
    return pipe, split, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(config.DATASETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=config.SEEDS)
    args = ap.parse_args()
    Path(config.MODEL_DIR).mkdir(exist_ok=True)
    Path(config.RESULTS_DIR).mkdir(exist_ok=True)
    log = []
    for name in args.datasets:
        ds = load(name)
        print(f"\n=== {name}: {ds.X.shape[0]} patients, {ds.X.shape[1]} features, {len(ds.diseases)} diseases")
        for seed in args.seeds:
            pipe, split, info = train_one(ds, name, seed)
            with open(Path(config.MODEL_DIR) / f"{name}_seed{seed}.pkl", "wb") as fh:
                pickle.dump({"pipeline": pipe, "split": split, "info": info}, fh)
            log.append(info)
            print(f"  seed {seed}: GHSOM {info['ghsom_maps']} maps, depth {info['ghsom_max_depth']}, "
                  f"{info['ghsom_leaf_clusters']} leaf clusters (purity {info['ghsom_weighted_purity']:.3f}); "
                  f"GNG {info['gng_init_nodes']} nodes; CART fidelity {info['cart_fidelity']:.3f}; "
                  f"held out {split.novel}; {info['train_seconds']}s")
    with open(Path(config.RESULTS_DIR) / "training_log.json", "w") as fh:
        json.dump(log, fh, indent=1, default=str)
    write_manifest()


def write_manifest():
    """Run record: library versions, platform, settings and SHA-256 fingerprints of data and rule files."""
    import hashlib
    import platform
    import sys
    import numpy, pandas, sklearn
    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    d = Path(config.DATA_DIR)
    files = [d / "Training.csv", d / "ddxplus" / "release_test_patients.csv.gz", d / "ddxplus" / "release_evidences.json",
             d / "rules" / "independent_rules.csv", d / "rules" / "ddxplus_rules.csv"]
    manifest = {"python": sys.version.split()[0], "numpy": numpy.__version__, "pandas": pandas.__version__,
                "scikit-learn": sklearn.__version__, "platform": platform.platform(),
                "seeds": config.SEEDS, "config": config.DATASETS,
                "sha256": {str(f): sha(f) for f in files if f.exists()}}
    with open(Path(config.RESULTS_DIR) / "run_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
