"""Experiment configuration (one block per dataset). Hyper-parameters were chosen on the
training / calibration split only - never on the test set."""

DATA_DIR = "data"
MODEL_DIR = "models"
RESULTS_DIR = "results"
SEEDS = list(range(10))          # ten seeds, as in the dissertation

DATASETS = {
    "kaggle": {
        "loader": {},
        "n_novel": 4,                    # diseases held out as "new" (not trained, not in ontology)
        "max_novel_per_disease": None,
        "pipeline": {
            "ghsom_params": {"tau1": 0.6, "tau2": 0.05, "purity_target": 0.95,
                             "min_child_samples": 6, "max_units": 64, "max_depth": 3},
            "alpha_fit": 0.05,           # GHSOM conformal rejection level
            "alpha_gng": 0.01,           # GNG acceptance level (more lenient = "aggressive" mapping)
            "purity_min": 0.9,           # Case 1 lookup only for clusters at least this pure
            "ontology_q": 0.05,          # ontology consistency threshold quantile
            "cart_depth": 8,
            "min_mondrian": 30,
        },
    },
    "ddxplus": {
        "loader": {"per_disease_cap": 40},
        "n_novel": 5,
        "max_novel_per_disease": None,
        "pipeline": {
            "ghsom_params": {"tau1": 0.85, "tau2": 1.0, "purity_target": 0.9,
                             "min_child_samples": 20, "max_units": 64, "max_depth": 3},
            "alpha_fit": 0.05,
            "alpha_gng": 0.01,
            "purity_min": 0.9,
            "ontology_q": 0.05,
            "cart_depth": 8,
            "min_mondrian": 30,
        },
    },
}
