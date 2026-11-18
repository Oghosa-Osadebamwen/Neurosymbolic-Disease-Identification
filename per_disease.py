"""per_disease.py - per-disease results for Appendices D and E (from results/per_patient.csv)."""
import pandas as pd
import config

pp = pd.read_csv(f"{config.RESULTS_DIR}/per_patient.csv")
rows = []
for (ds, dis), g in pp.groupby(["dataset", "true"]):
    k, n = g[~g.novel], g[g.novel]
    rows.append({"dataset": ds, "disease": dis,
                 "seeds_known": k.seed.nunique(), "known_test_patients": len(k),
                 "known_recall": (k.pred == k.true).mean() if len(k) else float("nan"),
                 "seeds_held_out": n.seed.nunique(), "held_out_patients": len(n),
                 "held_out_flagged_new": n.outcome.isin(["new", "old_new"]).mean() if len(n) else float("nan")})
out = pd.DataFrame(rows)
out.to_csv(f"{config.RESULTS_DIR}/per_disease.csv", index=False)
for ds in ["kaggle", "ddxplus"]:
    o = out[out.dataset == ds]
    print(ds, "diseases", len(o), "never held out:", int((o.seeds_held_out == 0).sum()),
          "| mean known recall", round(o.known_recall.mean(), 3),
          "| recall 1.0:", int((o.known_recall == 1).sum()), "| recall < 0.8:", list(o[o.known_recall < 0.8].disease))
