"""make_figures.py - figures from results/*.csv into results/figures/."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path("results")
F = R / "figures"
F.mkdir(parents=True, exist_ok=True)
SURF, INK, INK2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
METHODS = ["Symbolic-alone", "Neural-alone (GHSOM+GNG)", "Neural-alone, strict GNG", "Neuro-symbolic (full)",
           "Supervised: random forest + confidence threshold"]
LABELS = ["Symbolic-alone", "Neural-alone (lenient GNG)", "Neural-alone (strict GNG)", "Neuro-symbolic (full)",
          "Random forest + confidence threshold (labels)"]
NAMES = {"kaggle": "Kaggle (41 diseases)", "ddxplus": "DDXPlus (49 diseases)"}
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


df = pd.read_csv(R / "per_run.csv")
metrics = [("known_accuracy", "Known-disease\naccuracy"), ("novel_detection_rate", "New-disease\ndetection"),
           ("known_false_new_rate", "Known called new\n(lower = better)"), ("open_set_accuracy", "Open-set\naccuracy")]

# ---------------- main comparison, one figure per dataset
for ds in ["kaggle", "ddxplus"]:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    sub = df[df.dataset == ds]
    x = np.arange(len(metrics))
    w = 0.16
    for k, m in enumerate(METHODS):
        s = sub[sub.method == m]
        mu = np.array([s[c].mean() for c, _ in metrics])
        sd = np.array([s[c].std() for c, _ in metrics])
        pos = x + (k - 2) * (w + 0.008)
        ax.bar(pos, mu, w, color=SERIES[k], label=LABELS[k], yerr=sd,
               error_kw={"ecolor": INK2, "lw": 0.9, "capsize": 2}, edgecolor=SURF, linewidth=1)
        if m == "Neuro-symbolic (full)":
            for p, v, e in zip(pos, mu, sd):
                ax.text(p, v + e + 0.02, f"{v:.2f}", ha="center", fontsize=8, color=INK, fontweight="bold")
    ax.set_xticks(x, [l for _, l in metrics])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Rate (mean ± SD over 10 seeds)")
    style(ax)
    ax.legend(frameon=False, fontsize=8.3, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
    fig.tight_layout()
    fig.savefig(F / f"fig_main_{ds}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

# ---------------- routing of patients through the cases (neuro-symbolic)
pp = pd.read_csv(R / "per_patient.csv")
order = ["Case 1", "Ontology reasoning (low purity)", "GNG -> known", "Ontology override -> known", "Case 2", "Case 3"]
lab = {"Case 1": "Case 1 (GHSOM lookup)", "Ontology reasoning (low purity)": "Low-purity cluster: ontology reasoning",
       "GNG -> known": "Outlier: GNG + ontology -> known", "Ontology override -> known": "Outlier: ontology -> known",
       "Case 2": "Case 2 (new disease)", "Case 3": "Case 3 (earlier new disease)"}
cols = ["#2a78d6", "#86b6ef", "#1baf7a", "#eda100", "#e34948", "#6250d6"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for ax, ds in zip(axes, ["kaggle", "ddxplus"]):
    sub = pp[pp.dataset == ds]
    groups = [("Known-disease\npatients", sub[~sub.novel]), ("Held-out (new)\ndisease patients", sub[sub.novel])]
    for gi, (gname, g) in enumerate(groups):
        share = g.case.value_counts(normalize=True)
        left = 0
        for ci, c in enumerate(order):
            v = share.get(c, 0)
            ax.barh(gi, v, left=left, color=cols[ci], edgecolor=SURF, linewidth=1.5,
                    label=lab[c] if (ds == "kaggle" and gi == 0) else None)
            if v >= 0.06:
                ax.text(left + v / 2, gi, f"{v:.0%}", ha="center", va="center", fontsize=8, color="white")
            left += v
    ax.set_yticks([0, 1], [g[0] for g in groups])
    ax.set_xlim(0, 1)
    ax.set_title(NAMES[ds], loc="left", fontsize=10.5, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Share of test patients")
    ax.invert_yaxis()
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12))
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig(F / "fig_routes.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------- parameter sweeps (neuro-symbolic)
for fname, col, xlabel in [("sweep_alpha.csv", "alpha_fit", "GHSOM conformal rejection level α"),
                           ("sweep_q.csv", "ontology_q", "Ontology threshold quantile q")]:
    sw = pd.read_csv(R / fname)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    lines = [("known_accuracy", "Known-disease accuracy"), ("novel_detection_rate", "New-disease detection"),
             ("known_false_new_rate", "Known called new"), ("open_set_accuracy", "Open-set accuracy")]
    for ax, ds in zip(axes, ["kaggle", "ddxplus"]):
        g = sw[sw.dataset == ds].groupby(col)
        for k, (c, lb) in enumerate(lines):
            mu, sd = g[c].mean(), g[c].std()
            ax.plot(mu.index, mu.values, color=SERIES[k], lw=2, marker="o", ms=4.5, label=lb)
            ax.fill_between(mu.index, mu - sd, mu + sd, color=SERIES[k], alpha=0.12, lw=0)
        ax.set_title(NAMES[ds], loc="left", fontsize=10.5)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 1.05)
        style(ax)
    axes[0].set_ylabel("Rate (mean ± SD, 10 seeds)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8.5, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(F / f"fig_{col}_sweep.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
print("figures written to", F)
