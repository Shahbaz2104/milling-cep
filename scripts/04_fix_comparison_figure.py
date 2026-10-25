"""Regenerate the pooled model comparison table/figure from saved metrics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FIGURES, MODELS
from src.modeling import MODEL_COLORS, MODEL_LABELS

payload = json.loads((MODELS / "model_metrics.json").read_text(encoding="utf-8"))
schemes = ["tool_5fold", "tool_loo", "experiment_loo", "random_5fold"]
rows = []
for model, model_data in payload["models"].items():
    for scheme in schemes:
        m = model_data[scheme]["pooled"]
        rows.append(
            {
                "model": model,
                "scheme": scheme,
                "roc_auc": m["roc_auc"],
                "pr_auc": m["pr_auc"],
                "balanced_accuracy": m["balanced_accuracy"],
                "recall": m["recall"],
                "precision": m["precision"],
                "f1": m["f1"],
                "specificity": m["specificity"],
                "mcc": m["mcc"],
                "brier": m["brier"],
                "accuracy": m["accuracy"],
            }
        )
comparison = pd.DataFrame(rows)
comparison.to_csv(MODELS / "model_comparison_pooled.csv", index=False)

primary = comparison[comparison["scheme"] == "tool_5fold"].set_index("model")
order = ["Dummy", "LogisticRegression", "RandomForest", "HistGradientBoosting"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
x = np.arange(len(order))
for ax, metric, title in [
    (axes[0], "roc_auc", "ROC-AUC (pooled out-of-fold)"),
    (axes[1], "pr_auc", "PR-AUC (pooled out-of-fold)"),
]:
    vals = primary.loc[order, metric].values
    bars = ax.bar(x, vals, color=[MODEL_COLORS[m] for m in order], width=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in order], rotation=15, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel(metric.replace("_", "-").upper())
    ax.set_title(title)
    baseline = 0.5 if metric == "roc_auc" else float(payload["n_dangerous"] / payload["n_cycles"])
    ax.axhline(baseline, color="#777777", linestyle="--", linewidth=1)
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", fontsize=8)
fig.suptitle("Model comparison under tool-grouped 5-fold validation", fontsize=11, fontweight="bold", y=1.02)
fig.text(
    0.5,
    -0.04,
    "Dummy uses the training-fold prior; its pooled ROC-AUC is anti-correlated because tool folds differ strongly in class balance.",
    ha="center",
    fontsize=7,
    color="#555555",
)
fig.tight_layout()
fig.savefig(FIGURES / "fig07_model_comparison.png", bbox_inches="tight", facecolor="white")
plt.close(fig)
print(comparison[comparison["scheme"] == "tool_5fold"].to_string(index=False))
print("Wrote", MODELS / "model_comparison_pooled.csv")
print("Wrote", FIGURES / "fig07_model_comparison.png")
