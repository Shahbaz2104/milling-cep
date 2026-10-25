"""Select and persist the final random-forest model with an operating threshold."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_PROCESSED, FIGURES, MODELS, SENSOR_COLUMNS
from src.modeling import (
    CLASS_ORDER,
    MODEL_LABELS,
    PALETTE,
    base_pipeline,
    choose_threshold,
    metric_dict,
    run_cv,
    save_importance,
)

PARAMS = {"clf__min_samples_leaf": 3, "clf__max_depth": 6}


def main() -> None:
    clean = pd.read_csv(DATA_PROCESSED / "milling_clean.csv")
    X = clean[SENSOR_COLUMNS].copy()
    y = clean["RDOC_class"].astype(int).to_numpy()
    tool_groups = clean["ToolIndex"].astype(int).to_numpy()
    tool5_groups = clean["tool_5fold"].astype(int).to_numpy()

    print("Re-running nested tool-group CV for random forest...")
    primary = run_cv(
        "RandomForest",
        X,
        y,
        tool5_groups,
        LeaveOneGroupOut(),
        scheme="tool_5fold",
        tune=True,
        verbose=True,
    )
    threshold_info = choose_threshold(y, primary.oof_prob)
    threshold = float(threshold_info["max_f1_threshold"])
    selected = metric_dict(y, primary.oof_prob, threshold=threshold)
    print(f"Threshold={threshold:.3f}; selected metrics={selected}")

    print("Running leave-one-tool-out sensitivity for random forest...")
    loo = run_cv(
        "RandomForest",
        X,
        y,
        tool_groups,
        LeaveOneGroupOut(),
        scheme="tool_loo",
        tune=False,
        fixed_params=PARAMS,
        verbose=False,
    )
    loo_metrics = metric_dict(y, loo.oof_prob, threshold=threshold)

    final_pipe = base_pipeline("RandomForest", params=PARAMS)
    final_pipe.fit(X, y)
    joblib.dump(
        {
            "model": final_pipe,
            "model_name": "RandomForest",
            "features": list(SENSOR_COLUMNS),
            "threshold": threshold,
            "params": PARAMS,
            "class_labels": ["Optimal (RDOC 4.5 mm)", "Non-optimal (RDOC 8 mm)"],
        },
        MODELS / "final_model.joblib",
    )

    # OOF predictions used by the dashboard/report.
    pred_df = clean[
        [
            "FileName", "ExperimentID", "LayerID", "P", "F", "C", "ToolIndex",
            "MillingToolType", "ADOC_mm", "RDOC_mm", "RDOC_class", "class_label",
            "CycleToFailure", "CycleToFailureNormalized",
        ]
    ].copy()
    pred_df["prob_tool5_best"] = primary.oof_prob
    pred_df["prob_tool_loo_best"] = loo.oof_prob
    pred_df["pred_tool5_best"] = (pred_df["prob_tool5_best"] >= threshold).astype(int)
    pred_df["correct_tool5_best"] = (pred_df["pred_tool5_best"] == pred_df["RDOC_class"]).astype(int)
    pred_df.to_csv(MODELS / "predictions_best.csv", index=False)

    # Confusion matrix and calibration for the selected RF model.
    pred = (primary.oof_prob >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
        xticklabels=["Optimal", "Dangerous"], yticklabels=["Optimal", "Dangerous"],
    )
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_title(f"Confusion matrix at threshold {threshold:.2f}")
    frac_pos, mean_pred = calibration_curve(y, primary.oof_prob, n_bins=10, strategy="quantile")
    axes[1].plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
    axes[1].plot(mean_pred, frac_pos, marker="o", color="#6A994E", linewidth=2)
    axes[1].set_xlabel("Mean predicted probability")
    axes[1].set_ylabel("Observed dangerous fraction")
    axes[1].set_title("Calibration (quantile bins)")
    fig.suptitle("Selected model: Random forest", fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig09_confusion_calibration.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Per-tool OOF probabilities.
    per_tool = clean[["ToolIndex", "class_label", "RDOC_class"]].copy()
    per_tool["prob"] = loo.oof_prob
    per_tool["tool_label"] = "Tool " + per_tool["ToolIndex"].astype(int).astype(str)
    tool_order = per_tool.groupby("tool_label")["RDOC_class"].max().sort_values(ascending=False).index.tolist()
    fig, ax = plt.subplots(figsize=(11, 4.8))
    sns.boxplot(
        data=per_tool, x="tool_label", y="prob", order=tool_order, hue="class_label",
        hue_order=CLASS_ORDER, palette=PALETTE, width=0.6, showfliers=False, ax=ax,
    )
    sns.stripplot(
        data=per_tool, x="tool_label", y="prob", order=tool_order, hue="class_label",
        hue_order=CLASS_ORDER, palette=PALETTE, dodge=False, size=1.8, alpha=0.18, ax=ax,
    )
    ax.axhline(threshold, color="#333333", linestyle="--", linewidth=1)
    ax.set_xlabel("")
    ax.set_ylabel("Out-of-fold P(dangerous)")
    ax.set_title("Per-tool generalisation: Random forest (leave-one-tool-out)")
    if ax.legend_ is not None:
        ax.legend_.remove()
    handles = [
        Patch(facecolor=PALETTE["Optimal (RDOC 4.5 mm)"], label="Optimal (RDOC 4.5 mm)"),
        Patch(facecolor=PALETTE["Non-optimal (RDOC 8 mm)"], label="Non-optimal (RDOC 8 mm)"),
        Line2D([0], [0], color="#333333", linestyle="--", linewidth=1, label=f"threshold {threshold:.2f}"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper right")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig11_per_tool_probabilities.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    importance = save_importance(final_pipe, X, y, list(SENSOR_COLUMNS), FIGURES, MODELS)

    # Holdout demonstration.
    holdout_mask = clean["is_demo_holdout"].astype(int).to_numpy() == 1
    holdout_pipe = base_pipeline("RandomForest", params=PARAMS)
    holdout_pipe.fit(X.loc[~holdout_mask], y[~holdout_mask])
    holdout_prob = holdout_pipe.predict_proba(X.loc[holdout_mask])[:, 1]
    joblib.dump(
        {"model": holdout_pipe, "features": list(SENSOR_COLUMNS), "threshold": threshold, "model_name": "RandomForest"},
        MODELS / "holdout_model.joblib",
    )
    holdout = clean.loc[
        holdout_mask,
        ["FileName", "ExperimentID", "ToolIndex", "MillingToolType", "CycleToFailure", "RDOC_class", "class_label"],
    ].copy()
    holdout["prob_dangerous"] = holdout_prob
    holdout["pred_dangerous"] = (holdout_prob >= threshold).astype(int)
    holdout["correct"] = (holdout["pred_dangerous"] == holdout["RDOC_class"]).astype(int)
    holdout.to_csv(MODELS / "holdout_predictions.csv", index=False)
    unseen = (
        holdout.groupby(["ExperimentID", "class_label"], as_index=False)
        .agg(
            n_cycles=("FileName", "size"),
            prob_mean=("prob_dangerous", "mean"),
            prob_min=("prob_dangerous", "min"),
            prob_max=("prob_dangerous", "max"),
            frac_predicted_dangerous=("pred_dangerous", "mean"),
        )
        .sort_values("ExperimentID")
    )
    unseen["true_dangerous"] = (unseen["class_label"] == "Non-optimal (RDOC 8 mm)").astype(int)
    unseen.to_csv(MODELS / "unseen_experiment_summary.csv", index=False)
    reps = holdout.sort_values(["ExperimentID", "CycleToFailure"], ascending=[True, False]).groupby("ExperimentID", as_index=False).head(1).copy()
    reps = pd.concat(
        [reps, holdout.sort_values(["ExperimentID", "CycleToFailure"], ascending=[True, True]).groupby("ExperimentID", as_index=False).head(1)],
        ignore_index=True,
    ).drop_duplicates(subset=["ExperimentID", "FileName"])
    reps.to_csv(MODELS / "unseen_cases.csv", index=False)

    # Update the saved metrics payload.
    payload = json.loads((MODELS / "model_metrics.json").read_text(encoding="utf-8"))
    payload["best_model"] = "RandomForest"
    payload["best_model_label"] = MODEL_LABELS["RandomForest"]
    payload["threshold"] = threshold
    payload["threshold_info"] = threshold_info
    payload["best_params"] = PARAMS
    payload["selection_reason"] = (
        "Random forest was selected on the operating-point metrics (recall, precision, MCC, "
        "balanced accuracy, Brier) after threshold tuning. Hist gradient boosting had the highest "
        "threshold-independent PR-AUC, but random forest gave the best safety-oriented trade-off."
    )
    payload["selected_model_metrics"] = selected
    payload["selected_model_tool_loo_metrics"] = loo_metrics
    payload["models"]["RandomForest"]["tool_5fold"]["selected_threshold_metrics"] = selected
    payload["models"]["RandomForest"]["tool_loo"]["selected_threshold_metrics"] = loo_metrics
    (MODELS / "model_metrics.json").write_text(
        json.dumps(payload, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
        encoding="utf-8",
    )

    comparison = pd.read_csv(MODELS / "model_comparison_pooled.csv")
    lines = [
        "ME-446 CEP modeling summary",
        "===========================",
        "",
        f"Selected model: {MODEL_LABELS['RandomForest']} (RandomForest)",
        f"Decision threshold: {threshold:.3f} (max-F1 on nested tool-group OOF)",
        f"Selection rationale: {payload['selection_reason']}",
        "",
        "Selected model metrics (nested 5-fold tool-group OOF, threshold-tuned):",
        f"  recall={selected['recall']:.3f} precision={selected['precision']:.3f} "
        f"F1={selected['f1']:.3f} MCC={selected['mcc']:.3f} "
        f"balanced_acc={selected['balanced_accuracy']:.3f} Brier={selected['brier']:.3f}",
        f"  confusion_matrix={selected['confusion_matrix']}",
        "",
        "Selected model leave-one-tool-out sensitivity (threshold-tuned):",
        f"  recall={loo_metrics['recall']:.3f} precision={loo_metrics['precision']:.3f} "
        f"F1={loo_metrics['f1']:.3f} MCC={loo_metrics['mcc']:.3f}",
        "",
        "Primary model comparison (5-fold tool-group CV, threshold 0.5):",
        comparison[comparison["scheme"] == "tool_5fold"].to_string(index=False),
        "",
        "Leave-one-tool-out sensitivity (threshold 0.5):",
        comparison[comparison["scheme"] == "tool_loo"].to_string(index=False),
        "",
        "Random 5-fold split (negative control, leaky):",
        comparison[comparison["scheme"] == "random_5fold"].to_string(index=False),
        "",
        "Unseen experiment holdout (P002, P108):",
        unseen.to_string(index=False),
        "",
        "Top 10 permutation features for the selected random forest:",
        importance.head(10).to_string(index=False),
    ]
    (MODELS / "MODELING_SUMMARY.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote selected random-forest artifacts to {MODELS}")


if __name__ == "__main__":
    main()
