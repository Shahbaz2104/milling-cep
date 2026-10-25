"""Export Tableau-ready extracts for the milling-condition dashboard."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_PROCESSED, MODELS, TABLEAU
from src.modeling import metric_dict

PALETTE_LABELS = {0: "Optimal", 1: "Dangerous"}


def main() -> None:
    TABLEAU.mkdir(parents=True, exist_ok=True)

    clean = pd.read_csv(DATA_PROCESSED / "milling_clean.csv")
    pred = pd.read_csv(MODELS / "predictions_best.csv")
    sensor_long = pd.read_csv(DATA_PROCESSED / "sensor_long.csv")
    comparison = pd.read_csv(MODELS / "model_comparison_pooled.csv")
    importance = pd.read_csv(MODELS / "feature_importance.csv")
    unseen = pd.read_csv(MODELS / "unseen_experiment_summary.csv")
    holdout = pd.read_csv(MODELS / "holdout_predictions.csv")
    metrics = json.loads((MODELS / "model_metrics.json").read_text(encoding="utf-8"))
    threshold = float(metrics["threshold"])

    y = pred["RDOC_class"].astype(int).to_numpy()
    prob5 = pred["prob_tool5_best"].to_numpy()
    prob_loo = pred["prob_tool_loo_best"].to_numpy()
    m5 = metric_dict(y, prob5, threshold=threshold)
    mloo = metric_dict(y, prob_loo, threshold=threshold)

    # 1) Cycle-level fact table -------------------------------------------------
    cycles = pred.copy()
    cycles["tool_label"] = "Tool " + cycles["ToolIndex"].astype(int).astype(str)
    cycles["threshold"] = threshold
    cycles["pred_tool5"] = (cycles["prob_tool5_best"] >= threshold).astype(int)
    cycles["pred_loo"] = (cycles["prob_tool_loo_best"] >= threshold).astype(int)
    cycles["correct_tool5"] = (cycles["pred_tool5"] == cycles["RDOC_class"]).astype(int)
    cycles["correct_loo"] = (cycles["pred_loo"] == cycles["RDOC_class"]).astype(int)
    cycles["outcome_tool5"] = np.select(
        [
            (cycles["RDOC_class"] == 1) & (cycles["pred_tool5"] == 1),
            (cycles["RDOC_class"] == 1) & (cycles["pred_tool5"] == 0),
            (cycles["RDOC_class"] == 0) & (cycles["pred_tool5"] == 1),
        ],
        ["True positive", "False negative (missed)", "False positive (false alarm)"],
        default="True negative",
    )
    cycles["class_short"] = cycles["RDOC_class"].map(PALETTE_LABELS)
    cycles = cycles.rename(
        columns={
            "prob_tool5_best": "prob_primary",
            "prob_tool_loo_best": "prob_leave_one_tool_out",
            "CycleToFailureNormalized": "cycle_normalized",
        }
    )
    cycle_cols = [
        "FileName", "ExperimentID", "LayerID", "P", "F", "C", "ToolIndex", "tool_label",
        "MillingToolType", "ADOC_mm", "RDOC_mm", "RDOC_class", "class_short", "class_label",
        "CycleToFailure", "cycle_normalized", "prob_primary", "prob_leave_one_tool_out",
        "pred_tool5", "pred_loo", "correct_tool5", "correct_loo", "outcome_tool5", "threshold",
    ]
    cycles[cycle_cols].to_csv(TABLEAU / "tableau_cycles.csv", index=False)

    # 2) KPI tiles --------------------------------------------------------------
    kpis = {
        "cycles": int(len(pred)),
        "experiments": int(pred["ExperimentID"].nunique()),
        "tools": int(pred["ToolIndex"].nunique()),
        "dangerous_cycles": int(y.sum()),
        "dangerous_share": float(y.mean()),
        "threshold": threshold,
        "recall": m5["recall"],
        "precision": m5["precision"],
        "f1": m5["f1"],
        "mcc": m5["mcc"],
        "balanced_accuracy": m5["balanced_accuracy"],
        "brier": m5["brier"],
        "roc_auc": m5["roc_auc"],
        "pr_auc": m5["pr_auc"],
        "false_alarms": int(m5["confusion_matrix"][0][1]),
        "missed_dangerous": int(m5["confusion_matrix"][1][0]),
        "true_dangerous": int(m5["confusion_matrix"][1][1]),
        "true_optimal": int(m5["confusion_matrix"][0][0]),
        "loo_recall": mloo["recall"],
        "loo_precision": mloo["precision"],
        "loo_mcc": mloo["mcc"],
        "loo_roc_auc": mloo["roc_auc"],
        "loo_pr_auc": mloo["pr_auc"],
        "selected_model": metrics["best_model_label"],
    }
    pd.DataFrame([kpis]).to_csv(TABLEAU / "tableau_kpis.csv", index=False)

    # 3) Tool-level summary -----------------------------------------------------
    tool = (
        cycles.groupby(["ToolIndex", "tool_label", "class_short", "class_label", "RDOC_class"], as_index=False)
        .agg(
            cycles=("FileName", "size"),
            mean_probability=("prob_primary", "mean"),
            median_probability=("prob_primary", "median"),
            mean_probability_loo=("prob_leave_one_tool_out", "mean"),
            median_probability_loo=("prob_leave_one_tool_out", "median"),
            flagged_fraction=("pred_tool5", "mean"),
            flagged_fraction_loo=("pred_loo", "mean"),
            misses=("outcome_tool5", lambda s: int((s == "False negative (missed)").sum())),
            false_alarms=("outcome_tool5", lambda s: int((s == "False positive (false alarm)").sum())),
        )
        .sort_values(["RDOC_class", "ToolIndex"], ascending=[False, True])
    )
    tool["metric_name"] = np.where(
        tool["class_short"] == "Dangerous",
        "Recall (dangerous tool)",
        "False-alarm rate (optimal tool)",
    )
    tool.to_csv(TABLEAU / "tableau_tool_summary.csv", index=False)

    # 4) Sensor long extract (slim columns for Tableau) --------------------------
    sensor_long = sensor_long.assign(class_short=sensor_long["RDOC_class"].map(PALETTE_LABELS))
    slim = sensor_long[
        [
            "FileName", "ExperimentID", "ToolIndex", "class_short", "class_label",
            "CycleToFailureNormalized", "feature", "family", "channel_or_axis", "statistic", "value",
        ]
    ].rename(columns={"CycleToFailureNormalized": "cycle_normalized"})
    slim.to_csv(TABLEAU / "tableau_sensor_long.csv", index=False)

    # 5) Model comparison and importance ---------------------------------------
    comparison.rename(
        columns={
            "roc_auc": "ROC_AUC",
            "pr_auc": "PR_AUC",
            "balanced_accuracy": "balanced_accuracy",
        }
    ).to_csv(TABLEAU / "tableau_model_comparison.csv", index=False)
    importance.head(30).to_csv(TABLEAU / "tableau_feature_importance.csv", index=False)

    # 6) Unseen-case evidence ---------------------------------------------------
    unseen.to_csv(TABLEAU / "tableau_unseen_experiments.csv", index=False)
    holdout.rename(
        columns={
            "prob_dangerous": "prob_primary",
            "pred_dangerous": "predicted_dangerous",
            "CycleToFailure": "cycle_to_failure",
        }
    ).to_csv(TABLEAU / "tableau_unseen_cycles.csv", index=False)

    print(f"Wrote Tableau extracts to {TABLEAU}")
    for path in sorted(TABLEAU.glob("*.csv")):
        print(f"  {path.name:<36} {path.stat().st_size / 1024:8.1f} kB")


if __name__ == "__main__":
    main()
