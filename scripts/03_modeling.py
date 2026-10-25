"""Run the full modeling workflow and write models, metrics and figures."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_PROCESSED, FIGURES, MODELS, SENSOR_COLUMNS
from src.modeling import (
    MODEL_LABELS,
    base_pipeline,
    choose_threshold,
    logistic_coefficients,
    run_cv,
    save_curves_and_metrics,
    save_importance,
    tune_all_data,
)

MODEL_NAMES = ["LogisticRegression", "RandomForest", "HistGradientBoosting"]
ALL_MODELS = ["Dummy"] + MODEL_NAMES


def main() -> None:
    t_start = time.time()
    clean = pd.read_csv(DATA_PROCESSED / "milling_clean.csv")
    X = clean[SENSOR_COLUMNS].copy()
    y = clean["RDOC_class"].astype(int).to_numpy()
    tool_groups = clean["ToolIndex"].astype(int).to_numpy()
    tool5_groups = clean["tool_5fold"].astype(int).to_numpy()
    experiment_groups = clean["ExperimentID"].astype(str).to_numpy()

    assert len(X) == len(y) == len(tool_groups) == len(clean)
    assert X.isna().sum().sum() == 0
    assert np.isfinite(X.to_numpy()).all()
    print(f"Data: {X.shape[0]} cycles x {X.shape[1]} sensor features; dangerous={int(y.sum())} ({y.mean():.1%})")
    print(f"Tools: {len(np.unique(tool_groups))}; experiments: {len(np.unique(experiment_groups))}")

    results = {}

    # 1) Primary estimate: nested tuning inside 5-fold tool-group outer CV.
    print("\n[1/4] Nested 5-fold tool-group CV (primary estimate)")
    for name in ALL_MODELS:
        key = f"{name}__tool_5fold"
        results[key] = run_cv(
            name,
            X,
            y,
            tool5_groups,
            LeaveOneGroupOut(),
            scheme="tool_5fold",
            use_groups=True,
            tune=name != "Dummy",
        )

    # 2) Tune once on all data for sensitivity analyses and the final model.
    print("\n[2/4] Grouped hyperparameter tuning on all data")
    final_searches = {}
    final_params = {}
    for name in MODEL_NAMES:
        print(f"  Tuning {name}...")
        final_searches[name] = tune_all_data(name, X, y, tool_groups)
        final_params[name] = final_searches[name]["best_params"]
        print(f"    best group-CV AP={final_searches[name]['best_score']:.3f} params={final_params[name]}")
    final_params["Dummy"] = {}

    # 3) Sensitivity analyses with fixed tuned parameters.
    print("\n[3/4] Sensitivity analyses: leave-one-tool-out, leave-one-experiment-out, random split")
    for name in ALL_MODELS:
        key = f"{name}__tool_loo"
        results[key] = run_cv(
            name,
            X,
            y,
            tool_groups,
            LeaveOneGroupOut(),
            scheme="tool_loo",
            use_groups=True,
            tune=False,
            fixed_params=final_params[name],
        )
    for name in ALL_MODELS:
        key = f"{name}__experiment_loo"
        results[key] = run_cv(
            name,
            X,
            y,
            experiment_groups,
            LeaveOneGroupOut(),
            scheme="experiment_loo",
            use_groups=True,
            tune=False,
            fixed_params=final_params[name],
        )
    random_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20261025)
    for name in ALL_MODELS:
        key = f"{name}__random_5fold"
        results[key] = run_cv(
            name,
            X,
            y,
            tool_groups,
            random_cv,
            scheme="random_5fold",
            use_groups=False,
            tune=False,
            fixed_params=final_params[name],
        )

    # 4) Select the best model on the primary tool-group estimate.
    primary = {name: results[f"{name}__tool_5fold"].pooled_metrics for name in ALL_MODELS}
    best_model = max(
        ALL_MODELS,
        key=lambda n: (primary[n]["pr_auc"], primary[n]["roc_auc"], primary[n]["recall"]),
    )
    print(f"\n[4/4] Best model on tool_5fold PR-AUC: {best_model}")
    for name in ALL_MODELS:
        m = primary[name]
        print(
            f"  {MODEL_LABELS.get(name, name):24s} ROC-AUC={m['roc_auc']:.3f} "
            f"PR-AUC={m['pr_auc']:.3f} recall={m['recall']:.3f} precision={m['precision']:.3f} "
            f"balanced_acc={m['balanced_accuracy']:.3f}"
        )

    best_oof = results[f"{best_model}__tool_5fold"].oof_prob
    threshold_info = choose_threshold(y, best_oof)
    threshold = float(threshold_info["high_recall_threshold"])
    if threshold_info["high_recall_precision"] < 0.50:
        threshold = float(threshold_info["max_f1_threshold"])
    print(f"Threshold: {threshold:.3f} (info: {threshold_info})")

    # Save comparison, curves, confusion/calibration, per-tool plot, and predictions.
    save_curves_and_metrics(
        results,
        best_model,
        FIGURES,
        MODELS,
        y,
        clean,
        threshold,
    )

    # Final deployment model: tuned on all available data.
    final_bundle = {
        "model": final_searches[best_model]["best_estimator"],
        "model_name": best_model,
        "features": list(SENSOR_COLUMNS),
        "threshold": threshold,
        "params": final_params[best_model],
        "class_labels": ["Optimal (RDOC 4.5 mm)", "Non-optimal (RDOC 8 mm)"],
    }
    joblib.dump(final_bundle, MODELS / "final_model.joblib")

    # Feature importance and logistic coefficient table.
    importance = save_importance(
        final_bundle["model"],
        X,
        y,
        list(SENSOR_COLUMNS),
        FIGURES,
        MODELS,
    )
    logistic_params = final_params.get("LogisticRegression", {"clf__C": 1.0})
    logistic_coefficients(X, y, list(SENSOR_COLUMNS), MODELS, params=logistic_params)

    # Holdout demonstration: train without P002 and P108, predict them.
    holdout_mask = clean["is_demo_holdout"].astype(int).to_numpy() == 1
    train_mask = ~holdout_mask
    holdout_pipe = base_pipeline(best_model, params=final_params[best_model])
    holdout_pipe.fit(X.loc[train_mask], y[train_mask])
    holdout_prob = holdout_pipe.predict_proba(X.loc[holdout_mask])[:, 1]
    holdout = clean.loc[
        holdout_mask,
        ["FileName", "ExperimentID", "ToolIndex", "MillingToolType", "CycleToFailure", "RDOC_class", "class_label"],
    ].copy()
    holdout["prob_dangerous"] = holdout_prob
    holdout["pred_dangerous"] = (holdout_prob >= threshold).astype(int)
    holdout["correct"] = (holdout["pred_dangerous"] == holdout["RDOC_class"]).astype(int)
    holdout.to_csv(MODELS / "holdout_predictions.csv", index=False)
    joblib.dump(
        {"model": holdout_pipe, "features": list(SENSOR_COLUMNS), "threshold": threshold, "model_name": best_model},
        MODELS / "holdout_model.joblib",
    )

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

    # Two representative cycles: first and last cycle of each held-out experiment.
    reps = (
        holdout.sort_values(["ExperimentID", "CycleToFailure"], ascending=[True, False])
        .groupby("ExperimentID", as_index=False)
        .head(1)
        .copy()
    )
    reps = pd.concat(
        [
            reps,
            holdout.sort_values(["ExperimentID", "CycleToFailure"], ascending=[True, True])
            .groupby("ExperimentID", as_index=False)
            .head(1),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["ExperimentID", "FileName"])
    reps.to_csv(MODELS / "unseen_cases.csv", index=False)

    # Save a compact metrics JSON and a text summary.
    metrics_payload = {
        "best_model": best_model,
        "best_model_label": MODEL_LABELS.get(best_model, best_model),
        "threshold": threshold,
        "threshold_info": threshold_info,
        "best_params": final_params[best_model],
        "n_cycles": int(len(y)),
        "n_features": int(X.shape[1]),
        "n_dangerous": int(y.sum()),
        "n_optimal": int((1 - y).sum()),
        "n_tools": int(len(np.unique(tool_groups))),
        "n_experiments": int(len(np.unique(experiment_groups))),
        "models": {},
    }
    for name in ALL_MODELS:
        metrics_payload["models"][name] = {}
        for scheme in ["tool_5fold", "tool_loo", "experiment_loo", "random_5fold"]:
            res = results[f"{name}__{scheme}"]
            metrics_payload["models"][name][scheme] = {
                "pooled": res.pooled_metrics,
                "fold_metrics": res.fold_metrics,
                "elapsed_seconds": res.elapsed_seconds,
                "fold_params": res.fold_params,
            }
    metrics_payload["top_features"] = importance.head(10).to_dict(orient="records")
    metrics_payload["unseen_experiments"] = unseen.to_dict(orient="records")
    (MODELS / "model_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
        encoding="utf-8",
    )

    summary_lines = [
        "ME-446 CEP modeling summary",
        "===========================",
        "",
        f"Best model: {MODEL_LABELS.get(best_model, best_model)} ({best_model})",
        f"Decision threshold: {threshold:.3f}",
        f"Threshold rationale: {threshold_info}",
        f"Cycles: {len(y)}; features: {X.shape[1]}; dangerous: {int(y.sum())}; optimal: {int((1-y).sum())}",
        "",
        "Pooled out-of-fold metrics (primary: 5-fold tool-group CV):",
    ]
    for name in ALL_MODELS:
        m = primary[name]
        summary_lines.append(
            f"  {MODEL_LABELS.get(name, name):24s} "
            f"ROC-AUC={m['roc_auc']:.3f} PR-AUC={m['pr_auc']:.3f} "
            f"recall={m['recall']:.3f} precision={m['precision']:.3f} "
            f"balanced_acc={m['balanced_accuracy']:.3f} MCC={m['mcc']:.3f}"
        )
    summary_lines += [
        "",
        "Leave-one-tool-out (sensitivity):",
    ]
    for name in ALL_MODELS:
        m = results[f"{name}__tool_loo"].pooled_metrics
        summary_lines.append(
            f"  {MODEL_LABELS.get(name, name):24s} "
            f"ROC-AUC={m['roc_auc']:.3f} PR-AUC={m['pr_auc']:.3f} "
            f"recall={m['recall']:.3f} precision={m['precision']:.3f}"
        )
    summary_lines += [
        "",
        "Random 5-fold split (negative control; leaky):",
    ]
    for name in ALL_MODELS:
        m = results[f"{name}__random_5fold"].pooled_metrics
        summary_lines.append(
            f"  {MODEL_LABELS.get(name, name):24s} "
            f"ROC-AUC={m['roc_auc']:.3f} PR-AUC={m['pr_auc']:.3f} "
            f"accuracy={m['accuracy']:.3f}"
        )
    summary_lines += [
        "",
        "Unseen experiment holdout (P002, P108):",
        unseen.to_string(index=False),
        "",
        "Top 10 permutation features:",
        importance.head(10).to_string(index=False),
        "",
        f"Total modeling time: {time.time() - t_start:.1f} s",
    ]
    (MODELS / "MODELING_SUMMARY.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n" + "\n".join(summary_lines))
    print(f"\nWrote models to {MODELS}")


if __name__ == "__main__":
    main()
