"""Modeling utilities: pipelines, grouped validation, metrics and plots."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    LeaveOneGroupOut,
    RandomizedSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler

from src.config import RANDOM_STATE

PALETTE = {
    "Optimal (RDOC 4.5 mm)": "#2E86AB",
    "Non-optimal (RDOC 8 mm)": "#D1495B",
}
CLASS_ORDER = ["Optimal (RDOC 4.5 mm)", "Non-optimal (RDOC 8 mm)"]
MODEL_COLORS = {
    "Dummy": "#8D99AE",
    "LogisticRegression": "#2E86AB",
    "RandomForest": "#6A994E",
    "HistGradientBoosting": "#BC4749",
}
MODEL_LABELS = {
    "Dummy": "Dummy (prior)",
    "LogisticRegression": "Logistic regression",
    "RandomForest": "Random forest",
    "HistGradientBoosting": "Hist gradient boosting",
}


def signed_log1p(X):
    X = np.asarray(X, dtype=float, order="C")
    return np.ascontiguousarray(np.sign(X) * np.log1p(np.abs(X)))


def base_pipeline(model_name: str, params: dict[str, Any] | None = None) -> Pipeline:
    if model_name == "LogisticRegression":
        clf = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )
    elif model_name == "RandomForest":
        clf = RandomForestClassifier(
            n_estimators=150,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    elif model_name == "HistGradientBoosting":
        clf = HistGradientBoostingClassifier(
            max_iter=100,
            early_stopping=False,
            random_state=RANDOM_STATE,
        )
    elif model_name == "Dummy":
        clf = DummyClassifier(strategy="prior")
    else:
        raise ValueError(f"Unknown model: {model_name}")
    steps = [("log", FunctionTransformer(signed_log1p, validate=False))]
    if model_name == "LogisticRegression":
        steps.append(("scale", RobustScaler()))
    steps.append(("clf", clf))
    pipe = Pipeline(steps)
    if params:
        pipe.set_params(**params)
    return pipe


SEARCH_SPACES = {
    "LogisticRegression": {
        "clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
    },
    "RandomForest": {
        "clf__max_depth": [6, 10, None],
        "clf__min_samples_leaf": [1, 3, 5],
    },
    "HistGradientBoosting": {
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_leaf_nodes": [15, 31],
        "clf__l2_regularization": [0.0, 1.0],
    },
    "Dummy": {},
}

N_ITER = {
    "LogisticRegression": 4,
    "RandomForest": 3,
    "HistGradientBoosting": 3,
    "Dummy": 1,
}


def metric_dict(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    out: dict[str, Any] = {
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "accuracy": float(accuracy_score(y_true, pred)),
        "brier": float(brier_score_loss(y_true, prob)),
        "mcc": float(matthews_corrcoef(y_true, pred)) if len(np.unique(y_true)) > 1 else float("nan"),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "specificity": float(recall_score(y_true, pred, pos_label=0, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)) if len(np.unique(y_true)) > 1 else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }
    return out


def _safe_split(splitter, X, y, groups, use_groups: bool):
    if use_groups:
        return list(splitter.split(X, y, groups=groups))
    return list(splitter.split(X, y))


@dataclass
class CVResult:
    model_name: str
    scheme: str
    oof_prob: np.ndarray
    fold_metrics: list[dict[str, Any]]
    fold_params: list[dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    pooled_metrics: dict[str, Any] = field(default_factory=dict)
    per_fold_metrics: pd.DataFrame | None = None


def run_cv(
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    splitter,
    scheme: str,
    *,
    use_groups: bool = True,
    tune: bool = False,
    fixed_params: dict[str, Any] | None = None,
    n_iter: int | None = None,
    verbose: bool = True,
) -> CVResult:
    n = len(y)
    oof = np.full(n, np.nan, dtype=float)
    fold_metrics: list[dict[str, Any]] = []
    fold_params: list[dict[str, Any]] = []
    splits = _safe_split(splitter, X, y, groups, use_groups)
    t0 = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        y_train = y[train_idx]
        if len(np.unique(y_train)) < 2:
            raise ValueError(f"{scheme} fold {fold_idx} has a single class in training")
        params = fixed_params or {}
        if tune and model_name != "Dummy":
            estimator = base_pipeline(model_name)
            inner_groups = groups[train_idx] if use_groups else None
            inner_cv = StratifiedGroupKFold(
                n_splits=min(2, len(np.unique(inner_groups))),
                shuffle=True,
                random_state=RANDOM_STATE + fold_idx,
            )
            search = RandomizedSearchCV(
                estimator,
                param_distributions=SEARCH_SPACES[model_name],
                n_iter=n_iter or N_ITER[model_name],
                scoring="average_precision",
                cv=inner_cv,
                refit=True,
                n_jobs=1,
                random_state=RANDOM_STATE + fold_idx,
                verbose=0,
            )
            search.fit(X.iloc[train_idx], y_train, groups=inner_groups)
            best = search.best_estimator_
            best_params = search.best_params_
            fold_params.append(best_params)
        else:
            best = base_pipeline(model_name, params=params)
            best.fit(X.iloc[train_idx], y_train)
        prob = best.predict_proba(X.iloc[test_idx])[:, 1]
        oof[test_idx] = prob
        metrics = metric_dict(y[test_idx], prob, threshold=0.5)
        metrics.update({"fold": fold_idx, "n_train": int(len(train_idx)), "n_test": int(len(test_idx))})
        fold_metrics.append(metrics)
        if verbose:
            auc = metrics["roc_auc"]
            print(
                f"  [{scheme}] {model_name} fold {fold_idx + 1}/{len(splits)} "
                f"n_test={len(test_idx)} AUC={auc:.3f} PR-AUC={metrics['pr_auc']:.3f}"
            )

    pooled = metric_dict(y, oof, threshold=0.5)
    per_fold = pd.DataFrame(fold_metrics)
    return CVResult(
        model_name=model_name,
        scheme=scheme,
        oof_prob=oof,
        fold_metrics=fold_metrics,
        fold_params=fold_params,
        elapsed_seconds=time.time() - t0,
        pooled_metrics=pooled,
        per_fold_metrics=per_fold,
    )


def tune_all_data(
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_iter: int | None = None,
) -> dict[str, Any]:
    estimator = base_pipeline(model_name)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator,
        param_distributions=SEARCH_SPACES[model_name],
        n_iter=n_iter or N_ITER[model_name],
        scoring="average_precision",
        cv=cv,
        refit=True,
        n_jobs=1,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    search.fit(X, y, groups=groups)
    return {
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
        "best_estimator": search.best_estimator_,
    }


def choose_threshold(y: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    precision, recall, thresholds = precision_recall_curve(y, prob)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    best_idx = int(np.argmax(f1[:-1])) if len(thresholds) else 0
    # Engineering preference: highest precision among thresholds with recall >= 0.90.
    candidates = np.where(recall[:-1] >= 0.90)[0]
    if len(candidates):
        high_recall_idx = candidates[np.argmax(precision[candidates])]
    else:
        high_recall_idx = best_idx
    return {
        "max_f1_threshold": float(thresholds[best_idx]) if len(thresholds) else 0.5,
        "max_f1": float(f1[best_idx]),
        "high_recall_threshold": float(thresholds[high_recall_idx]) if len(thresholds) else 0.5,
        "high_recall_precision": float(precision[high_recall_idx]),
        "high_recall_recall": float(recall[high_recall_idx]),
    }


def save_curves_and_metrics(
    results: dict[str, CVResult],
    best_model: str,
    figures_dir,
    models_dir,
    y: np.ndarray,
    clean: pd.DataFrame,
    threshold: float,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Comparison table: primary tool-5 and leave-one-tool-out.
    rows = []
    for scheme in ["tool_5fold", "tool_loo", "experiment_loo", "random_5fold"]:
        for name, res in results.items():
            if res.scheme != scheme:
                continue
            m = res.pooled_metrics
            rows.append(
                {
                    "model": res.model_name,
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
    comparison.to_csv(models_dir / "model_comparison_pooled.csv", index=False)

    fold_frames = []
    for res in results.values():
        if res.per_fold_metrics is None:
            continue
        frame = res.per_fold_metrics.copy()
        frame["model"] = res.model_name
        frame["scheme"] = res.scheme
        fold_frames.append(frame)
    pd.concat(fold_frames, ignore_index=True).to_csv(models_dir / "model_comparison_folds.csv", index=False)

    # Model comparison chart: primary scheme.
    primary = comparison[comparison["scheme"] == "tool_5fold"].set_index("model")
    order = [m for m in primary.index if m in MODEL_COLORS]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(order))
    width = 0.36
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
        baseline = 0.5 if metric == "roc_auc" else float(np.mean(y))
        ax.axhline(baseline, color="#777777", linestyle="--", linewidth=1)
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", fontsize=8)
    fig.suptitle("Model comparison under tool-grouped 5-fold validation", fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig07_model_comparison.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ROC and PR curves for all models on the primary scheme.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for name in ["Dummy", "LogisticRegression", "RandomForest", "HistGradientBoosting"]:
        res = results.get(f"{name}__tool_5fold")
        if res is None:
            continue
        color = MODEL_COLORS.get(name, "#777777")
        lw = 2.4 if name == best_model else 1.5
        fpr, tpr, _ = roc_curve(y, res.oof_prob)
        axes[0].plot(fpr, tpr, color=color, linewidth=lw, label=f"{MODEL_LABELS.get(name, name)} (AUC {res.pooled_metrics['roc_auc']:.3f})")
        precision, recall, _ = precision_recall_curve(y, res.oof_prob)
        axes[1].plot(recall, precision, color=color, linewidth=lw, label=f"{MODEL_LABELS.get(name, name)} (AP {res.pooled_metrics['pr_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curves")
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    axes[1].set_xlabel("Recall (dangerous class)")
    axes[1].set_ylabel("Precision (dangerous class)")
    axes[1].set_title("Precision-recall curves")
    axes[1].legend(frameon=False, fontsize=7, loc="lower left")
    fig.suptitle("Out-of-fold curves under 5-fold tool-group validation", fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig08_model_curves.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Best model: confusion matrix and calibration.
    best = results[f"{best_model}__tool_5fold"]
    prob = best.oof_prob
    pred = (prob >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=axes[0],
        xticklabels=["Optimal", "Dangerous"],
        yticklabels=["Optimal", "Dangerous"],
    )
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_title(f"Confusion matrix at threshold {threshold:.2f}")
    frac_pos, mean_pred = calibration_curve(y, prob, n_bins=10, strategy="quantile")
    axes[1].plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
    axes[1].plot(mean_pred, frac_pos, marker="o", color=MODEL_COLORS.get(best_model, "#333333"), linewidth=2)
    axes[1].set_xlabel("Mean predicted probability")
    axes[1].set_ylabel("Observed dangerous fraction")
    axes[1].set_title("Calibration (quantile bins)")
    fig.suptitle(f"Best model: {MODEL_LABELS.get(best_model, best_model)}", fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig09_confusion_calibration.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Per-tool OOF probabilities (leave-one-tool-out).
    loo = results[f"{best_model}__tool_loo"]
    per_tool = clean[["ToolIndex", "class_label", "RDOC_class", "ExperimentID"]].copy()
    per_tool["prob"] = loo.oof_prob
    per_tool["tool_label"] = "Tool " + per_tool["ToolIndex"].astype(int).astype(str)
    tool_order = (
        per_tool.groupby("tool_label")["RDOC_class"].max().sort_values(ascending=False).index.tolist()
    )
    fig, ax = plt.subplots(figsize=(11, 4.8))
    sns.boxplot(
        data=per_tool,
        x="tool_label",
        y="prob",
        order=tool_order,
        hue="class_label",
        hue_order=CLASS_ORDER,
        palette=PALETTE,
        width=0.6,
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=per_tool,
        x="tool_label",
        y="prob",
        order=tool_order,
        hue="class_label",
        hue_order=CLASS_ORDER,
        palette=PALETTE,
        dodge=False,
        size=1.8,
        alpha=0.18,
        ax=ax,
    )
    ax.axhline(threshold, color="#333333", linestyle="--", linewidth=1)
    ax.set_xlabel("")
    ax.set_ylabel("Out-of-fold P(dangerous)")
    ax.set_title(f"Per-tool generalisation: {MODEL_LABELS.get(best_model, best_model)} (leave-one-tool-out)")
    if ax.legend_ is not None:
        ax.legend_.remove()
    custom_handles = [
        Patch(facecolor=PALETTE["Optimal (RDOC 4.5 mm)"], label="Optimal (RDOC 4.5 mm)"),
        Patch(facecolor=PALETTE["Non-optimal (RDOC 8 mm)"], label="Non-optimal (RDOC 8 mm)"),
        Line2D([0], [0], color="#333333", linestyle="--", linewidth=1, label=f"threshold {threshold:.2f}"),
    ]
    ax.legend(handles=custom_handles, frameon=False, fontsize=8, loc="upper right")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig11_per_tool_probabilities.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Save the best-model OOF predictions for dashboard and report.
    pred_df = clean[
        [
            "FileName",
            "ExperimentID",
            "LayerID",
            "P",
            "F",
            "C",
            "ToolIndex",
            "MillingToolType",
            "ADOC_mm",
            "RDOC_mm",
            "RDOC_class",
            "class_label",
            "CycleToFailure",
            "CycleToFailureNormalized",
        ]
    ].copy()
    pred_df["prob_tool5_best"] = best.oof_prob
    pred_df["prob_tool_loo_best"] = loo.oof_prob
    pred_df["pred_tool5_best"] = (pred_df["prob_tool5_best"] >= threshold).astype(int)
    pred_df["correct_tool5_best"] = (pred_df["pred_tool5_best"] == pred_df["RDOC_class"]).astype(int)
    pred_df.to_csv(models_dir / "predictions_best.csv", index=False)


def save_importance(
    final_pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    figures_dir,
    models_dir,
) -> pd.DataFrame:
    result = permutation_importance(
        final_pipeline,
        X,
        y,
        scoring="roc_auc",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance.to_csv(models_dir / "feature_importance.csv", index=False)

    top = importance.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    colors = ["#2E86AB" if f.startswith("Accelerometer") else "#C76D2E" for f in top["feature"]]
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color=colors, alpha=0.9)
    ax.set_xlabel("Permutation importance (drop in ROC-AUC)")
    ax.set_title("Top sensor features for the final model")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig10_feature_importance.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return importance


def logistic_coefficients(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    models_dir,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    pipe = base_pipeline("LogisticRegression", params=params or {"clf__C": 1.0})
    pipe.fit(X, y)
    coef = pipe.named_steps["clf"].coef_[0]
    out = pd.DataFrame({"feature": feature_names, "coefficient": coef}).sort_values(
        "coefficient", key=lambda s: s.abs(), ascending=False
    )
    out.to_csv(models_dir / "logistic_coefficients.csv", index=False)
    return out
