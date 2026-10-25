"""Exploratory data analysis for the milling CEP.

Generates publication-quality figures and a short engineering-insights note.
This script is descriptive only; no model selection uses these full-data plots.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import DATA_PROCESSED, FIGURES, REPORTS, SENSOR_COLUMNS

PALETTE = {
    "Optimal (RDOC 4.5 mm)": "#2E86AB",
    "Non-optimal (RDOC 8 mm)": "#D1495B",
}
CLASS_ORDER = ["Optimal (RDOC 4.5 mm)", "Non-optimal (RDOC 8 mm)"]
FAMILY_COLORS = {"Accelerometer": "#3A7CA5", "Current": "#C76D2E"}

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def rank_auc(values: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC with dangerous (label=1) as positive class."""
    r = rankdata(values)
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def short_feature(name: str) -> str:
    s = (
        name.replace("Accelerometer - ", "Acc ")
        .replace("Current - ", "Curr ")
        .replace("Driving axle ", "Drive ")
        .replace("Spindle ", "Spdl ")
    )
    return s


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_class_balance(df: pd.DataFrame) -> dict:
    levels = {
        "Cycles": df["class_label"].value_counts().reindex(CLASS_ORDER),
        "Experiments": df.groupby("ExperimentID")["class_label"].max().value_counts().reindex(CLASS_ORDER),
        "Tools": df.groupby("ToolIndex")["class_label"].max().value_counts().reindex(CLASS_ORDER),
    }
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (title, counts) in zip(axes, levels.items()):
        total = counts.sum()
        bars = ax.bar(counts.index, counts.values, color=[PALETTE[c] for c in counts.index], width=0.65)
        for bar, value in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.02,
                f"{int(value)}\n({value / total:.1%})",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_title(f"{title} by class")
        ax.set_ylabel("Count")
        ax.set_ylim(0, counts.max() * 1.28)
        ax.tick_params(axis="x", rotation=12)
    fig.suptitle("Class balance: non-optimal RDOC is a minority condition", fontsize=11, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "fig01_class_balance.png")
    return {k: {str(i): int(v) for i, v in v.items()} for k, v in levels.items()}


def fig_confounding(df: pd.DataFrame) -> pd.DataFrame:
    tool = (
        df.groupby("ToolIndex")
        .agg(
            n=("FileName", "size"),
            RDOC=("RDOC_mm", "first"),
            ADOC=("ADOC_mm", "first"),
            tool_type=("MillingToolType", "first"),
            hardness=("Hardness_HRC", "first"),
            holder=("ToolHolderLength_mm", "first"),
            rdoc_class=("RDOC_class", "first"),
        )
        .reset_index()
        .sort_values(["rdoc_class", "ToolIndex"], ascending=[False, True])
    )
    mat = pd.DataFrame(
        {
            "RDOC 4.5": (tool["RDOC"] == 4.5).astype(int).to_numpy(),
            "RDOC 8": (tool["RDOC"] == 8.0).astype(int).to_numpy(),
            "ADOC 5": (tool["ADOC"] == 5.0).astype(int).to_numpy(),
            "ADOC 10": (tool["ADOC"] == 10.0).astype(int).to_numpy(),
        },
        index=[f"Tool {int(t)}  (n={int(n)})" for t, n in zip(tool["ToolIndex"], tool["n"])],
    )
    annot = mat.replace({0: "", 1: "X"})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})
    sns.heatmap(
        mat,
        ax=axes[0],
        annot=annot,
        fmt="",
        cmap=sns.color_palette(["#F4F6F8", "#2E86AB"], as_cmap=True),
        cbar=False,
        linewidths=0.6,
        linecolor="white",
        annot_kws={"fontsize": 9, "fontweight": "bold", "color": "white"},
    )
    axes[0].set_title("Each tool is used under one RDOC / ADOC condition")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")
    axes[0].tick_params(axis="y", rotation=0)

    sns.boxplot(
        data=df,
        x="class_label",
        y="Hardness_HRC",
        order=CLASS_ORDER,
        hue="class_label",
        palette=PALETTE,
        width=0.5,
        legend=False,
        ax=axes[1],
    )
    sns.stripplot(
        data=df,
        x="class_label",
        y="Hardness_HRC",
        order=CLASS_ORDER,
        color="#333333",
        size=1.4,
        alpha=0.25,
        ax=axes[1],
    )
    means = df.groupby("class_label")["Hardness_HRC"].mean().reindex(CLASS_ORDER)
    for i, (label, mean) in enumerate(means.items()):
        axes[1].scatter(i, mean, marker="D", s=35, color="white", edgecolor="#333333", zorder=5)
        axes[1].annotate(f"mean {mean:.2f}", (i, mean), xytext=(8, 6), textcoords="offset points", fontsize=8)
    axes[1].set_title("Hardness also differs by class")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Mean workpiece hardness (HRC)")
    axes[1].tick_params(axis="x", rotation=8)
    fig.suptitle("Confounding: process settings and material differ between tools", fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig02_tool_condition_confounding.png")
    return tool


def compute_univariate(df: pd.DataFrame) -> pd.DataFrame:
    X = df[SENSOR_COLUMNS].astype(float)
    y = df["RDOC_class"].to_numpy()
    rows = []
    for col in SENSOR_COLUMNS:
        auc = rank_auc(X[col].to_numpy(), y)
        rows.append(
            {
                "feature": col,
                "auc": auc,
                "separation": max(auc, 1 - auc),
                "direction": "higher in dangerous" if auc >= 0.5 else "lower in dangerous",
                "family": col.split(" - ")[0],
                "statistic": col.split(" - ")[-1],
            }
        )
    out = pd.DataFrame(rows).sort_values("separation", ascending=False)
    return out


def fig_sensor_distributions(df: pd.DataFrame, uni: pd.DataFrame) -> None:
    top = uni.head(6)["feature"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2))
    for ax, feat in zip(axes.ravel(), top):
        auc = float(uni.loc[uni["feature"] == feat, "auc"].iloc[0])
        sns.boxplot(
            data=df,
            x="class_label",
            y=feat,
            order=CLASS_ORDER,
            hue="class_label",
            palette=PALETTE,
            width=0.55,
            legend=False,
            showfliers=False,
            ax=ax,
        )
        sns.stripplot(
            data=df,
            x="class_label",
            y=feat,
            order=CLASS_ORDER,
            color="#333333",
            size=1.2,
            alpha=0.22,
            ax=ax,
        )
        ax.set_title(f"{short_feature(feat)}\nAUC = {auc:.3f}")
        ax.set_xlabel("")
        ax.set_ylabel("Raw ADC statistic")
        ax.tick_params(axis="x", rotation=10)
    fig.suptitle("Top univariate sensor separations (descriptive, not group-validated)", fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig03_top_sensor_distributions.png")


def fig_pca_correlation(df: pd.DataFrame, uni: pd.DataFrame) -> None:
    X = df[SENSOR_COLUMNS].astype(float)
    Z = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=20261025)
    scores = pca.fit_transform(Z)
    pca_df = pd.DataFrame(
        {
            "PC1": scores[:, 0],
            "PC2": scores[:, 1],
            "class_label": df["class_label"].values,
            "tool_type": df["MillingToolType"].astype(str).values,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.1, 1]})
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="class_label",
        style="tool_type",
        palette=PALETTE,
        hue_order=CLASS_ORDER,
        s=22,
        alpha=0.7,
        ax=axes[0],
    )
    axes[0].set_title(
        f"PCA of 120 sensor statistics (PC1 {pca.explained_variance_ratio_[0]:.1%}, PC2 {pca.explained_variance_ratio_[1]:.1%})"
    )
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend(title="", loc="best", frameon=False)

    top = uni.head(18)["feature"].tolist()
    corr = df[top].astype(float).corr()
    labels = [short_feature(f) for f in top]
    sns.heatmap(
        corr,
        ax=axes[1],
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"shrink": 0.7, "label": "Pearson r"},
        xticklabels=labels,
        yticklabels=labels,
    )
    axes[1].set_title("Strong multicollinearity among top features")
    axes[1].tick_params(axis="x", rotation=90, labelsize=6)
    axes[1].tick_params(axis="y", rotation=0, labelsize=6)
    fig.suptitle("Sensor patterns separate classes, but tool signatures are also visible", fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig04_pca_correlation.png")


def fig_wear_confound(df: pd.DataFrame, uni: pd.DataFrame) -> dict:
    feat = uni.iloc[0]["feature"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    plot_df = df[["CycleToFailureNormalized", feat, "class_label"]].dropna().copy()
    sns.scatterplot(
        data=plot_df,
        x="CycleToFailureNormalized",
        y=feat,
        hue="class_label",
        hue_order=CLASS_ORDER,
        palette=PALETTE,
        s=18,
        alpha=0.45,
        ax=axes[0],
    )
    bins = np.linspace(0, 1, 11)
    plot_df["bin"] = pd.cut(plot_df["CycleToFailureNormalized"], bins, include_lowest=True)
    trend = plot_df.groupby(["bin", "class_label"], observed=True)[feat].mean().reset_index()
    trend["bin_mid"] = trend["bin"].apply(lambda interval: interval.mid)
    for label in CLASS_ORDER:
        sub = trend[trend["class_label"] == label]
        axes[0].plot(sub["bin_mid"], sub[feat], color=PALETTE[label], linewidth=2.2, marker="o", markersize=4)
    axes[0].set_xlabel("Remaining tool life (normalized; 0 = failure)")
    axes[0].set_ylabel(short_feature(feat))
    axes[0].set_title(f"{short_feature(feat)} vs wear stage")
    axes[0].legend(title="", frameon=False, loc="best")

    sns.boxplot(
        data=df,
        x="class_label",
        y="CycleToFailure",
        order=CLASS_ORDER,
        hue="class_label",
        palette=PALETTE,
        width=0.5,
        legend=False,
        showfliers=False,
        ax=axes[1],
    )
    sns.stripplot(
        data=df,
        x="class_label",
        y="CycleToFailure",
        order=CLASS_ORDER,
        color="#333333",
        size=1.5,
        alpha=0.25,
        ax=axes[1],
    )
    axes[1].set_title("Tool life differs by class")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Cycles to failure")
    axes[1].tick_params(axis="x", rotation=10)
    fig.suptitle("Wear is a second confound: dangerous tools also fail sooner", fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig05_wear_confound.png")
    return {
        "feature": feat,
        "cycle_to_failure_by_class": df.groupby("class_label")["CycleToFailure"].agg(["mean", "median", "min", "max"]).round(2).to_dict(),
    }


def fig_univariate_auc(uni: pd.DataFrame) -> None:
    top = uni.head(18).iloc[::-1]
    colors = [FAMILY_COLORS.get(f, "#777777") for f in top["family"]]
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    ax.barh([short_feature(f) for f in top["feature"]], top["separation"], color=colors, alpha=0.9)
    ax.axvline(0.5, color="#444444", linestyle="--", linewidth=1)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Univariate separation from chance: max(AUC, 1 - AUC)")
    ax.set_title("Sensor statistics with strongest class separation")
    for y, (auc, sep, direction) in enumerate(zip(top["auc"], top["separation"], top["direction"])):
        tag = "high" if "higher" in direction else "low"
        ax.text(sep + 0.004, y, f"AUC {auc:.3f} ({tag} in dangerous)", va="center", fontsize=7)
    for label, family in zip(ax.get_yticklabels(), top["family"]):
        label.set_color(FAMILY_COLORS.get(family, "#777777"))
    ax.text(0.985, 0.01, "blue = accelerometer, orange = current", transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#555555")
    fig.tight_layout()
    save(fig, "fig06_univariate_auc.png")


def write_insights(df: pd.DataFrame, uni: pd.DataFrame, levels: dict, wear: dict, tool: pd.DataFrame) -> None:
    top = uni.head(8)
    lines = []
    lines.append("# EDA and Engineering Insights\n")
    lines.append("Generated from the cleaned 968-cycle dataset. Figures are in `figures/`.\n")
    lines.append("## Class balance\n")
    lines.append(
        f"- Cycles: {levels['Cycles']['Optimal (RDOC 4.5 mm)']} optimal vs "
        f"{levels['Cycles']['Non-optimal (RDOC 8 mm)']} non-optimal "
        f"({levels['Cycles']['Non-optimal (RDOC 8 mm)'] / sum(levels['Cycles'].values()):.1%} dangerous).\n"
        f"- Experiments: {levels['Experiments']['Optimal (RDOC 4.5 mm)']} optimal vs "
        f"{levels['Experiments']['Non-optimal (RDOC 8 mm)']} non-optimal.\n"
        f"- Tools: {levels['Tools']['Optimal (RDOC 4.5 mm)']} optimal vs "
        f"{levels['Tools']['Non-optimal (RDOC 8 mm)']} non-optimal.\n"
    )
    lines.append("## Insights\n")
    lines.append(
        "1. **The class is a minority condition.** Only 21.9% of cycles, 8/24 experiments and 5/14 tools "
        "operate at the non-optimal RDOC of 8 mm. Accuracy alone would be misleading; recall for the dangerous "
        "class and PR-AUC are the appropriate metrics.\n"
    )
    lines.append(
        "2. **RDOC is perfectly confounded with tool/experiment identity.** Every tool and every experiment "
        "uses exactly one RDOC/ADOC combination, and tool type 2 is never used at RDOC = 8 mm. A random row "
        "split lets the model memorise tool signatures and inflates performance. All model selection must use "
        "tool-grouped validation with pooled out-of-fold predictions.\n"
    )
    lines.append(
        "3. **The sensor signal is physically plausible and strong.** Current standard deviation on the drive-X "
        "and spindle channels, and mean acceleration on the drive axes, are the strongest univariate features. "
        "A larger radial engagement increases cutting force, spindle/drive current and vibration, which is "
        "consistent with the observed direction. The best univariate AUCs are shown in `fig06_univariate_auc.png`.\n"
    )
    lines.append(
        f"4. **Wear is a second confound.** Dangerous tools fail sooner "
        f"(mean cycles to failure {wear['cycle_to_failure_by_class']['mean']['Non-optimal (RDOC 8 mm)']:.1f} vs "
        f"{wear['cycle_to_failure_by_class']['mean']['Optimal (RDOC 4.5 mm)']:.1f}). Sensor features also drift with "
        "remaining tool life, so part of the class separation may reflect wear stage rather than RDOC alone. "
        "This must be discussed as a limitation and, ideally, checked with early-life-only sensitivity analysis.\n"
    )
    lines.append(
        "5. **Hardness and tool type differ by class.** The dangerous group has lower mean hardness and is "
        "entirely composed of tool type 1, while tool type 2 appears only in the optimal group. These material/tool "
        "differences are plausible false-positive drivers; the tool 7 error case in the modelling section is an example.\n"
    )
    lines.append("## Top univariate features\n")
    lines.append("| Feature | AUC | Direction |")
    lines.append("|---|---:|---|")
    for _, row in top.iterrows():
        lines.append(f"| `{row['feature']}` | {row['auc']:.3f} | {row['direction']} |")
    lines.append("\n## Tool-level condition map\n")
    lines.append(tool[["ToolIndex", "n", "RDOC", "ADOC", "tool_type", "hardness", "rdoc_class"]].to_markdown(index=False))
    lines.append("\n## Figures\n")
    lines.append("- `fig01_class_balance.png`\n- `fig02_tool_condition_confounding.png`\n- `fig03_top_sensor_distributions.png`\n- `fig04_pca_correlation.png`\n- `fig05_wear_confound.png`\n- `fig06_univariate_auc.png`\n")
    (REPORTS / "EDA_INSIGHTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PROCESSED / "milling_clean.csv")
    levels = fig_class_balance(df)
    tool = fig_confounding(df)
    uni = compute_univariate(df)
    uni.to_csv(DATA_PROCESSED / "univariate_auc.csv", index=False)
    fig_sensor_distributions(df, uni)
    fig_pca_correlation(df, uni)
    wear = fig_wear_confound(df, uni)
    fig_univariate_auc(uni)
    write_insights(df, uni, levels, wear, tool)
    summary = {
        "levels": levels,
        "top_features": uni.head(10).to_dict(orient="records"),
        "wear": wear,
    }
    (DATA_PROCESSED / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2)[:4000])
    print("Figures written to", FIGURES)


if __name__ == "__main__":
    main()
