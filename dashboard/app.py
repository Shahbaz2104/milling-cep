"""Streamlit dashboard for the ME-446 milling-condition CEP."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_PROCESSED, FIGURES, MODELS, SENSOR_COLUMNS
from src.modeling import MODEL_LABELS, metric_dict

st.set_page_config(
    page_title="Milling Condition Safety Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "Optimal (RDOC 4.5 mm)": "#2E86AB",
    "Non-optimal (RDOC 8 mm)": "#D1495B",
}
CLASS_ORDER = ["Optimal (RDOC 4.5 mm)", "Non-optimal (RDOC 8 mm)"]

CUSTOM_CSS = """
<style>
.block-container {padding-top: 1.8rem; padding-bottom: 2rem; max-width: 1500px;}
h1, h2, h3 {letter-spacing: -0.02em;}
div[data-testid="stMetric"] {
    background: #F7F9FC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 12px 14px;
}
div[data-testid="stMetricLabel"] {font-size: 0.78rem;}
.small-note {color: #5B6573; font-size: 0.86rem; line-height: 1.45;}
.callout {
    background: #F4F7FB;
    border-left: 4px solid #2E86AB;
    padding: 12px 16px;
    border-radius: 6px;
    margin: 8px 0 14px 0;
    font-size: 0.93rem;
}
.warn {
    background: #FFF7ED;
    border-left: 4px solid #C76D2E;
    padding: 12px 16px;
    border-radius: 6px;
    margin: 8px 0 14px 0;
    font-size: 0.93rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data
def load_predictions() -> pd.DataFrame:
    return pd.read_csv(MODELS / "predictions_best.csv")


@st.cache_data
def load_clean() -> pd.DataFrame:
    return pd.read_csv(DATA_PROCESSED / "milling_clean.csv")


@st.cache_data
def load_sensor_long() -> pd.DataFrame:
    return pd.read_csv(DATA_PROCESSED / "sensor_long.csv")


@st.cache_data
def load_metrics() -> dict:
    return json.loads((MODELS / "model_metrics.json").read_text(encoding="utf-8"))


@st.cache_data
def load_comparison() -> pd.DataFrame:
    return pd.read_csv(MODELS / "model_comparison_pooled.csv")


@st.cache_data
def load_importance() -> pd.DataFrame:
    return pd.read_csv(MODELS / "feature_importance.csv")


@st.cache_data
def load_cleaning_log() -> str:
    return (DATA_PROCESSED / "CLEANING_LOG.md").read_text(encoding="utf-8")


@st.cache_resource
def load_bundle() -> dict:
    return joblib.load(MODELS / "final_model.joblib")


REQUIRED = [
    MODELS / "predictions_best.csv",
    MODELS / "model_metrics.json",
    MODELS / "model_comparison_pooled.csv",
    MODELS / "feature_importance.csv",
    MODELS / "final_model.joblib",
    DATA_PROCESSED / "milling_clean.csv",
    DATA_PROCESSED / "sensor_long.csv",
]
missing = [str(p.relative_to(ROOT)) for p in REQUIRED if not p.exists()]
if missing:
    st.error("Missing required artifacts: " + ", ".join(missing))
    st.info("Run `.venv/bin/python -m src.data`, `scripts/02_eda.py`, and `scripts/03_modeling.py` first.")
    st.stop()

pred = load_predictions()
clean = load_clean()
sensor_long = load_sensor_long()
metrics = load_metrics()
comparison = load_comparison()
importance = load_importance()
bundle = load_bundle()
threshold_default = float(bundle["threshold"])
selected_model = bundle["model_name"]
selected_label = MODEL_LABELS.get(selected_model, selected_model)

if "class_label" not in pred.columns:
    pred["class_label"] = np.where(pred["RDOC_class"] == 1, CLASS_ORDER[1], CLASS_ORDER[0])

with st.sidebar:
    st.title("Milling CEP")
    st.caption("ME-446 AI & IoT · NED University")
    st.divider()
    st.subheader("Selected model")
    st.markdown(f"**{selected_label}**")
    st.caption(f"Trained on {metrics['n_cycles']} cycles · {metrics['n_features']} sensor features")
    threshold = st.slider(
        "Decision threshold",
        min_value=0.05,
        max_value=0.95,
        value=round(threshold_default, 2),
        step=0.01,
        help="Lower thresholds increase dangerous-condition recall but also false alarms.",
    )
    st.caption(f"Default threshold selected on grouped out-of-fold predictions: {threshold_default:.3f}")
    st.divider()
    st.subheader("Dataset")
    st.markdown(
        f"- **Cycles:** {len(pred):,}\n"
        f"- **Experiments:** {pred['ExperimentID'].nunique()}\n"
        f"- **Tools:** {pred['ToolIndex'].nunique()}\n"
        f"- **Dangerous:** {(pred['RDOC_class'] == 1).sum()} ({(pred['RDOC_class'] == 1).mean():.1%})"
    )
    st.divider()
    st.caption(
        "Data: Piecuch & Żabiński, *Scientific Data* 12, 650 (2025). "
        "DOI 10.1038/s41597-025-04923-y. CC BY 4.0."
    )
    with st.expander("How to use this dashboard"):
        st.write(
            "Use the tabs to inspect class balance, explore cycles, check per-tool generalisation, "
            "score an existing or uploaded cycle, and review validation evidence. The threshold slider "
            "applies to every probability view."
        )

st.title("Milling Condition Safety Monitor")
st.markdown(
    '<div class="small-note">Detecting non-optimal milling conditions (RDOC = 8 mm) from '
    "accelerometer and spindle/drive current statistics. The classifier is evaluated with "
    "tool-grouped validation, not random row splits.</div>",
    unsafe_allow_html=True,
)

# Recompute the operating point at the slider threshold.
y = pred["RDOC_class"].astype(int).to_numpy()
prob_primary = pred["prob_tool5_best"].to_numpy()
prob_loo = pred["prob_tool_loo_best"].to_numpy()
pred_at_threshold = (prob_primary >= threshold).astype(int)
selected_metrics = metric_dict(y, prob_primary, threshold=threshold)
loo_metrics = metric_dict(y, prob_loo, threshold=threshold)

overview_tab, explorer_tab, tool_tab, predict_tab, performance_tab, data_tab = st.tabs(
    [
        "Overview",
        "Cycle explorer",
        "Tool generalisation",
        "Score a cycle",
        "Model performance",
        "Data & limitations",
    ]
)

with overview_tab:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Cycles", f"{len(pred):,}")
    c2.metric("Experiments", pred["ExperimentID"].nunique())
    c3.metric("Tools", pred["ToolIndex"].nunique())
    c4.metric("Dangerous share", f"{(y == 1).mean():.1%}")
    c5.metric("Recall at threshold", f"{selected_metrics['recall']:.1%}")

    st.markdown(
        '<div class="callout"><b>Engineering framing.</b> RDOC and ADOC are constant within every '
        "experiment and tool, so a random row split lets the model memorise tool identity. The primary "
        "estimate therefore holds out whole tool groups; random-split scores are shown only as a leakage "
        "control.</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.05, 0.95])
    with left:
        st.subheader("Class balance")
        counts = (
            pred["class_label"]
            .value_counts()
            .reindex(CLASS_ORDER)
            .rename_axis("class_label")
            .reset_index(name="cycles")
        )
        fig = px.bar(
            counts,
            x="class_label",
            y="cycles",
            color="class_label",
            color_discrete_map=PALETTE,
            text="cycles",
        )
        fig.update_layout(showlegend=False, height=330, margin=dict(l=10, r=10, t=10, b=10))
        fig.update_xaxes(title="")
        fig.update_yaxes(title="Cycles")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "The dangerous class is the minority (21.9%). Recall and PR-AUC are therefore more "
            "informative than raw accuracy."
        )
    with right:
        st.subheader("Operating point")
        m1, m2 = st.columns(2)
        m1.metric("Recall", f"{selected_metrics['recall']:.1%}")
        m2.metric("Precision", f"{selected_metrics['precision']:.1%}")
        m3, m4 = st.columns(2)
        m3.metric("F1", f"{selected_metrics['f1']:.3f}")
        m4.metric("MCC", f"{selected_metrics['mcc']:.3f}")
        m5, m6 = st.columns(2)
        m5.metric("Balanced accuracy", f"{selected_metrics['balanced_accuracy']:.1%}")
        m6.metric("Brier score", f"{selected_metrics['brier']:.3f}")
        cm = np.array(selected_metrics["confusion_matrix"])
        st.caption(
            f"Confusion at threshold {threshold:.2f}: {cm[0, 0]} true optimal, {cm[0, 1]} false alarms, "
            f"{cm[1, 0]} missed dangerous cycles, {cm[1, 1]} true dangerous."
        )

    st.divider()
    st.subheader("Why grouped validation matters")
    img1, img2 = st.columns(2)
    with img1:
        st.image(
            str(FIGURES / "fig02_tool_condition_confounding.png"),
            caption="Every tool and experiment carries a single RDOC/ADOC condition.",
            use_container_width=True,
        )
    with img2:
        st.image(
            str(FIGURES / "fig06_univariate_auc.png"),
            caption="Univariate separation of the strongest sensor features.",
            use_container_width=True,
        )

with explorer_tab:
    st.subheader("Cycle-level exploration")
    f1, f2, f3, f4 = st.columns([1.0, 1.0, 1.1, 1.1])
    with f1:
        experiments = st.multiselect(
            "Experiment",
            sorted(pred["ExperimentID"].unique()),
            default=None,
            placeholder="All experiments",
        )
    with f2:
        tools = st.multiselect(
            "Tool",
            sorted(pred["ToolIndex"].unique()),
            default=None,
            placeholder="All tools",
        )
    with f3:
        class_filter = st.radio(
            "Actual class",
            ["All", "Optimal", "Dangerous"],
            horizontal=True,
        )
    with f4:
        prob_source = st.radio(
            "Probability source",
            ["5-fold tool groups", "Leave-one-tool-out"],
            horizontal=True,
            help="The first is the primary estimate; the second shows each tool scored by a model that never saw it.",
        )

    view = pred.copy()
    if experiments:
        view = view[view["ExperimentID"].isin(experiments)]
    if tools:
        view = view[view["ToolIndex"].isin(tools)]
    if class_filter == "Optimal":
        view = view[view["RDOC_class"] == 0]
    elif class_filter == "Dangerous":
        view = view[view["RDOC_class"] == 1]

    prob_col = "prob_tool5_best" if prob_source == "5-fold tool groups" else "prob_tool_loo_best"
    view = view.assign(pred_at_threshold=(view[prob_col] >= threshold).astype(int))
    view["tool_label"] = "Tool " + view["ToolIndex"].astype(int).astype(str)

    fig = px.scatter(
        view,
        x="CycleToFailureNormalized",
        y=prob_col,
        color="class_label",
        color_discrete_map=PALETTE,
        hover_data=["FileName", "ExperimentID", "ToolIndex", "CycleToFailure"],
        labels={
            "CycleToFailureNormalized": "Normalised remaining cycles (1 = first cycle, 0 = failure)",
            prob_col: "P(dangerous)",
            "class_label": "Actual class",
        },
        opacity=0.7,
    )
    fig.add_hline(y=threshold, line_dash="dash", line_color="#333333")
    fig.update_yaxes(range=[-0.02, 1.02])
    fig.update_layout(height=420, legend_title_text="", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"{len(view):,} cycles shown. The dashed line is the current threshold ({threshold:.2f}). "
        "Probability is out-of-fold."
    )

    st.subheader("Filtered cycles")
    table_cols = [
        "FileName", "ExperimentID", "LayerID", "ToolIndex", "RDOC_mm", "class_label",
        "CycleToFailure", "CycleToFailureNormalized", prob_col, "pred_at_threshold",
    ]
    st.dataframe(
        view[table_cols].sort_values(prob_col, ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            prob_col: st.column_config.ProgressColumn(
                "P(dangerous)", min_value=0.0, max_value=1.0, format="%.3f"
            ),
            "pred_at_threshold": st.column_config.CheckboxColumn("Predicted dangerous"),
        },
    )
    st.download_button(
        "Download filtered cycles (CSV)",
        view[table_cols].to_csv(index=False).encode("utf-8"),
        file_name="filtered_cycles.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Inspect one cycle")
    cycle = st.selectbox("Cycle", view["FileName"].tolist())
    row = pred[pred["FileName"] == cycle].iloc[0]
    left, right = st.columns([0.8, 1.2])
    with left:
        p = float(row[prob_col])
        st.metric("P(dangerous)", f"{p:.1%}")
        st.metric("Decision", "Dangerous" if p >= threshold else "Optimal")
        st.write(
            f"**Actual:** {row['class_label']}  \n"
            f"**Experiment / tool:** {row['ExperimentID']} / tool {int(row['ToolIndex'])}  \n"
            f"**Remaining cycles:** {int(row['CycleToFailure'])}  \n"
            f"**RDOC / ADOC:** {row['RDOC_mm']:.1f} / {row['ADOC_mm']:.1f} mm"
        )
    with right:
        top_features = importance.head(12)["feature"].tolist()
        cycle_sensors = sensor_long[
            (sensor_long["FileName"] == cycle) & (sensor_long["feature"].isin(top_features))
        ][["feature", "value", "family"]].copy()
        baseline = (
            sensor_long[sensor_long["RDOC_class"] == 0]
            .groupby("feature")["value"]
            .agg(["mean", "std"])
            .reset_index()
        )
        cycle_sensors = cycle_sensors.merge(baseline, on="feature", how="left")
        cycle_sensors["z_vs_optimal"] = (
            (cycle_sensors["value"] - cycle_sensors["mean"]) / cycle_sensors["std"].replace(0, np.nan)
        )
        cycle_sensors = cycle_sensors.sort_values("z_vs_optimal", key=lambda s: s.abs(), ascending=False)
        fig = px.bar(
            cycle_sensors.head(12),
            x="z_vs_optimal",
            y="feature",
            orientation="h",
            color="family",
            color_discrete_map={"Accelerometer": "#2E86AB", "Current": "#C76D2E"},
            labels={"z_vs_optimal": "Z-score vs optimal-class mean", "feature": ""},
        )
        fig.add_vline(x=0, line_color="#777777")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Top 12 globally important features for this cycle, expressed relative to optimal cycles.")

with tool_tab:
    st.subheader("Per-tool generalisation")
    st.markdown(
        '<div class="warn"><b>Known failure mode.</b> Optimal tool 7 is often scored above the threshold '
        "(high false-alarm rate), while the lower tails of dangerous tools 5 and 6 fall below it. "
        "This is the main residual risk under unseen-tool deployment.</div>",
        unsafe_allow_html=True,
    )

    tool_order = (
        pred.assign(tool_label="Tool " + pred["ToolIndex"].astype(int).astype(str))
        .groupby("tool_label")["RDOC_class"]
        .max()
        .sort_values(ascending=False)
        .index.tolist()
    )
    plot_df = pred.assign(
        tool_label="Tool " + pred["ToolIndex"].astype(int).astype(str),
        pred_loo=(pred["prob_tool_loo_best"] >= threshold).astype(int),
    )
    fig = px.box(
        plot_df,
        x="tool_label",
        y="prob_tool_loo_best",
        color="class_label",
        color_discrete_map=PALETTE,
        category_orders={"tool_label": tool_order, "class_label": CLASS_ORDER},
        points="all",
        hover_data=["FileName", "ExperimentID", "CycleToFailure"],
        labels={"tool_label": "", "prob_tool_loo_best": "P(dangerous)", "class_label": "Actual class"},
    )
    fig.add_hline(y=threshold, line_dash="dash", line_color="#333333")
    fig.update_yaxes(range=[-0.02, 1.02])
    fig.update_layout(height=500, legend_title_text="", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Leave-one-tool-out predictions: each tool is scored by a model trained without that tool.")

    st.subheader("Tool-level summary")
    summary = (
        pred.assign(
            tool_label="Tool " + pred["ToolIndex"].astype(int).astype(str),
            pred_loo=(pred["prob_tool_loo_best"] >= threshold).astype(int),
        )
        .groupby(["ToolIndex", "class_label", "tool_label"], as_index=False)
        .agg(
            cycles=("FileName", "size"),
            mean_probability=("prob_tool_loo_best", "mean"),
            median_probability=("prob_tool_loo_best", "median"),
            flagged_fraction=("pred_loo", "mean"),
        )
        .sort_values(["class_label", "ToolIndex"])
    )
    summary["class_label"] = pd.Categorical(summary["class_label"], categories=CLASS_ORDER, ordered=True)
    summary = summary.sort_values(["class_label", "ToolIndex"]).drop(columns=["class_label"])
    summary = summary.rename(
        columns={
            "ToolIndex": "Tool",
            "cycles": "Cycles",
            "mean_probability": "Mean P(dangerous)",
            "median_probability": "Median P(dangerous)",
            "flagged_fraction": "Fraction flagged",
        }
    )
    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Mean P(dangerous)": st.column_config.NumberColumn(format="%.3f"),
            "Median P(dangerous)": st.column_config.NumberColumn(format="%.3f"),
            "Fraction flagged": st.column_config.ProgressColumn(
                min_value=0.0, max_value=1.0, format="%.2f"
            ),
        },
    )
    st.caption(
        "For dangerous tools, 'Fraction flagged' is recall; for optimal tools it is the false-positive rate. "
        "Tool 11 contributes only one cycle."
    )

with predict_tab:
    st.subheader("Score a new cycle")
    mode = st.radio("Input mode", ["Existing cycle", "Upload sensor CSV"], horizontal=True)
    if mode == "Existing cycle":
        left, right = st.columns([1.0, 1.0])
        with left:
            experiment = st.selectbox(
                "Experiment",
                sorted(pred["ExperimentID"].unique()),
                index=0,
            )
            cycles = pred[pred["ExperimentID"] == experiment]["FileName"].tolist()
            cycle = st.selectbox("Cycle", cycles)
            row = pred[pred["FileName"] == cycle].iloc[0]
            X_row = clean.loc[clean["FileName"] == cycle, SENSOR_COLUMNS]
            probability = float(bundle["model"].predict_proba(X_row)[0, 1])
            decision = "Dangerous condition" if probability >= threshold else "Optimal condition"
            st.metric("P(dangerous)", f"{probability:.1%}")
            st.metric("Decision", decision)
            st.write(
                f"**Actual label:** {row['class_label']}  \n"
                f"**Experiment / tool:** {row['ExperimentID']} / tool {int(row['ToolIndex'])}  \n"
                f"**Remaining cycles:** {int(row['CycleToFailure'])}"
            )
        with right:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number={"suffix": "%"},
                    title={"text": "P(non-optimal)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#D1495B" if probability >= threshold else "#2E86AB"},
                        "threshold": {
                            "line": {"color": "#333333", "width": 3},
                            "thickness": 0.8,
                            "value": threshold * 100,
                        },
                        "steps": [
                            {"range": [0, threshold * 100], "color": "#E8F1F8"},
                            {"range": [threshold * 100, 100], "color": "#FBE9EC"},
                        ],
                    },
                )
            )
            fig.update_layout(height=330, margin=dict(l=20, r=20, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("The black marker is the current decision threshold.")
    else:
        uploaded = st.file_uploader("Upload a CSV with the 120 sensor-statistic columns", type=["csv"])
        st.download_button(
            "Download CSV template (first 3 cycles)",
            clean[SENSOR_COLUMNS].head(3).to_csv(index=False).encode("utf-8"),
            file_name="milling_sensor_template.csv",
            mime="text/csv",
        )
        if uploaded is not None:
            try:
                uploaded_df = pd.read_csv(uploaded)
            except Exception as exc:
                st.error(f"Could not read CSV: {exc}")
            else:
                missing_cols = [c for c in SENSOR_COLUMNS if c not in uploaded_df.columns]
                if missing_cols:
                    st.error(f"Missing {len(missing_cols)} required sensor columns, e.g. {missing_cols[:3]}")
                else:
                    try:
                        probs = bundle["model"].predict_proba(uploaded_df[SENSOR_COLUMNS])[:, 1]
                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")
                    else:
                        out = uploaded_df.copy()
                        out["prob_dangerous"] = probs
                        out["pred_dangerous"] = (probs >= threshold).astype(int)
                        st.dataframe(
                            out[["prob_dangerous", "pred_dangerous"]].head(50),
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "prob_dangerous": st.column_config.ProgressColumn(
                                    "P(dangerous)", min_value=0.0, max_value=1.0, format="%.3f"
                                ),
                                "pred_dangerous": st.column_config.CheckboxColumn("Predicted dangerous"),
                            },
                        )
                        st.download_button(
                            "Download predictions (CSV)",
                            out.to_csv(index=False).encode("utf-8"),
                            file_name="milling_predictions.csv",
                            mime="text/csv",
                        )

with performance_tab:
    st.subheader("Validation evidence")
    st.markdown(
        f"Selected **{selected_label}** at threshold **{threshold:.2f}**. The primary estimate is "
        "pooled out-of-fold performance under 5-fold tool-group validation; leave-one-tool-out and "
        "leave-one-experiment-out are sensitivity analyses. Random 5-fold is a leakage control."
    )

    metric_table = pd.DataFrame(
        [
            {
                "Metric": "ROC-AUC",
                "5-fold tool groups": selected_metrics["roc_auc"],
                "Leave-one-tool-out": loo_metrics["roc_auc"],
            },
            {
                "Metric": "PR-AUC",
                "5-fold tool groups": selected_metrics["pr_auc"],
                "Leave-one-tool-out": loo_metrics["pr_auc"],
            },
            {
                "Metric": "Recall (dangerous)",
                "5-fold tool groups": selected_metrics["recall"],
                "Leave-one-tool-out": loo_metrics["recall"],
            },
            {
                "Metric": "Precision (dangerous)",
                "5-fold tool groups": selected_metrics["precision"],
                "Leave-one-tool-out": loo_metrics["precision"],
            },
            {
                "Metric": "F1 (dangerous)",
                "5-fold tool groups": selected_metrics["f1"],
                "Leave-one-tool-out": loo_metrics["f1"],
            },
            {
                "Metric": "MCC",
                "5-fold tool groups": selected_metrics["mcc"],
                "Leave-one-tool-out": loo_metrics["mcc"],
            },
            {
                "Metric": "Balanced accuracy",
                "5-fold tool groups": selected_metrics["balanced_accuracy"],
                "Leave-one-tool-out": loo_metrics["balanced_accuracy"],
            },
            {
                "Metric": "Brier score",
                "5-fold tool groups": selected_metrics["brier"],
                "Leave-one-tool-out": loo_metrics["brier"],
            },
        ]
    )
    st.dataframe(
        metric_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "5-fold tool groups": st.column_config.NumberColumn(format="%.3f"),
            "Leave-one-tool-out": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    left, right = st.columns(2)
    with left:
        st.subheader("ROC curve")
        fpr, tpr, _ = roc_curve(y, prob_primary)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Random forest", line=dict(color="#6A994E", width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(color="#999999", dash="dash")))
        fig.update_xaxes(title="False positive rate")
        fig.update_yaxes(title="True positive rate")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), legend=dict(x=0.55, y=0.05))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Precision-recall curve")
        precision, recall, _ = precision_recall_curve(y, prob_primary)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="Random forest", line=dict(color="#6A994E", width=3)))
        fig.add_hline(y=y.mean(), line_dash="dash", line_color="#999999")
        fig.update_xaxes(title="Recall (dangerous)")
        fig.update_yaxes(title="Precision (dangerous)")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), legend=dict(x=0.05, y=0.05))
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        cm = np.array(selected_metrics["confusion_matrix"])
        fig = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            x=["Predicted optimal", "Predicted dangerous"],
            y=["Actual optimal", "Actual dangerous"],
            aspect="auto",
        )
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Calibration")
        frac_pos, mean_pred = calibration_curve(y, prob_primary, n_bins=10, strategy="quantile")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line=dict(color="#999999", dash="dash")))
        fig.add_trace(go.Scatter(x=mean_pred, y=frac_pos, mode="lines+markers", name="Random forest", line=dict(color="#6A994E", width=2)))
        fig.update_xaxes(title="Mean predicted probability")
        fig.update_yaxes(title="Observed dangerous fraction")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), legend=dict(x=0.55, y=0.05))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model comparison")
    comparison_view = comparison.copy()
    comparison_view = comparison_view.rename(
        columns={
            "model": "Model",
            "scheme": "Validation scheme",
            "roc_auc": "ROC-AUC",
            "pr_auc": "PR-AUC",
            "balanced_accuracy": "Balanced accuracy",
            "recall": "Recall",
            "precision": "Precision",
            "f1": "F1",
            "specificity": "Specificity",
            "mcc": "MCC",
            "brier": "Brier",
            "accuracy": "Accuracy",
        }
    )
    scheme = st.selectbox(
        "Validation scheme",
        ["tool_5fold", "tool_loo", "experiment_loo", "random_5fold"],
        format_func={
            "tool_5fold": "5-fold tool groups (primary)",
            "tool_loo": "Leave-one-tool-out (sensitivity)",
            "experiment_loo": "Leave-one-experiment-out (sensitivity)",
            "random_5fold": "Random 5-fold (leaky control)",
        }.get,
    )
    st.dataframe(
        comparison_view[comparison_view["Validation scheme"] == scheme].drop(columns=["Validation scheme"]),
        use_container_width=True,
        hide_index=True,
        column_config={
            c: st.column_config.NumberColumn(format="%.3f")
            for c in ["ROC-AUC", "PR-AUC", "Balanced accuracy", "Recall", "Precision", "F1", "Specificity", "MCC", "Brier", "Accuracy"]
        },
    )
    if scheme == "random_5fold":
        st.warning(
            "Random row splits leak tool/condition identity: the same tool appears in training and test. "
            "Use the grouped results for the honest estimate."
        )
    else:
        st.caption(
            "Random forest was selected on the operating-point trade-off (recall, precision, MCC, "
            "balanced accuracy, Brier). Hist gradient boosting had the highest threshold-independent PR-AUC, "
            "but random forest gave the better safety-oriented operating point."
        )

    st.subheader("Feature importance")
    top = importance.head(15).iloc[::-1]
    fig = px.bar(
        top,
        x="importance_mean",
        y="feature",
        orientation="h",
        error_x="importance_std",
        color=top["feature"].str.startswith("Accelerometer").map({True: "Accelerometer", False: "Current"}),
        color_discrete_map={"Accelerometer": "#2E86AB", "Current": "#C76D2E"},
        labels={"importance_mean": "Permutation importance (drop in ROC-AUC)", "feature": "", "color": ""},
    )
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=10), legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Permutation importance is computed on the full data with the final forest. Correlated sensor "
        "features can share or mask importance; use this as a global sanity check, not a causal claim."
    )

    with st.expander("Leakage-control numbers"):
        leak = comparison[comparison["scheme"].isin(["tool_5fold", "random_5fold"])]
        st.dataframe(leak, use_container_width=True, hide_index=True)
        st.write(
            "Random-split ROC-AUC is near 1.0 because tool and condition are aliased. The tool-grouped "
            "estimate is lower and is the number quoted in the report."
        )

with data_tab:
    st.subheader("Data source and licence")
    st.markdown(
        "**Dataset:** Piecuch, M. & Żabiński, T. *A dataset for milling process monitoring* "
        "(Scientific Data 12, 650, 2025).  \n"
        "**DOI:** [10.1038/s41597-025-04923-y](https://doi.org/10.1038/s41597-025-04923-y)  \n"
        "**Data DOI:** [10.6084/m9.figshare.28589216](https://doi.org/10.6084/m9.figshare.28589216)  \n"
        "**Licence:** CC BY 4.0"
    )
    st.divider()
    st.subheader("Cleaning summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cycles", f"{len(clean):,}")
    c2.metric("Sensor features", len(SENSOR_COLUMNS))
    c3.metric("Missing values", 0)
    c4.metric("Censored cycles", int((clean["CycleToFailure"] == 0).sum()))
    st.markdown(
        "- Raw CSV had a BOM, semicolon delimiter, a redundant first row, and comma decimals in metadata; "
        "all were parsed explicitly.\n"
        "- `ToolRotation`, `FeedRate`, and `ToolDiameter` are constant and were excluded.\n"
        "- Metadata, IDs, wear labels, and failure counters were excluded from the classifier.\n"
        "- 52 cycles were flagged by an IQR rule on ≥5 features but retained; they are not automatically errors."
    )
    with st.expander("View cleaning log"):
        st.markdown(load_cleaning_log())

    st.divider()
    st.subheader("Limitations and honest caveats")
    st.markdown(
        "- **Confounded design:** RDOC and ADOC are constant within each tool/experiment, so the model can "
        "learn tool identity. Grouped CV mitigates but cannot fully remove this.\n"
        "- **Small tool count:** 14 tools, 5 dangerous; leave-one-tool-out is the relevant stress test, and "
        "tool 7 is a persistent false-positive source.\n"
        "- **Holdout scope:** P002 and P108 are unseen experiments, but their tools (2 and 102) also appear "
        "in other experiments. The leave-one-tool-out analysis is the true unseen-tool check.\n"
        "- **Raw signals unavailable:** the 25 GB raw-data archive was not available locally; models use the "
        "published 120 statistical features.\n"
        "- **Threshold is a policy choice:** 0.44 favours recall; a higher threshold reduces false alarms but "
        "misses more dangerous cycles.\n"
        "- **Tableau:** a Tableau-ready extract and build guide are provided separately; Streamlit is the "
        "primary interactive dashboard."
    )

    st.divider()
    st.subheader("Reproduce")
    st.code(
        """# from the project root
.venv/bin/python -m src.data
.venv/bin/python scripts/02_eda.py
.venv/bin/python scripts/03_modeling.py
.venv/bin/python scripts/05_select_rf.py
streamlit run dashboard/app.py""",
        language="bash",
    )
    st.caption("Model artifacts are in models/. Figures are in figures/. Processed data are in data/processed/.")
