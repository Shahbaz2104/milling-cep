"""Data loading, cleaning, auditing and fold generation for the milling CEP."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CONSTANT_METADATA,
    CSV_METADATA_COLUMNS,
    DATA_PROCESSED,
    DATA_RAW,
    LEAKAGE_OR_ID_COLUMNS,
    RAW_CSV,
    RAW_XLSX,
    SENSOR_COLUMNS,
)

PROCESSED_FILES = {
    "clean": DATA_PROCESSED / "milling_clean.csv",
    "features": DATA_PROCESSED / "features.csv",
    "metadata": DATA_PROCESSED / "metadata.csv",
    "dictionary": DATA_PROCESSED / "data_dictionary.csv",
    "folds": DATA_PROCESSED / "folds.csv",
    "sensor_long": DATA_PROCESSED / "sensor_long.csv",
    "audit": DATA_PROCESSED / "audit.json",
    "cleaning_log": DATA_PROCESSED / "CLEANING_LOG.md",
}

XLSX_RENAME = {
    "ExperimentIndex": "ExperimentID_xlsx",
    "SampleIndex": "SampleIndex_xlsx",
    "ToolIndex": "ToolIndex_xlsx",
    "MillingToolType": "MillingToolType_xlsx",
    "ADOC [mm]": "ADOC_mm_xlsx",
    "RDOC [mm]": "RDOC_mm_xlsx",
    "HardnessMean [HRC]": "Hardness_HRC_xlsx",
    "CycleToFailure": "CycleToFailure_xlsx",
    "CycleToFailureNormalized": "CycleToFailureNormalized_xlsx",
    "ToolHolderLength [mm]": "ToolHolderLength_mm_xlsx",
    "ToolRotation [rpm]": "ToolRotation_rpm",
    "FeedRate [mm/min]": "FeedRate_mm_min",
    "ToolDiameter [mm]": "ToolDiameter_mm",
}


def _to_numeric_comma(series: pd.Series) -> pd.Series:
    """Convert a possibly comma-decimal text column to numeric."""
    return pd.to_numeric(
        series.astype(str).str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    )


def load_raw_csv() -> pd.DataFrame:
    """Load the authors' processed feature CSV.

    The first physical row is the Excel-export artefact ``Column1...Column131``;
    row 2 contains the real variable names.  Metadata use comma decimals while
    the 120 sensor columns use dots, so the file is read as text first and the
    affected metadata columns are converted explicitly.
    """
    df = pd.read_csv(RAW_CSV, sep=";", header=1, encoding="utf-8-sig")
    if len(df.columns) != 131:
        raise ValueError(f"Expected 131 columns, found {len(df.columns)}")
    missing = [c for c in CSV_METADATA_COLUMNS + SENSOR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df


def load_raw_metadata() -> pd.DataFrame:
    """Load the clean metadata workbook."""
    df = pd.read_excel(RAW_XLSX, sheet_name=0)
    return df.rename(columns=XLSX_RENAME)


def verify_sources(csv_df: pd.DataFrame, meta_df: pd.DataFrame) -> dict:
    """Cross-check the CSV metadata against the XLSX workbook."""
    checks = {}
    pairs = {
        "SampleIndex": "SampleIndex_xlsx",
        "ToolIndex": "ToolIndex_xlsx",
        "MillingToolType": "MillingToolType_xlsx",
        "ADOC_mm": "ADOC_mm_xlsx",
        "RDOC_mm": "RDOC_mm_xlsx",
        "Hardness_HRC": "Hardness_HRC_xlsx",
        "CycleToFailure": "CycleToFailure_xlsx",
        "CycleToFailureNormalized": "CycleToFailureNormalized_xlsx",
        "ToolHolderLength_mm": "ToolHolderLength_mm_xlsx",
    }
    for left, right in pairs.items():
        a = csv_df[left]
        b = meta_df[right]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            match = np.isclose(a.astype(float), b.astype(float), equal_nan=True)
        else:
            match = a.astype(str).str.strip().values == b.astype(str).str.strip().values
        checks[f"{left}__vs__{right}"] = bool(np.all(match))
        if not np.all(match):
            bad = int((~match).sum()) if hasattr(match, "sum") else 1
            checks[f"{left}__mismatches"] = bad
    checks["row_order_matches"] = bool(
        (csv_df["FileName"].astype(str).values == meta_df["ExperimentID_xlsx"].astype(str).values).all()
    )
    checks["row_counts"] = {
        "csv": int(len(csv_df)),
        "xlsx": int(len(meta_df)),
    }
    return checks


def build_clean_dataset() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (clean cycle-level dataframe, xlsx metadata, audit dict)."""
    raw = load_raw_csv()
    meta = load_raw_metadata()

    df = raw.copy()

    # Rename CSV metadata to explicit, unit-bearing names.
    df = df.rename(
        columns={
            "TollIndex": "ToolIndex",
            "ADOC": "ADOC_mm",
            "RDOC": "RDOC_mm",
            "HardnessMean": "Hardness_HRC",
            "ToolHolderLength": "ToolHolderLength_mm",
        }
    )

    # Numeric conversions with mixed decimal conventions.
    for col in ["ADOC_mm", "RDOC_mm", "Hardness_HRC", "CycleToFailureNormalized"]:
        df[col] = _to_numeric_comma(df[col])
    for col in [
        "NumberOfCycle",
        "SampleIndex",
        "ToolIndex",
        "MillingToolType",
        "ToolHolderLength_mm",
        "CycleToFailure",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Merge the three constant process parameters from the workbook.
    const = meta[["ExperimentID_xlsx", "ToolRotation_rpm", "FeedRate_mm_min", "ToolDiameter_mm"]].copy()
    df = df.merge(
        const.rename(columns={"ExperimentID_xlsx": "FileName"}),
        on="FileName",
        how="left",
        validate="one_to_one",
    )
    source_checks = verify_sources(df, meta)

    # Parse the experiment / layer / cycle identifiers.
    parsed = df["FileName"].str.extract(r"^P(?P<P_num>\d+)_F(?P<F_num>\d+)_C(?P<C_num>\d+)$")
    if parsed.isna().any().any():
        bad = df.loc[parsed.isna().any(axis=1), "FileName"].tolist()
        raise ValueError(f"Could not parse FileName values: {bad}")
    df["P"] = "P" + parsed["P_num"]
    df["F"] = parsed["F_num"].astype(int)
    df["C"] = parsed["C_num"].astype(int)
    df["ExperimentID"] = df["P"]
    df["LayerID"] = df["P"] + "_F" + parsed["F_num"].str.zfill(2)
    df["ToolID"] = "T" + df["ToolIndex"].astype(str)

    # Target: dangerous/non-optimal RDOC (8 mm) vs optimal (4.5 mm).
    df["RDOC_class"] = (df["RDOC_mm"] == 8.0).astype(int)
    df["class_label"] = np.where(
        df["RDOC_class"] == 1,
        "Non-optimal (RDOC 8 mm)",
        "Optimal (RDOC 4.5 mm)",
    )

    # Outlier flag: feature-wise IQR rule (k=5), count of extreme features.
    # Vibration/current statistics are naturally heavy-tailed, so the flag is
    # advisory only and no rows are removed.
    sensor = df[SENSOR_COLUMNS].astype(float)
    q1 = sensor.quantile(0.25)
    q3 = sensor.quantile(0.75)
    iqr = (q3 - q1).replace(0, np.nan)
    iqr = iqr.fillna(sensor.std(axis=0)).fillna(1.0)
    lower = q1 - 5.0 * iqr
    upper = q3 + 5.0 * iqr
    flags = ((sensor < lower) | (sensor > upper)).sum(axis=1)
    df["n_outlier_features"] = flags.astype(int)
    df["outlier_flag"] = (flags >= 5).astype(int)

    # Preserve the original row order for reproducibility.
    df = df.reset_index(drop=True)

    audit = make_audit(df, raw, meta, source_checks)
    return df, meta, audit


def make_audit(
    df: pd.DataFrame,
    raw: pd.DataFrame,
    meta: pd.DataFrame,
    source_checks: dict,
) -> dict:
    sensor = df[SENSOR_COLUMNS].astype(float)
    class_counts = df["RDOC_class"].value_counts().sort_index().to_dict()
    audit = {
        "dataset": {
            "raw_file": RAW_CSV.name,
            "metadata_file": RAW_XLSX.name,
            "n_rows": int(len(df)),
            "n_columns": int(len(df.columns)),
            "n_sensor_features": len(SENSOR_COLUMNS),
            "n_metadata_source_columns": len(CSV_METADATA_COLUMNS),
            "n_experiments": int(df["ExperimentID"].nunique()),
            "n_layers": int(df["LayerID"].nunique()),
            "n_tools": int(df["ToolIndex"].nunique()),
            "experiments": sorted(df["ExperimentID"].unique().tolist()),
            "tools": sorted(df["ToolIndex"].dropna().astype(int).unique().tolist()),
        },
        "source_checks": source_checks,
        "missing_values_total": int(df.isna().sum().sum()),
        "missing_values_by_column": {
            k: int(v) for k, v in df.isna().sum().items() if v > 0
        },
        "duplicate_rows_full": int(df.duplicated().sum()),
        "duplicate_sensor_rows": int(sensor.duplicated().sum()),
        "infinite_sensor_values": int(np.isinf(sensor.to_numpy()).sum()),
        "zero_variance_sensor_features": sensor.columns[sensor.std(axis=0) == 0].tolist(),
        "class_counts_cycles": {int(k): int(v) for k, v in class_counts.items()},
        "class_ratio_cycles": {
            str(int(k)): round(float(v) / len(df), 4) for k, v in class_counts.items()
        },
        "class_counts_experiments": df.groupby("ExperimentID")["RDOC_class"].max().value_counts().sort_index().to_dict(),
        "class_counts_tools": df.groupby("ToolIndex")["RDOC_class"].max().value_counts().sort_index().to_dict(),
        "experiments_with_both_classes": int(
            (df.groupby("ExperimentID")["RDOC_class"].nunique() > 1).sum()
        ),
        "tools_with_both_classes": int(
            (df.groupby("ToolIndex")["RDOC_class"].nunique() > 1).sum()
        ),
        "tool_type_2_dangerous_cycles": int(
            len(df[(df["MillingToolType"] == 2) & (df["RDOC_class"] == 1)])
        ),
        "constant_metadata": {
            k: bool(df[k].nunique(dropna=False) == 1) for k in CONSTANT_METADATA
        },
        "constant_metadata_values": {
            k: (None if df[k].dropna().empty else float(df[k].dropna().iloc[0]))
            for k in CONSTANT_METADATA
        },
        "metadata_ranges": {
            "ADOC_mm": [float(df["ADOC_mm"].min()), float(df["ADOC_mm"].max())],
            "RDOC_mm": [float(df["RDOC_mm"].min()), float(df["RDOC_mm"].max())],
            "Hardness_HRC": [float(df["Hardness_HRC"].min()), float(df["Hardness_HRC"].max())],
            "ToolHolderLength_mm": [
                float(df["ToolHolderLength_mm"].min()),
                float(df["ToolHolderLength_mm"].max()),
            ],
            "CycleToFailure": [
                float(df["CycleToFailure"].min()),
                float(df["CycleToFailure"].max()),
            ],
            "NumberOfCycle": [
                float(df["NumberOfCycle"].min()),
                float(df["NumberOfCycle"].max()),
            ],
        },
        "adoc_rdoc_crosstab": {
            f"ADOC_{int(adoc)}_RDOC_{rdoc}": int(n)
            for (adoc, rdoc), n in df.groupby(["ADOC_mm", "RDOC_mm"]).size().items()
        },
        "censored_cycles_cycle_to_failure_zero": int((df["CycleToFailure"] == 0).sum()),
        "single_cycle_experiments": df.groupby("ExperimentID").size().loc[lambda s: s == 1].index.tolist(),
        "accelerometer_saturation_rows": int((sensor.filter(like="Accelerometer") <= -32768).any(axis=1).sum()),
        "outlier_rows_any": int((df["n_outlier_features"] > 0).sum()),
        "outlier_rows_flagged": int(df["outlier_flag"].sum()),
        "outlier_feature_count_max": int(df["n_outlier_features"].max()),
        "sensor_min": float(sensor.to_numpy().min()),
        "sensor_max": float(sensor.to_numpy().max()),
        "sensor_median_abs": float(np.median(np.abs(sensor.to_numpy()))),
        "data_quality_issues": [],
    }

    dq = audit["data_quality_issues"]
    dq.append("CSV has a redundant first header row (Column1...Column131); real headers are on row 2.")
    dq.append("Metadata columns use comma decimals while sensor columns use dot decimals.")
    dq.append("ToolRotation, FeedRate and ToolDiameter are constant and cannot be useful predictors.")
    dq.append("RDOC and ADOC are constant within every experiment and every tool; class is confounded with tool identity.")
    dq.append("Milling tool type 2 is never associated with RDOC = 8 mm.")
    dq.append("14 cycles have CycleToFailure = 0 (right-censored / failure cycle records).")
    dq.append("P114_F01_C1 is a single-cycle early failure and an acknowledged outlier in the source paper.")
    dq.append("Multiple accelerometer minimum values reach the 16-bit ADC floor (-32768), indicating clipping.")
    dq.append("Sensor statistics are raw ADC-domain values, not calibrated g / A; conversion is documented in the source paper.")
    dq.append("The 25 GB raw time-series archive is not available locally; this project uses the authors' processed feature matrix.")
    return audit


def assign_folds(df: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic group folds and a two-experiment holdout demo."""
    folds = df[["FileName", "ExperimentID", "ToolIndex", "RDOC_class", "class_label"]].copy()
    folds["tool_loo"] = folds["ToolIndex"].astype(int)
    folds["experiment_loo"] = folds["ExperimentID"]

    # Five tool folds, each containing at least one dangerous and one optimal tool.
    dangerous_tools = sorted(df.loc[df["RDOC_class"] == 1, "ToolIndex"].unique())
    optimal_tools = sorted(df.loc[df["RDOC_class"] == 0, "ToolIndex"].unique())
    tool_fold = {}
    for i, tool in enumerate(dangerous_tools):
        tool_fold[int(tool)] = i % 5
    for i, tool in enumerate(optimal_tools):
        tool_fold[int(tool)] = i % 5
    folds["tool_5fold"] = folds["ToolIndex"].astype(int).map(tool_fold)

    # Five experiment folds, class-balanced at the experiment level.
    dangerous_exp = sorted(df.loc[df["RDOC_class"] == 1, "ExperimentID"].unique())
    optimal_exp = sorted(df.loc[df["RDOC_class"] == 0, "ExperimentID"].unique())
    exp_fold = {}
    for i, exp in enumerate(dangerous_exp):
        exp_fold[exp] = i % 5
    for i, exp in enumerate(optimal_exp):
        exp_fold[exp] = i % 5
    folds["experiment_5fold"] = folds["ExperimentID"].map(exp_fold)

    # Dedicated unseen-case demo: one dangerous and one optimal experiment.
    demo_experiments = ["P002", "P108"]
    folds["is_demo_holdout"] = folds["ExperimentID"].isin(demo_experiments).astype(int)

    return folds


def make_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in SENSOR_COLUMNS:
        parts = [p.strip() for p in col.split(" - ")]
        family = parts[0]
        channel = parts[1] if len(parts) > 1 else ""
        stat = parts[2] if len(parts) > 2 else ""
        rows.append(
            {
                "field": col,
                "role": "feature",
                "family": family,
                "channel_or_axis": channel,
                "statistic": stat,
                "dtype": "float64",
                "units": "raw ADC count (uncalibrated)",
                "description": f"{family} {channel} {stat} statistic for one milling cycle.",
                "notes": "Derived by the dataset authors from 25 kHz vibration / 0.5 kHz current signals.",
            }
        )
    meta_description = {
        "FileName": ("identifier", "Pxxx_Fyy_Czz cycle file name."),
        "ExperimentID": ("group_id", "Workpiece/sample experiment identifier (Pxxx)."),
        "LayerID": ("group_id", "Layer identifier within an experiment (Pxxx_Fyy)."),
        "P": ("group_id", "Parsed sample/workpiece id."),
        "F": ("group_id", "Parsed layer number."),
        "C": ("group_id", "Parsed cycle number within a layer."),
        "NumberOfCycle": ("metadata", "Cumulative number of cycles performed by the tool (known to the operator)."),
        "SampleIndex": ("metadata", "Sample/workpiece index from the metadata workbook."),
        "ToolIndex": ("group_id", "Cutting-tool index; the recommended cross-validation group."),
        "MillingToolType": ("metadata", "Cutter type: 1 = Van Hoorn, 2 = PARA Tooling."),
        "ADOC_mm": ("metadata", "Axial depth of cut (mm): 5 or 10."),
        "RDOC_mm": ("target_source", "Radial depth of cut (mm): 4.5 (optimal) or 8 (non-optimal)."),
        "Hardness_HRC": ("metadata", "Mean workpiece hardness (HRC)."),
        "ToolHolderLength_mm": ("metadata", "Tool holder length (mm): 80 or 160."),
        "CycleToFailure": ("metadata", "Cycles remaining until tool failure (right-censored at 0)."),
        "CycleToFailureNormalized": ("metadata", "Remaining tool life normalized to [0, 1]; 0 at failure."),
        "ToolRotation_rpm": ("constant", "Spindle speed; constant 3200 rpm."),
        "FeedRate_mm_min": ("constant", "Feed rate; constant 640 mm/min."),
        "ToolDiameter_mm": ("constant", "Tool diameter; constant 10 mm."),
        "RDOC_class": ("target", "1 = non-optimal/dangerous RDOC 8 mm; 0 = optimal 4.5 mm."),
        "class_label": ("target", "Human-readable class label."),
        "n_outlier_features": ("quality", "Number of sensor features outside the feature-wise 5xIQR fence for this cycle."),
        "outlier_flag": ("quality", "1 if at least five sensor features are outside their 5xIQR fence."),
    }
    for col, (role, desc) in meta_description.items():
        if col in df.columns:
            rows.append(
                {
                    "field": col,
                    "role": role,
                    "family": "metadata",
                    "channel_or_axis": "",
                    "statistic": "",
                    "dtype": str(df[col].dtype),
                    "units": "",
                    "description": desc,
                    "notes": "",
                }
            )
    return pd.DataFrame(rows)


def save_outputs(df: pd.DataFrame, audit: dict) -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    folds = assign_folds(df)
    clean = df.merge(folds.drop(columns=["ExperimentID", "ToolIndex", "RDOC_class", "class_label"]), on="FileName", how="left", validate="one_to_one")

    clean.to_csv(PROCESSED_FILES["clean"], index=False)
    clean[["FileName", "ExperimentID", "LayerID", "P", "F", "C", "ToolIndex", "MillingToolType", "ADOC_mm", "RDOC_mm", "RDOC_class", "class_label", "NumberOfCycle"] + SENSOR_COLUMNS].to_csv(
        PROCESSED_FILES["features"], index=False
    )
    metadata_cols = [
        "FileName", "ExperimentID", "LayerID", "P", "F", "C",
        "NumberOfCycle", "SampleIndex", "ToolIndex", "MillingToolType",
        "ADOC_mm", "RDOC_mm", "Hardness_HRC", "ToolHolderLength_mm",
        "CycleToFailure", "CycleToFailureNormalized",
        "ToolRotation_rpm", "FeedRate_mm_min", "ToolDiameter_mm",
        "RDOC_class", "class_label", "outlier_flag", "n_outlier_features",
    ]
    clean[metadata_cols].to_csv(PROCESSED_FILES["metadata"], index=False)
    folds.to_csv(PROCESSED_FILES["folds"], index=False)
    make_data_dictionary(clean).to_csv(PROCESSED_FILES["dictionary"], index=False)

    long = clean.melt(
        id_vars=["FileName", "ExperimentID", "LayerID", "P", "F", "C", "ToolIndex", "MillingToolType", "ADOC_mm", "RDOC_mm", "RDOC_class", "class_label", "CycleToFailure", "CycleToFailureNormalized"],
        value_vars=SENSOR_COLUMNS,
        var_name="feature",
        value_name="value",
    )
    parts = long["feature"].str.split(" - ", expand=True)
    long["family"] = parts[0]
    long["channel_or_axis"] = parts[1]
    long["statistic"] = parts[2]
    long.to_csv(PROCESSED_FILES["sensor_long"], index=False)

    with open(PROCESSED_FILES["audit"], "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2)

    PROCESSED_FILES["cleaning_log"].write_text(render_cleaning_log(audit), encoding="utf-8")


def render_cleaning_log(audit: dict) -> str:
    d = audit["dataset"]
    cls = audit["class_counts_cycles"]
    total = d["n_rows"]
    checks = audit["source_checks"]
    checks_txt = "\n".join(f"- `{k}`: {v}" for k, v in checks.items())
    issues = "\n".join(f"- {x}" for x in audit["data_quality_issues"])
    return f"""# Data Cleaning Log

## Source
- Feature matrix: `{d['raw_file']}` (authors' processed 120-feature dataset).
- Metadata: `{d['metadata_file']}`.
- Dataset: Piecuch & Żabiński, *Scientific Data* 12, 650 (2025), DOI 10.1038/s41597-025-04923-y.
- Data archive: DOI 10.6084/m9.figshare.28589216, CC BY 4.0.
- Raw time-series archive (25 GB) is not stored locally; the processed feature matrix is used as the input data for this analysis.

## Structure
- Rows: **{d['n_rows']}** milling cycles.
- Sensor features: **{d['n_sensor_features']}** (8 accelerometer channels + 12 current channels × min/max/mean/std/skew/kurtosis).
- Metadata source columns: **{d['n_metadata_source_columns']}**.
- Experiments: **{d['n_experiments']}**; layers: **{d['n_layers']}**; tools: **{d['n_tools']}**.

## Cross-source alignment
{checks_txt}

All shared metadata fields match the XLSX workbook row for row.

## Target definition
- `RDOC_class = 1` if RDOC = 8 mm (non-optimal / dangerous), otherwise `0` (optimal, 4.5 mm).
- Cycle counts: optimal = {cls.get(0, 0)}, dangerous = {cls.get(1, 0)} ({cls.get(1, 0) / total:.1%} dangerous).

## Data quality
- Missing values: **{audit['missing_values_total']}**.
- Duplicate full rows: **{audit['duplicate_rows_full']}**.
- Duplicate sensor vectors: **{audit['duplicate_sensor_rows']}**.
- Infinite sensor values: **{audit['infinite_sensor_values']}**.
- Zero-variance sensor features: **{len(audit['zero_variance_sensor_features'])}**.
- Censored cycles (`CycleToFailure = 0`): **{audit['censored_cycles_cycle_to_failure_zero']}**.
- Single-cycle experiments: **{', '.join(audit['single_cycle_experiments']) or 'none'}**.
- Accelerometer rows touching the ADC floor: **{audit['accelerometer_saturation_rows']}**.
- Rows with at least five 5xIQR extreme features: **{audit['outlier_rows_flagged']}** (kept, not deleted).

## Critical design issue
- RDOC and ADOC are constant within every experiment and every tool: experiments with both classes = **{audit['experiments_with_both_classes']}**, tools with both classes = **{audit['tools_with_both_classes']}**.
- Therefore random row-level splits leak the tool/experiment identity. All model selection uses tool-grouped cross-validation and pooled out-of-fold predictions.
- `MillingToolType = 2` cycles with dangerous RDOC: **{audit['tool_type_2_dangerous_cycles']}**.

## Decisions
1. Retained all 968 cycles, including the single-cycle early failure and censored records; they are valid process observations and removing them would be unjustified.
2. Did not alter accelerator clipping values; they are genuine sensor-domain observations.
3. Excluded all identifiers, process settings, material hardness, tool metadata and cycle-to-failure fields from the sensor-only feature matrix.
4. Dropped `ToolRotation_rpm`, `FeedRate_mm_min` and `ToolDiameter_mm` as predictors because they are constant.
5. Added an outlier flag for analysis only; no outlier rows were deleted.
6. Created deterministic tool and experiment folds plus a two-experiment holdout for the unseen-case demonstration.

## Known issues / limitations
{issues}
"""


if __name__ == "__main__":
    clean_df, meta_df, audit = build_clean_dataset()
    save_outputs(clean_df, audit)
    print(json.dumps(audit, indent=2))
    print(f"Saved to {DATA_PROCESSED}")
