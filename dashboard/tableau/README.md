# Tableau dashboard build guide

This folder contains Tableau-ready extracts of the cleaned milling dataset, the
model outputs and the validation evidence. The CEP requires a Tableau Public
workbook (or an approved alternative); the Streamlit app in `dashboard/app.py`
is the working interactive dashboard delivered with this project, and these
extracts are the data source for building the equivalent Tableau workbook.

Regenerate every file with:

```bash
.venv/bin/python scripts/06_tableau_export.py
```

## Files

| File | Grain | Rows | What it is for |
|---|---|---:|---|
| `tableau_cycles.csv` | one milling cycle | 968 | Main fact table: metadata, class label, out-of-fold probabilities, predictions, error type. |
| `tableau_sensor_long.csv` | cycle × sensor statistic | 116,160 | Long sensor table for distributions and per-cycle sensor signatures. Join to cycles on `FileName`. |
| `tableau_tool_summary.csv` | one tool | 14 | Per-tool recall (dangerous tools) or false-alarm rate (optimal tools). |
| `tableau_kpis.csv` | one row | 1 | Pre-computed KPI tiles and confusion-matrix counts, so header numbers match the report exactly. |
| `tableau_model_comparison.csv` | model × validation scheme | 16 | Model comparison table (ROC-AUC, PR-AUC, recall, precision, MCC, Brier, accuracy). |
| `tableau_feature_importance.csv` | one feature | 30 | Top permutation importances for the selected random forest. |
| `tableau_unseen_experiments.csv` | one unseen experiment | 2 | Aggregate behaviour on P002 (dangerous) and P108 (optimal). |
| `tableau_unseen_cycles.csv` | one held-out cycle | 127 | Cycle-level probabilities for the unseen experiments (held-out model). |

### Key columns

`tableau_cycles.csv`

| Column | Meaning |
|---|---|
| `FileName` | Cycle identifier (`P002_F01_C1`); join key to the sensor table. |
| `ExperimentID`, `LayerID`, `P`, `F`, `C` | Experiment / layer / cycle hierarchy. |
| `ToolIndex`, `tool_label` | Tool identity; **the grouping unit for honest validation**. |
| `ADOC_mm`, `RDOC_mm` | Axial and radial depth of cut. |
| `RDOC_class`, `class_short` | Target: 1 = dangerous (RDOC 8 mm), 0 = optimal (4.5 mm). |
| `CycleToFailure`, `cycle_normalized` | Remaining cycles and normalized life (1 = first cut, 0 = failure). |
| `prob_primary` | Out-of-fold P(dangerous) under 5-fold tool-group validation. |
| `prob_leave_one_tool_out` | P(dangerous) with the tool excluded from training (stress test). |
| `pred_tool5`, `pred_loo` | Decisions at the tuned threshold (0.4437). |
| `outcome_tool5` | True positive / false negative / false positive / true negative. |
| `threshold` | The threshold used for the saved decisions. |

`sensor_long.csv` columns: `FileName`, `ExperimentID`, `ToolIndex`, `class_short`,
`class_label`, `cycle_normalized`, `feature`, `family` (Accelerometer/Current),
`channel_or_axis`, `statistic` (min/max/mean/std/skew/kurtosis), `value`.

## Import

1. Open Tableau Public and connect to **Text file** → `tableau_cycles.csv`.
2. Add `tableau_sensor_long.csv` and `tableau_tool_summary.csv` as additional
   connections.
3. On the Data Source page, drag the three tables into the canvas:
   - `tableau_cycles` ↔ `tableau_sensor_long` on **FileName** (1:many).
   - `tableau_cycles` ↔ `tableau_tool_summary` on **ToolIndex** (many:1).
   Tableau's relationship model handles the different grains; do not physically
   join the sensor table to the cycle table for the cycle-level views.
4. Set data types: `RDOC_class`, `pred_tool5`, `pred_loo`, `ToolIndex` → integer;
   `class_short` and `outcome_tool5` → string; `prob_primary` → decimal.
5. Rename the data source to **Milling CEP**.

## Calculated fields

Create these in the data source (they aggregate correctly under filters).

```
// Confusion-matrix cells
TP  = IF [RDOC_class] = 1 AND [pred_tool5] = 1 THEN 1 ELSE 0 END
FN  = IF [RDOC_class] = 1 AND [pred_tool5] = 0 THEN 1 ELSE 0 END
FP  = IF [RDOC_class] = 0 AND [pred_tool5] = 1 THEN 1 ELSE 0 END
TN  = IF [RDOC_class] = 0 AND [pred_tool5] = 0 THEN 1 ELSE 0 END

Recall (dangerous)     = SUM([TP]) / (SUM([TP]) + SUM([FN]))
Precision (dangerous)  = SUM([TP]) / (SUM([TP]) + SUM([FP]))
False-alarm rate       = SUM([FP]) / (SUM([FP]) + SUM([TN]))
Specificity            = 1 - [False-alarm rate]
Balanced accuracy      = ([Recall (dangerous)] + [Specificity]) / 2
F1                     = 2 * [Precision (dangerous)] * [Recall (dangerous)]
                         / ([Precision (dangerous)] + [Recall (dangerous)])
Flagged                = IF [pred_tool5] = 1 THEN "Flagged dangerous" ELSE "Cleared" END
Class label            = IF [RDOC_class] = 1 THEN "Non-optimal (RDOC 8 mm)"
                         ELSE "Optimal (RDOC 4.5 mm)" END
Predicted label        = IF [pred_tool5] = 1 THEN "Predicted dangerous"
                         ELSE "Predicted optimal" END
Life stage             = IF [cycle_normalized] >= 0.66 THEN "Early"
                         ELSEIF [cycle_normalized] >= 0.33 THEN "Mid"
                         ELSE "Late" END
Probability band       = IF [prob_primary] < 0.2 THEN "0.0–0.2"
                         ELSEIF [prob_primary] < 0.4 THEN "0.2–0.4"
                         ELSEIF [prob_primary] < 0.6 THEN "0.4–0.6"
                         ELSEIF [prob_primary] < 0.8 THEN "0.6–0.8"
                         ELSE "0.8–1.0" END
```

Optional threshold parameter (Analysis → Create Parameter), type Float, current
value `0.4437`, display format `0.000`. Then use a parameter-driven decision:

```
Predicted (parameter) = IF [prob_primary] >= [Decision threshold] THEN 1 ELSE 0 END
```

Use `pred_tool5` for the saved numbers in the report. If you switch to the
parameter, replace `[pred_tool5]` inside the confusion cells with
`[Predicted (parameter)]` and note that the workbook is no longer showing the
pre-registered operating point.

## Worksheets

### 1. KPI header (5 BANs)

Use `tableau_kpis.csv` (no joins) so the numbers exactly match the report:
`cycles`, `dangerous_cycles` or `dangerous_share`, `recall`, `precision`, `mcc`.
Format recall/precision as percentages and MCC/Brier to three decimals.

### 2. Class balance

`tableau_cycles` → bar chart: `class_short` on Columns, `COUNT(FileName)` on
Rows, `class_short` on Color. Add a label with the count and the share.

### 3. Tool condition map

`tableau_cycles` → `tool_label` on Rows, `RDOC_mm` on Text (or AVG), `COUNT(FileName)`
on Text. Color by `class_short`. This makes the confounding visible: every tool
has exactly one RDOC value.

### 4. Probability distribution by class

`tableau_cycles` → box-and-whisker (Show Me → Box Plot) with `class_short` on
Columns and `prob_primary` on Rows. Add the reference line `0.4437`. Point
detail shows the overlap between classes that drives the false alarms.

### 5. Probability vs remaining life

Scatter: `cycle_normalized` on Columns (reverse axis so 1 = first cut on the
left), `prob_primary` on Rows, `class_short` on Color, `ToolIndex` on Detail.
Add a constant reference line at the threshold and a trend line per class. This
is the wear-confound check.

### 6. Confusion matrix

Use `outcome_tool5` on Rows and Columns as a heatmap with `COUNT(FileName)` as
text and color. Alternatively build it from the four calculated fields TP/FN/FP/TN.

### 7. Error analysis

Table: `ToolIndex`, `class_short`, `outcome_tool5`, `COUNT(FileName)`. Filter to
false negatives and false positives. Add `AVG(prob_primary)` and
`AVG(cycle_normalized)` to see whether errors cluster early or late in tool life.

### 8. Model comparison

`tableau_model_comparison.csv` → text table with `model` on Rows, `scheme` on
Columns, and one sheet per metric (ROC-AUC, PR-AUC, recall, precision, F1, MCC,
Brier). Add a `scheme` filter so the grouped schemes and the random-split
leakage control can be shown separately. Never present the random split as the
headline.

### 9. Feature importance

`tableau_feature_importance.csv` → horizontal bar, `feature` on Rows,
`importance_mean` on Columns, error bars from `importance_std` (Gantt bar trick
or reference band), colored by whether `feature` starts with "Accelerometer".

### 10. Sensor signature

`tableau_sensor_long` → small multiples: `family` on Columns, `class_short` on
Color, `AVG(value)` or a distribution by `statistic`. Filter to six features
with a `feature` quick filter. Because sensor statistics are raw ADC-domain
values, label the axis "ADC counts (uncalibrated)".

### 11. Unseen experiments

`tableau_unseen_experiments.csv` → bar of `frac_predicted_dangerous` by
`ExperimentID`, with `prob_mean` as a label. Add `tableau_unseen_cycles` as a
second view showing the per-cycle probabilities and the threshold line.

## Dashboard assembly

Create three dashboards, all sized **1366 × 768** (or Automatic):

1. **Operating overview** — KPI header, class balance, tool condition map,
   probability distribution, threshold parameter.
2. **Error and wear analysis** — probability vs remaining life, error analysis
   table, confusion matrix, tool-level summary.
3. **Model evidence and unseen cases** — model comparison, feature importance,
   sensor signature, unseen experiment view, plus a text box with the
   validation caveat.

Filters (apply to all sheets using the relevant data source):

- `ExperimentID`, `ToolIndex`, `class_short`, `outcome_tool5`.
- `cycle_normalized` range slider for life stage.
- `family` and `statistic` for the sensor view.

Use **Apply to all worksheets using this data source** so the pages stay
consistent. Avoid filtering the KPI sheet by tool: the header must always show
the full test-set numbers.

Design notes: use the same palette as the Streamlit app (optimal `#2E86AB`,
dangerous `#D1495B`, neutral `#5B6573`), white background, light gridlines,
and a single accent for the selected model. Put the caveat "tool-grouped
validation; not a random split" in the dashboard footer, not in a tooltip.

## Publishing and evidence

1. File → Save to Tableau Public as `Milling Condition Safety Monitor`.
2. Copy the public URL into the report and the submission README.
3. Take screenshots of all three dashboards with a filter applied, save them in
   `reports/screenshots/`, and cite them in the report as dashboard evidence.
4. Export the workbook (`File → Export Packaged Workbook`, `.twbx`) for the
   submission. The workbook is a binary Tableau artifact and must be created in
   Tableau Desktop/Public; it cannot be generated by the Python pipeline.

## Notes and caveats

- `prob_primary` and `prob_leave_one_tool_out` are **out-of-fold** predictions
  for the 968 training cycles. They are not predictions from a model fitted on
  all cycles. The `final_model.joblib` bundle is the deployable model; use the
  Streamlit "Score a cycle" tab for live scoring.
- The per-tool summary reports recall for dangerous tools and false-alarm rate
  for optimal tools; do not average the `flagged_fraction` column across both
  classes and call it accuracy.
- Tool 11 has one cycle. Keep it in the tool map but annotate it; do not draw
  conclusions from its per-tool rate.
- `tableau_sensor_long.csv` is 16 MB. If Tableau Public rejects the extract,
  filter it to the top 24 features from `tableau_feature_importance.csv` before
  publishing, or use an extract with only the mean/std statistics.
