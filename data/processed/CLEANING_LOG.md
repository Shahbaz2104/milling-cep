# Data Cleaning Log

## Source
- Feature matrix: `FeatureAndMetadata_Milling.csv` (authors' processed 120-feature dataset).
- Metadata: `metadata.xlsx`.
- Dataset: Piecuch & Żabiński, *Scientific Data* 12, 650 (2025), DOI 10.1038/s41597-025-04923-y.
- Data archive: DOI 10.6084/m9.figshare.28589216, CC BY 4.0.
- Raw time-series archive (25 GB) is not stored locally; the processed feature matrix is used as the input data for this analysis.

## Structure
- Rows: **968** milling cycles.
- Sensor features: **120** (8 accelerometer channels + 12 current channels × min/max/mean/std/skew/kurtosis).
- Metadata source columns: **11**.
- Experiments: **24**; layers: **172**; tools: **14**.

## Cross-source alignment
- `SampleIndex__vs__SampleIndex_xlsx`: True
- `ToolIndex__vs__ToolIndex_xlsx`: True
- `MillingToolType__vs__MillingToolType_xlsx`: True
- `ADOC_mm__vs__ADOC_mm_xlsx`: True
- `RDOC_mm__vs__RDOC_mm_xlsx`: True
- `Hardness_HRC__vs__Hardness_HRC_xlsx`: True
- `CycleToFailure__vs__CycleToFailure_xlsx`: True
- `CycleToFailureNormalized__vs__CycleToFailureNormalized_xlsx`: True
- `ToolHolderLength_mm__vs__ToolHolderLength_mm_xlsx`: True
- `row_order_matches`: True
- `row_counts`: {'csv': 968, 'xlsx': 968}

All shared metadata fields match the XLSX workbook row for row.

## Target definition
- `RDOC_class = 1` if RDOC = 8 mm (non-optimal / dangerous), otherwise `0` (optimal, 4.5 mm).
- Cycle counts: optimal = 756, dangerous = 212 (21.9% dangerous).

## Data quality
- Missing values: **0**.
- Duplicate full rows: **0**.
- Duplicate sensor vectors: **0**.
- Infinite sensor values: **0**.
- Zero-variance sensor features: **0**.
- Censored cycles (`CycleToFailure = 0`): **14**.
- Single-cycle experiments: **P114**.
- Accelerometer rows touching the ADC floor: **4**.
- Rows with at least five 5xIQR extreme features: **52** (kept, not deleted).

## Critical design issue
- RDOC and ADOC are constant within every experiment and every tool: experiments with both classes = **0**, tools with both classes = **0**.
- Therefore random row-level splits leak the tool/experiment identity. All model selection uses tool-grouped cross-validation and pooled out-of-fold predictions.
- `MillingToolType = 2` cycles with dangerous RDOC: **0**.

## Decisions
1. Retained all 968 cycles, including the single-cycle early failure and censored records; they are valid process observations and removing them would be unjustified.
2. Did not alter accelerator clipping values; they are genuine sensor-domain observations.
3. Excluded all identifiers, process settings, material hardness, tool metadata and cycle-to-failure fields from the sensor-only feature matrix.
4. Dropped `ToolRotation_rpm`, `FeedRate_mm_min` and `ToolDiameter_mm` as predictors because they are constant.
5. Added an outlier flag for analysis only; no outlier rows were deleted.
6. Created deterministic tool and experiment folds plus a two-experiment holdout for the unseen-case demonstration.

## Known issues / limitations
- CSV has a redundant first header row (Column1...Column131); real headers are on row 2.
- Metadata columns use comma decimals while sensor columns use dot decimals.
- ToolRotation, FeedRate and ToolDiameter are constant and cannot be useful predictors.
- RDOC and ADOC are constant within every experiment and every tool; class is confounded with tool identity.
- Milling tool type 2 is never associated with RDOC = 8 mm.
- 14 cycles have CycleToFailure = 0 (right-censored / failure cycle records).
- P114_F01_C1 is a single-cycle early failure and an acknowledged outlier in the source paper.
- Multiple accelerometer minimum values reach the 16-bit ADC floor (-32768), indicating clipping.
- Sensor statistics are raw ADC-domain values, not calibrated g / A; conversion is documented in the source paper.
- The 25 GB raw time-series archive is not available locally; this project uses the authors' processed feature matrix.
