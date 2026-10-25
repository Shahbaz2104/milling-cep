# EDA and Engineering Insights

Generated from the cleaned 968-cycle dataset. Figures are in `figures/`.

## Class balance

- Cycles: 756 optimal vs 212 non-optimal (21.9% dangerous).
- Experiments: 16 optimal vs 8 non-optimal.
- Tools: 9 optimal vs 5 non-optimal.

## Insights

1. **The class is a minority condition.** Only 21.9% of cycles, 8/24 experiments and 5/14 tools operate at the non-optimal RDOC of 8 mm. Accuracy alone would be misleading; recall for the dangerous class and PR-AUC are the appropriate metrics.

2. **RDOC is perfectly confounded with tool/experiment identity.** Every tool and every experiment uses exactly one RDOC/ADOC combination, and tool type 2 is never used at RDOC = 8 mm. A random row split lets the model memorise tool signatures and inflates performance. All model selection must use tool-grouped validation with pooled out-of-fold predictions.

3. **The sensor signal is physically plausible and strong.** Current standard deviation on the drive-X and spindle channels, and mean acceleration on the drive axes, are the strongest univariate features. A larger radial engagement increases cutting force, spindle/drive current and vibration, which is consistent with the observed direction. The best univariate AUCs are shown in `fig06_univariate_auc.png`.

4. **Wear is a second confound.** Dangerous tools fail sooner (mean cycles to failure 22.7 vs 55.8). Sensor features also drift with remaining tool life, so part of the class separation may reflect wear stage rather than RDOC alone. This must be discussed as a limitation and, ideally, checked with early-life-only sensitivity analysis.

5. **Hardness and tool type differ by class.** The dangerous group has lower mean hardness and is entirely composed of tool type 1, while tool type 2 appears only in the optimal group. These material/tool differences are plausible false-positive drivers; the tool 7 error case in the modelling section is an example.

## Top univariate features

| Feature | AUC | Direction |
|---|---:|---|
| `Accelerometer - Y Driving axle +Y - mean` | 0.939 | higher in dangerous |
| `Accelerometer - X Driving axle +Z - mean` | 0.935 | higher in dangerous |
| `Current - Driving axle X L1 - std` | 0.913 | higher in dangerous |
| `Current - Driving axle X L3 - std` | 0.913 | higher in dangerous |
| `Current - Driving axle X L2 - std` | 0.913 | higher in dangerous |
| `Current - Spindle L1 - std` | 0.894 | higher in dangerous |
| `Current - Spindle L2 - std` | 0.894 | higher in dangerous |
| `Current - Spindle L3 - std` | 0.894 | higher in dangerous |

## Tool-level condition map

|   ToolIndex |   n |   RDOC |   ADOC |   tool_type |   hardness |   rdoc_class |
|------------:|----:|-------:|-------:|------------:|-----------:|-------------:|
|           2 |  48 |    8   |      5 |           1 |      36.67 |            1 |
|           3 |  44 |    8   |     10 |           1 |      40    |            1 |
|           4 |  41 |    8   |      5 |           1 |      35.33 |            1 |
|           5 |  51 |    8   |     10 |           1 |      38.17 |            1 |
|           6 |  28 |    8   |      5 |           1 |      40    |            1 |
|           7 | 124 |    4.5 |     10 |           1 |      40.33 |            0 |
|           8 | 116 |    4.5 |      5 |           1 |      41.67 |            0 |
|           9 | 149 |    4.5 |      5 |           1 |      37.33 |            0 |
|          10 |  15 |    4.5 |     10 |           1 |      37.33 |            0 |
|          11 |   1 |    4.5 |     10 |           1 |      37.17 |            0 |
|         101 |  84 |    4.5 |     10 |           2 |      39.33 |            0 |
|         102 |  82 |    4.5 |      5 |           2 |      38.33 |            0 |
|         103 |  60 |    4.5 |     10 |           2 |      39.83 |            0 |
|         105 | 125 |    4.5 |      5 |           2 |      38.33 |            0 |

## Figures

- `fig01_class_balance.png`
- `fig02_tool_condition_confounding.png`
- `fig03_top_sensor_distributions.png`
- `fig04_pca_correlation.png`
- `fig05_wear_confound.png`
- `fig06_univariate_auc.png`
