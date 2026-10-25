# Milling Condition Safety Monitor

Detecting non-optimal CNC milling conditions (RDOC = 8 mm) from accelerometer and
spindle/drive-current statistics.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://milling-cep-iguwxgwerngzavkpz2aguj.streamlit.app)

**ME-446 Artificial Intelligence & Internet of Things — Complex Engineering Problem, Fall 2026**
Department of Mechanical Engineering, NED University of Engineering and Technology, Karachi.

---

## Live dashboard

**https://milling-cep-iguwxgwerngzavkpz2aguj.streamlit.app**

The app is deployed on **Streamlit Community Cloud** (free). No secrets, database or API
key is required: it reads the committed model and data artifacts.

To deploy another copy:

1. Open the [deploy page](https://share.streamlit.io/deploy?repository=https://github.com/Shahbaz2104/milling-cep&branch=main&mainModule=dashboard/app.py)
   and sign in with GitHub.
2. If the fields are not prefilled, enter:
   - Repository: `Shahbaz2104/milling-cep`
   - Branch: `main`
   - Main file path: `dashboard/app.py`
3. Choose the Python version **3.12** in Advanced settings.
4. Pick a subdomain such as `milling-condition-monitor`, then click **Deploy**.

The app is usually live in a few minutes at
`https://<your-subdomain>.streamlit.app`. Free apps sleep after inactivity and wake on the
first visit. Logs are visible only to GitHub users with write access to the repository.

**Alternative:** any container host works. The start command is:

```bash
streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

## The problem

A 10 mm cutter run at a radial depth of cut of 8 mm engages 80 % of its diameter. Chip
load, cutting force and spindle torque rise; chatter, accelerated wear, scrap and tool
breakage become likely. The operator knows the programmed setting, but a process monitor
does not — it sees only vibration and current.

This project treats RDOC as a binary target and asks whether the sensor signature alone
can flag a dangerous cut. It deliberately does **not** reproduce the source paper's tool-wear
model; the classification target is a new problem on the same public data.

| | |
|---|---|
| Task | Binary classification (optimal 4.5 mm vs dangerous 8 mm) |
| Inputs | 120 accelerometer/current statistics per cycle |
| Cycles | 968 (212 dangerous, 21.9 %) |
| Experiments / tools | 24 / 14 |
| Validation | 5-fold **tool-group** CV, leave-one-tool-out, leave-one-experiment-out |
| Selected model | Random forest, 150 trees, depth ≤ 6 |
| Operating threshold | 0.444 (max F1 on grouped out-of-fold probabilities) |

## Headline results

| Metric | 5-fold tool groups | Leave-one-tool-out |
|---|---:|---:|
| Recall (dangerous) | **0.962** | 0.962 |
| Precision | 0.699 | 0.718 |
| F1 | 0.810 | 0.823 |
| MCC | 0.762 | 0.778 |
| Balanced accuracy | 0.923 | 0.928 |
| ROC-AUC | 0.942 | 0.950 |
| PR-AUC | 0.688 | 0.712 |
| Brier score | 0.082 | 0.076 |

Held-out experiments (never used in training):

- **P002** (dangerous): 45 of 46 cycles flagged (97.8 %).
- **P108** (optimal): 0 of 81 cycles flagged.
- The single P002 miss is the first cycle of the experiment (P = 0.439, just below 0.444).

The important methodological point: RDOC is constant within every tool and experiment, so a
random row split leaks tool identity. Random-split ROC-AUC reaches 0.998; the honest
tool-grouped estimate is 0.942. All reported numbers are grouped.

## Repository layout

```
data/raw/              Original dataset files (unchanged)
data/processed/        Cleaned data, audit, folds, data dictionary, long sensor table
src/                   Data pipeline and modelling utilities
scripts/               EDA, modelling, model selection, Tableau export, slide build
models/                Final model, metrics, predictions, unseen-case evidence
figures/               11 report/dashboard figures
dashboard/app.py       Streamlit dashboard (6 tabs, threshold slider, live scoring)
dashboard/tableau/     Tableau-ready extracts + build guide
notebooks/             Executed reproducible notebook + HTML/PDF exports
reports/               Technical report (MD/HTML/PDF), slides, poster, EDA insights
```

## Quick start

Python 3.12 is used. Create a virtual environment and install the pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # runtime + dashboard
# pip install -r requirements-dev.txt  # notebook, report and slide tooling
```

Reproduce the pipeline from the project root:

```bash
python -m src.data                  # cleaning, audit, folds, long sensor table
python scripts/02_eda.py            # figures 01-06, EDA_INSIGHTS.md
python scripts/03_modeling.py       # model comparison, figures 07-10
python scripts/05_select_rf.py      # final model, threshold, unseen cases, figure 11
python scripts/06_tableau_export.py # Tableau extracts
streamlit run dashboard/app.py      # interactive dashboard
```

Rebuild the documents:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/milling_cep_analysis.ipynb
jupyter nbconvert --to html notebooks/milling_cep_analysis.ipynb --output-dir notebooks/exports
python scripts/07_build_slides.py
libreoffice --headless --convert-to pdf reports/presentation/MILLING_CEP_SLIDES.pptx
```

The poster is a print-ready A2 HTML source at `reports/poster/MILLING_CEP_POSTER.html`;
print it to PDF with Chrome (A2, no margins).

Random seed: `20261025`.

## Deliverables

| Artifact | Path |
|---|---|
| Technical report (PDF) | `reports/TECHNICAL_REPORT.pdf` |
| Executed notebook (HTML) | `notebooks/exports/milling_cep_analysis.html` |
| Executed notebook (PDF) | `notebooks/exports/milling_cep_analysis.pdf` |
| Presentation (PPTX / PDF) | `reports/presentation/MILLING_CEP_SLIDES.pptx` / `.pdf` |
| Poster (A2 PDF) | `reports/poster/MILLING_CEP_POSTER.pdf` |
| Streamlit dashboard | `dashboard/app.py` |
| Tableau extracts + guide | `dashboard/tableau/` |
| Cleaned dataset | `data/processed/milling_clean.csv` |
| Data audit | `data/processed/CLEANING_LOG.md`, `audit.json` |

The Streamlit app is the working interactive dashboard delivered with the project. The
Tableau CSVs and build guide let the same views be assembled in Tableau Public; the `.twbx`
workbook must be built in Tableau itself.

## Limitations

- **Confounded design.** RDOC is constant within each tool and experiment, so tool geometry,
  material batch, hardness and wear stage are entangled with the target. Grouped validation
  limits the optimism but cannot establish that the model responds to RDOC alone.
- **Wear confound.** Dangerous tools fail sooner (mean 22.7 vs 55.8 cycles); the false alarms
  on optimal tool 7 cluster late in tool life.
- **Small tool count.** 14 tools, 5 dangerous; one tool drives 75 of the 88 false alarms.
- **Pre-computed features.** The 25 GB raw waveforms were not available; the authors' 120
  statistics are used as-is.
- **Uncalibrated units.** Values are raw ADC counts from one machine and one campaign.
- **Threshold is a policy choice.** 0.444 favours recall. The dashboard exposes the trade-off.

The model is a screening alarm, not a causal instrument and not an automatic stop.

## Data and citation

Piecuch, M. & Żabiński, T. *A new open dataset from a milling process – data for
classification and estimation of tool life.* **Scientific Data 12, 650 (2025).**
<https://doi.org/10.1038/s41597-025-04923-y>

Dataset archive: <https://doi.org/10.6084/m9.figshare.28589216> — **CC BY 4.0**.
The raw files in `data/raw/` are redistributed unchanged under that licence.

If you use this analysis, cite the dataset above and the course project:

> Shahbaz. *Milling Condition Safety Monitor: Detecting Non-Optimal Cutting Parameters from
> Vibration and Current Signals.* ME-446 CEP, NED University, 2026.

## References

- Pedregosa, F. et al. Scikit-learn: Machine learning in Python. *JMLR* 12, 2825–2830 (2011).
- Breiman, L. Random forests. *Machine Learning* 45, 5–32 (2001).
- Saito, T. & Rehmsmeier, M. The precision-recall plot is more informative than the ROC plot
  when evaluating binary classifiers on imbalanced datasets. *PLOS ONE* 10, e0118432 (2015).
- Chicco, D. & Jurman, G. The advantages of MCC over F1 score and accuracy in binary
  classification evaluation. *BMC Genomics* 21, 6 (2020).
