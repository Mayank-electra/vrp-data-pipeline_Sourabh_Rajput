# ⚡ VRP Telemetry Data Cleaning & Preprocessing Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade, modular, dataset-agnostic Python data engineering pipeline designed to clean, standardize, analyze, and preprocess Electric Vehicle (EV) telemetry from the Vehicle Routing Problem (VRP) fleet prior to JEV model analysis.

The pipeline runs seamlessly on existing raw telemetry files (e.g. `2026-10-27_EV_Analysis_Data.csv` and `voltlens-*.csv`) and dynamically accommodates any new raw CSV with **zero code edits**.

---

## 🏗️ Architecture & Pipeline Flow

```
Raw CSV / Multiple Telemetry Logs
                │
                ▼
┌───────────────────────────────────────────────┐
│ 1. Data Ingestion & Alignment                 │
│    - Multi-file concatenation & union         │
│    - Mixed UTC timestamp parsing (+00:00 / Z) │
│    - Chronological reordering                 │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│ 2. Data Cleaning & Domain Guards              │
│    - snake_case column normalization          │
│    - Deduplication & unconfirmed sensor drops │
│    - High-missingness dropping (>60%)         │
│    - EV physical domain guards:               │
│      * Speed clamped to [0, 150] km/h         │
│      * Non-negative cell_delta_mv (min 0 mV)  │
│      * SOC clamped to [0, 100]%               │
│    - IQR-based outlier capping (winsorization)│
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│ 3. Exploratory Data Analysis & Visualizations │
│    - Missingness horizontal bar chart         │
│    - Feature-target correlation ranking       │
│    - Feature pairwise correlation heatmap     │
│    - Numerical feature outlier boxplots       │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│ 4. Leak-Free Data Splitting (70/15/15)        │
│    - Chronological (time-series safe) or      │
│      Random train/val/test partitioning       │
│    - Segregated BEFORE fitting any estimator  │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│ 5. Feature Selection (Fitted on X_train)      │
│    - Constant / zero-variance feature removal │
│    - Multicollinearity elimination (|r| > 0.9)│
│    - Optional SelectKBest univariate ranking  │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│ 6. Preprocessing ColumnTransformer            │
│    - Numeric: SimpleImputer(median) + Scaler  │
│      (StandardScaler / MinMaxScaler / Robust) │
│    - Categorical: SimpleImputer(most_frequent)│
│      + OneHotEncoder(handle_unknown='ignore') │
│    - Preserves legible column names           │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│ 7. Production Artifact Exports                │
│    - data/processed/train.csv (0 NaNs)        │
│    - data/processed/val.csv   (0 NaNs)        │
│    - data/processed/test.csv  (0 NaNs)        │
│    - data/processed/preprocessor.joblib       │
│    - outputs/cleaning_report.json             │
└───────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```text
vrp-data-pipeline/
├── data/
│   ├── raw/                              # Raw input CSVs (e.g. 2026-10-27_EV_Analysis_Data.csv)
│   └── processed/                        # Outputs: train.csv, val.csv, test.csv, preprocessor.joblib
├── src/
│   ├── __init__.py
│   ├── config.py                         # Centralized thresholds, physical guards, metadata specs
│   ├── data_loader.py                    # Multi-CSV loader, schema aligner, UTC timestamp parser
│   ├── cleaning.py                       # Column cleaner, deduplication, domain guards, IQR capping
│   ├── analysis.py                       # Pearson/Spearman correlation & 300 DPI visualization engine
│   ├── feature_selection.py              # Zero-variance dropper & leak-free collinearity remover
│   ├── preprocessing.py                  # Scikit-learn ColumnTransformer & VRPPreprocessor class
│   └── pipeline.py                       # Master CLI runner executing end-to-end pipeline
├── outputs/
│   ├── plots/                            # High-resolution PNG visualizations
│   │   ├── missingness_bar_chart.png
│   │   ├── correlation_heatmap.png
│   │   ├── feature_target_bar_plot.png
│   │   └── outlier_boxplots.png
│   └── cleaning_report.json              # Comprehensive before/after audit trail
├── notebooks/
│   └── vrp_pipeline_walkthrough.ipynb    # Clean, fully-executed step-by-step review notebook
├── app.py                                # Production Streamlit web application & visual dashboard
├── tests/
│   └── test_pipeline.py                  # Pytest test suite (zero NaNs, zero leakage, reloadability)
├── requirements.txt                      # Pinned Python package dependencies
├── .gitignore                            # Clean Git hygiene excluding data & caches
└── README.md                             # Comprehensive pipeline documentation
```

---

## ⚙️ Strict Engineering Constraints & Leakage Prevention

1. **Zero Data Leakage:**
   Data is split into **70% Train / 15% Validation / 15% Test** *before* fitting any transformer, imputer, encoder, feature selector, or scaler. All parameters (e.g. median imputation statistics, standard scaler mean/variance, one-hot category levels, collinearity rankings) are learned strictly from `X_train` and then applied out-of-sample via `.transform()` to `X_val` and `X_test`.

2. **EV Physical Domain Boundary Guards:**
   - **Speed (`speed`, `speed_raw`, `speed_valid`, `speed_kmh`):** Clamped to physical bounds `[0.0, 150.0] km/h`.
   - **Cell Delta Voltage (`cell_delta_mv`):** Physically required to be non-negative ($\ge 0.0\text{ mV}$). Erroneous negative readings are clamped to $0.0$.
   - **State of Charge (`soc`):** Validated in `[0.0, 100.0]%`.
   - **Pack Voltage (`pack_voltage`):** Clamped to valid battery operating ranges `[100.0, 600.0] V`.

3. **IQR-Based Outlier Capping (Winsorization):**
   Instead of dropping rows and corrupting high-frequency telemetry timestamps, extreme values beyond $[Q_1 - 1.5\times\text{IQR}, Q_3 + 1.5\times\text{IQR}]$ are winsorized to the boundary limits. Row removal is also available via `--outlier-strategy remove`.

4. **Collinearity Elimination (|r| > 0.90):**
   Strongly redundant sensor channels (e.g. `speed_raw` vs `speed_valid`, `power_kw` vs `power_calc_kw`, `max_cell_mv` vs `min_cell_mv`) are detected on `X_train`. The feature with lower correlation to the target is safely removed.

---

## 🚀 Quickstart & Installation

### 1. Environment Setup

```bash
# Clone the repository
git clone <repo-url>
cd vrp-data-pipeline

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 💻 CLI Usage Guide

### Standard Run on Real EV Dataset (Chronological Split)
```bash
python src/pipeline.py --input data/raw/2026-10-27_EV_Analysis_Data.csv --target power_kw --split-mode chronological --scaler standard
```

### Processing Raw `voltlens-*.csv` Files with Sensor Dropping
```bash
python src/pipeline.py --input "data/raw/voltlens-*.csv" --target power_kw --drop-unconfirmed --split-mode chronological
```

### Random Splitting with MinMaxScaler
```bash
python src/pipeline.py --input data/raw/2026-10-27_EV_Analysis_Data.csv --target power_kw --split-mode random --scaler minmax
```

### Advanced Parameter Customization
```bash
python src/pipeline.py \
  --input data/raw/2026-10-27_EV_Analysis_Data.csv \
  --target power_kw \
  --split-mode chronological \
  --scaler robust \
  --missing-threshold 0.50 \
  --corr-threshold 0.85 \
  --iqr-multiplier 1.5 \
  --outlier-strategy cap \
  --output-dir data/processed \
  --plots-dir outputs/plots \
  --report-path outputs/cleaning_report.json
```

### CLI Flag Reference
| Flag | Default | Description |
|---|---|---|
| `--input` | `data/raw/2026-10-27_EV_Analysis_Data.csv` | File path, folder, or glob pattern of raw CSVs |
| `--target` | `power_kw` | Target column name for supervised model analysis |
| `--split-mode` | `chronological` | Split strategy: `chronological` (time safe) or `random` |
| `--scaler` | `standard` | Feature scaling method: `standard`, `minmax`, or `robust` |
| `--drop-unconfirmed`| `False` | Drops `*_unconfirmed` sensor columns |
| `--outlier-strategy`| `cap` | Strategy for handling IQR outliers: `cap` or `remove` |
| `--missing-threshold`| `0.60` | Threshold ratio to drop high-missing columns |
| `--corr-threshold` | `0.90` | Pairwise correlation threshold to drop collinear features |
| `--k-best` | `0` | Select top K predictive features via SelectKBest (0 = retain all) |

---

## 🌐 Streamlit Interactive Dashboard

Launch the browser-based dashboard to inspect telemetry, adjust parameters, visualize correlation heatmaps, and download processed splits:

```bash
python -m streamlit run app.py
```

### Dashboard Features:
- **Data Ingestion:** Select any existing raw file from `data/raw/` or upload custom telemetry CSVs.
- **Dynamic Configuration:** Interactive sliders for missingness threshold, collinearity limit, scaler selection, and outlier strategies.
- **Visual Analytics:** Real-time rendering of high-res missingness charts, outlier boxplots, correlation heatmaps, and feature-target bar charts.
- **Artifact Downloads:** One-click downloads for `train.csv`, `val.csv`, `test.csv`, `preprocessor.joblib`, and `cleaning_report.json`.

---

## 🧪 Automated Testing

Execute the test suite to verify leak-free splitting, EV domain guard enforcement, zero NaNs in output splits, and preprocessor reloadability:

```bash
python -m pytest tests/test_pipeline.py -v
```

All 8 comprehensive unit and integration tests pass:
- `test_timestamp_parsing`: Standardizes mixed `+00:00` and `Z` formats to UTC.
- `test_ev_domain_guards`: Validates speed limits, non-negative cell delta, and SOC.
- `test_high_missing_column_dropping`: Confirms high missingness pruning.
- `test_collinear_feature_elimination`: Validates collinear pair removal.
- `test_zero_data_leakage_and_splits`: Asserts imputers/scalers fit strictly on `X_train`.
- `test_zero_nans_in_final_splits`: Guarantees 0 NaNs in `train.csv`, `val.csv`, `test.csv`.
- `test_preprocessor_reloadability`: Reloads `preprocessor.joblib` and transforms unseen batches.
- `test_cleaning_report_structure`: Validates structured JSON audit keys.

---

## 📊 Preprocessor Reloadability Example

```python
import pandas as pd
from src.preprocessing import VRPPreprocessor

# Reload the fitted preprocessor
preprocessor = VRPPreprocessor.load("data/processed/preprocessor.joblib")

# Transform new unseen telemetry data
new_batch = pd.read_csv("data/raw/new_telemetry.csv")
transformed_features = preprocessor.transform(new_batch)

print("Transformed shape:", transformed_features.shape)
print("NaN count:", transformed_features.isna().sum().sum())
```
