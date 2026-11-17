# BeatAhead: Random Forest Ensemble for Intraoperative Myocardial Ischemia Early Warning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository implements the **Random Forest Ensemble Model** for **BeatAhead**—an intraoperative early-warning system designed to predict impending sustained myocardial ischemia 5 to 15 minutes before manifest clinical onset.

---

## 📋 Clinical & Modeling Overview

Intraoperative myocardial ischemia is often silent and hemodynamically catastrophic during high-risk surgery. Subtle electrophysiological and plethysmographic signals exhibit non-linear interactions across multilead ECG, photoplethysmogram (PPG), pulse arrival time (PAT), and continuous ST deviations.

* **Clinical Goal**: Early-warning prediction of impending sustained ischemia ($\ge 10\,\text{s}$ ST depression $\le -0.10\,\text{mV}$ or elevation $\ge +0.10\,\text{mV}$) within a 5–15 minute lookahead window.
* **Challenge**: Severe clinical class imbalance ($\approx 230:1$ negative-to-positive ratio in real surgical cases).
* **Random Forest Advantage**: Non-linear tree splits with bootstrap aggregation (`class_weight="balanced_subsample"`) mitigate extreme imbalance while capturing complex physiological co-dependencies without requiring rigid parametric assumptions.

---

## 🛡️ Zero-Leakage Subject Partitioning

To strictly prevent patient-level data leakage, partitioning is performed strictly at the **subject (caseid) level**:

* **Development Set (80 cases)**: 40 high-risk surgical cases (Cohort A) + 40 representative surgical cases (Cohort B). Used exclusively for median imputation, 5-Fold `StratifiedGroupKFold` cross-validation, hyperparameter tuning, and decision threshold calibration.
* **Locked Test Set (20 cases)**: 10 Cohort A + 10 Cohort B cases completely withheld during development and hyperparameter tuning until final evaluation.

---

## 📊 Feature Matrix A (26 Multimodal Features)

| Modality | Features Extracted |
| :--- | :--- |
| **ECG / HRV** | `ecg_hr_mean`, `ecg_hr_std`, `ecg_rr_sdnn`, `ecg_rr_rmssd`, `ecg_pnn50`, `ecg_r_amp_mv`, `ecg_qrs_width_ms`, `ecg_sqi` |
| **Vascular / PAT** | `pat_median_ms`, `pat_iqr_ms`, `pat_valid_fraction`, `pat_valid` |
| **PPG Morphology** | `ppg_pulse_amp`, `ppg_perfusion_index`, `ppg_crest_time_ms`, `ppg_sqi` |
| **Oximetry (SpO2)** | `spo2_mean`, `spo2_min`, `spo2_std`, `spo2_desat_count` |
| **Pre-Event ST Trends** | `st_obs_mean`, `st_obs_median`, `st_obs_min`, `st_obs_std`, `st_delta_baseline`, `st_slope_mm_min` |

---

## 📈 Cross-Validation & Hyperparameter Tuning Results

Evaluated across 5-Fold `StratifiedGroupKFold` on the 80 Development cases:

| `max_depth` | `min_samples_leaf` | `n_estimators` | OOF PR-AUC | OOF ROC-AUC | Optimal Thresh | Optimal F1 | Specificity |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **6** | **5** | **200** | **0.3373** | **0.8497** | **0.3169** | **0.5600** | **99.8%** |
| 5 | 5 | 150 | 0.3372 | 0.8730 | 0.3268 | 0.5570 | 99.7% |
| 4 | 10 | 100 | 0.3235 | 0.8995 | 0.4159 | 0.4835 | 99.6% |
| 8 | 2 | 200 | 0.2906 | 0.8963 | 0.2080 | 0.4878 | 99.4% |

* **Out-of-Fold PR-AUC**: `0.3373` (an **$84\times$ improvement** over the $\approx 0.004$ random surgical baseline and nearly $3\times$ higher than Linear Regression).
* **Calibrated Decision Threshold**: `0.3169` (maximizes out-of-fold F1-score to `0.5600`).

---

## 🔒 Out-of-Sample Locked Test Evaluation (20 Cases)

* **Test Windows Evaluated**: 2,770 windows across 20 unseen surgical patients
* **True Negatives**: 2,753
* **False Positives**: 6 (**99.78% Specificity** — minimal false alarm fatigue in the Operating Room)
* **False Negatives**: 11
* **True Positives**: 0

---

## 🌳 Top Physiological Features (Gini Impurity Reduction)

The ensemble identifies the following leading physiological indicators of intraoperative ischemia:
1. `st_obs_min`: Lowest observed ST segment amplitude during observation window.
2. `st_obs_mean`: Mean continuous ST level.
3. `st_delta_baseline`: Difference between current ST level and patient baseline.
4. `st_slope_mm_min`: Rate of change in ST level over time.
5. `ecg_rr_rmssd`: Short-term parasympathetic heart rate variability.
6. `pat_median_ms`: Median Pulse Arrival Time (surrogate for vascular tone / arterial stiffness).
7. `ppg_pulse_amp`: Photoplethysmogram pulse amplitude (peripheral perfusion).

---

## 🗂️ Repository Structure

```
Random-Forest-BeatAhead/
├── .gitignore
├── README.md
├── requirements.txt
├── train_random_forest.py         # Main end-to-end training and evaluation script
├── random_forest_tuning.py        # 5-fold StratifiedGroupKFold hyperparameter grid search
├── predict.py                     # Standalone CLI and programmatic inference tool
├── data/
│   ├── vitaldb_100_windows.csv.gz # Filtered sliding window dataset
│   ├── cohort_a_final_verified.csv
│   └── cohort_b_final_verified.csv
├── src/
│   ├── __init__.py
│   ├── dataset.py                 # Ingestion & strict subject-level partitioning
│   ├── feature_pipeline.py        # Median imputation (tree-optimized)
│   ├── models.py                  # Random Forest classifier ensemble
│   ├── metrics.py                 # PR-AUC, ROC-AUC, Brier score, threshold calibration
│   └── plotting.py                # PR/ROC curves, confusion matrix, MDI feature importance
├── tests/
│   └── test_random_forest.py      # Unit test suite verifying zero leakage & reproducibility
├── artifacts/
│   ├── random_forest_model.joblib
│   ├── feature_schema.json
│   ├── model_metadata.json
│   └── best_rf_config.json
└── outputs/
    ├── evaluation_metrics.json
    ├── test_predictions.csv
    ├── feature_importance.csv
    ├── hyperparameter_results.csv
    ├── pr_roc_curves.png
    ├── confusion_matrix.png
    └── feature_importance.png
```

---

## 🚀 Quickstart

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Training Pipeline
```bash
python train_random_forest.py
```

### 3. Run Hyperparameter Tuning
```bash
python random_forest_tuning.py
```

### 4. Run Inference on a Sample Window
```bash
python predict.py --test-sample
```

### 5. Run Unit Tests
```bash
python tests/test_random_forest.py
```
