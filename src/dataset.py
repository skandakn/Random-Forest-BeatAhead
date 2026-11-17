"""
src/dataset.py - Dataset ingestion, filtering, and leakage-free subject partitioning for Random Forest.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

FEATURES_MATRIX_A: List[str] = [
    'ecg_hr_mean', 'ecg_hr_std', 'ecg_rr_sdnn', 'ecg_rr_rmssd', 'ecg_pnn50',
    'ecg_r_amp_mv', 'ecg_qrs_width_ms', 'ecg_sqi',
    'pat_median_ms', 'pat_iqr_ms', 'pat_valid_fraction', 'pat_valid',
    'ppg_pulse_amp', 'ppg_perfusion_index', 'ppg_crest_time_ms', 'ppg_sqi',
    'spo2_mean', 'spo2_min', 'spo2_std', 'spo2_desat_count',
    'st_obs_mean', 'st_obs_median', 'st_obs_min', 'st_obs_std',
    'st_delta_baseline', 'st_slope_mm_min'
]

FEATURES_MATRIX_B: List[str] = [
    'ecg_hr_mean', 'ecg_hr_std', 'ecg_rr_sdnn', 'ecg_rr_rmssd', 'ecg_pnn50',
    'ecg_r_amp_mv', 'ecg_qrs_width_ms', 'ecg_sqi',
    'pat_median_ms', 'pat_iqr_ms', 'pat_valid_fraction', 'pat_valid',
    'ppg_pulse_amp', 'ppg_perfusion_index', 'ppg_crest_time_ms', 'ppg_sqi',
    'spo2_mean', 'spo2_min', 'spo2_std', 'spo2_desat_count'
]


def load_clean_dataset(data_dir: Path) -> pd.DataFrame:
    """Loads and filters the BeatAhead window dataset.

    Applies Signal Quality Index (SQI) PASS filter and keeps valid binary labels (0 or 1).
    """
    csv_path = data_dir / "vitaldb_100_windows.csv.gz"
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path, compression="gzip")
    df_clean = df[(df["sqi_status"] == "PASS") & (df["target_label"].isin([0, 1]))].copy()
    df_clean["target_label"] = df_clean["target_label"].astype(int)
    return df_clean


def create_subject_level_partitions(
    df_clean: pd.DataFrame, data_dir: Path, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int], List[int]]:
    """Creates a strict two-tier subject-level partition across all 100 verified cases.

    - 80 Development cases (40 Cohort A, 40 Cohort B; 6 positive cases)
    - 20 Locked Test cases (10 Cohort A, 10 Cohort B; 2 positive cases)
    Zero patient overlap between development and test partitions (zero data leakage).
    """
    cohort_a_path = data_dir / "cohort_a_final_verified.csv"
    cohort_b_path = data_dir / "cohort_b_final_verified.csv"

    if not cohort_a_path.exists() or not cohort_b_path.exists():
        raise FileNotFoundError("Cohort audit verification CSVs missing from data directory.")

    df_a = pd.read_csv(cohort_a_path)
    df_b = pd.read_csv(cohort_b_path)
    cids_a = df_a["caseid"].tolist()
    cids_b = df_b["caseid"].tolist()

    pos_cases_set = set(df_clean[df_clean["target_label"] == 1]["caseid"].unique())

    pos_a = [c for c in cids_a if c in pos_cases_set]
    neg_a = [c for c in cids_a if c not in pos_cases_set]
    pos_b = [c for c in cids_b if c in pos_cases_set]
    neg_b = [c for c in cids_b if c not in pos_cases_set]

    rng = np.random.RandomState(seed)

    shuffled_pos_a = rng.permutation(pos_a)
    shuffled_neg_a = rng.permutation(neg_a)
    shuffled_pos_b = rng.permutation(pos_b)
    shuffled_neg_b = rng.permutation(neg_b)

    test_pos_a = shuffled_pos_a[:1].tolist()
    dev_pos_a = shuffled_pos_a[1:].tolist()
    test_neg_a = shuffled_neg_a[:9].tolist()
    dev_neg_a = shuffled_neg_a[9:].tolist()

    test_pos_b = shuffled_pos_b[:1].tolist()
    dev_pos_b = shuffled_pos_b[1:].tolist()
    test_neg_b = shuffled_neg_b[:9].tolist()
    dev_neg_b = shuffled_neg_b[9:].tolist()

    test_cases = sorted(test_pos_a + test_neg_a + test_pos_b + test_neg_b)
    dev_cases = sorted(dev_pos_a + dev_neg_a + dev_pos_b + dev_neg_b)

    # Verification: Mutual exclusivity and partition completeness
    assert len(set(dev_cases).intersection(set(test_cases))) == 0, "Leakage: Dev and Test overlap!"
    assert len(dev_cases) + len(test_cases) == 100, f"Expected 100 cases, got {len(dev_cases) + len(test_cases)}"
    assert len(test_cases) == 20, f"Expected 20 test cases, got {len(test_cases)}"
    assert len(dev_cases) == 80, f"Expected 80 dev cases, got {len(dev_cases)}"

    df_dev = df_clean[df_clean["caseid"].isin(dev_cases)].copy().reset_index(drop=True)
    df_test = df_clean[df_clean["caseid"].isin(test_cases)].copy().reset_index(drop=True)

    return df_dev, df_test, dev_cases, test_cases
