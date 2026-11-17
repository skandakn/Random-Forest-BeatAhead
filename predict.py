"""
predict.py - Standalone inference script for BeatAhead Random Forest model.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = BASE_DIR / "artifacts" / "random_forest_model.joblib"
METADATA_PATH = BASE_DIR / "artifacts" / "model_metadata.json"


def load_model():
    """Loads the serialized trained Random Forest model."""
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at {ARTIFACT_PATH}. Run train_random_forest.py first.")
    return joblib.load(ARTIFACT_PATH)


def predict_sample(sample_data: Union[Dict[str, float], pd.DataFrame]) -> Dict[str, Any]:
    """Runs inference on a single feature record or dataframe."""
    model = load_model()
    if isinstance(sample_data, dict):
        df = pd.DataFrame([sample_data])
    else:
        df = sample_data.copy()

    # Validate feature presence
    missing = [f for f in model.feature_cols if f not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required features: {missing}")

    probs = model.predict_proba(df)
    preds = model.predict(df)
    threshold = model.threshold

    results = []
    for p, y in zip(probs, preds):
        if p >= threshold:
            risk_tier = "CRITICAL / HIGH RISK (Impending Ischemia)"
        elif p >= threshold * 0.5:
            risk_tier = "ELEVATED / MODERATE RISK (Monitor Closely)"
        else:
            risk_tier = "STABLE / LOW RISK"

        results.append({
            "probability": float(round(p, 5)),
            "prediction": int(y),
            "threshold": float(round(threshold, 5)),
            "risk_tier": risk_tier,
        })

    return results[0] if len(results) == 1 else {"predictions": results}


def main():
    parser = argparse.ArgumentParser(description="BeatAhead Random Forest Inference CLI")
    parser.add_argument("--input", type=str, help="Path to input JSON or CSV file with feature values.")
    parser.add_argument("--test-sample", action="store_true", help="Run inference on a synthetic test sample.")
    args = parser.parse_args()

    if args.test_sample or not args.input:
        print("Running inference on sample physiological observation...")
        model = load_model()
        sample = {f: 0.0 for f in model.feature_cols}
        sample["ecg_hr_mean"] = 84.5
        sample["st_obs_mean"] = -0.15  # ST depression
        sample["st_delta_baseline"] = -0.12
        sample["st_slope_mm_min"] = -0.05
        sample["spo2_mean"] = 98.2
        sample["pat_median_ms"] = 215.0

        res = predict_sample(sample)
        print(json.dumps(res, indent=2))
        return

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file {input_path} does not exist.")
        sys.exit(1)

    if input_path.suffix == ".json":
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        res = predict_sample(data)
    else:
        df = pd.read_csv(input_path)
        res = predict_sample(df)

    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
