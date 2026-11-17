"""
src/models.py - Random Forest model definition for BeatAhead ischemia prediction.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from .feature_pipeline import TabularPreprocessor


class BeatAheadRandomForestClassifier:
    """End-to-end Random Forest ensemble for early warning of intraoperative myocardial ischemia.

    Combines:
    1. Median Imputation fitted strictly on development set
    2. Tuned Random Forest with balanced subsampling (mitigates 230:1 clinical class imbalance)
    3. Calibrated decision threshold calibrated on out-of-fold predictions
    """

    def __init__(
        self,
        feature_cols: List[str],
        n_estimators: int = 200,
        max_depth: int = 6,
        min_samples_leaf: int = 5,
        max_features: str = "sqrt",
        class_weight: str = "balanced_subsample",
        random_state: int = 42,
        threshold: float = 0.5,
        n_jobs: int = -1,
    ):
        self.feature_cols = list(feature_cols)
        self.preprocessor = TabularPreprocessor(self.feature_cols)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.threshold = threshold
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "BeatAheadRandomForestClassifier":
        """Fits imputer and random forest ensemble on training partition."""
        X_imp = self.preprocessor.fit_transform(X)
        self.model.fit(X_imp, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts positive class probability (P(ischemia=1))."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        X_imp = self.preprocessor.transform(X)
        return self.model.predict_proba(X_imp)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Predicts binary classification based on decision threshold."""
        thresh = threshold if threshold is not None else self.threshold
        probs = self.predict_proba(X)
        return (probs >= thresh).astype(int)

    def get_feature_importances(self) -> pd.DataFrame:
        """Returns MDI (Mean Decrease in Impurity) feature importances."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        importances = self.model.feature_importances_
        df_imp = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df_imp
