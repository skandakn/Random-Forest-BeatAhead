"""
src/feature_pipeline.py - Feature imputation pipeline for Random Forest models.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


class TabularPreprocessor:
    """Preprocesses physiological feature matrices for tree-based ensemble models.

    Tree ensembles are scale-invariant, so standard scaling is omitted, but median
    imputation is fitted strictly on the development partition to handle sensor dropouts.
    """

    def __init__(self, feature_cols: List[str]):
        self.feature_cols = list(feature_cols)
        self.imputer = SimpleImputer(strategy="median")
        self.is_fitted = False

    def fit(self, X: pd.DataFrame) -> "TabularPreprocessor":
        """Fits imputer exclusively on training data."""
        X_mat = X[self.feature_cols].copy()
        self.imputer.fit(X_mat)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Imputes missing features using training set median statistics."""
        if not self.is_fitted:
            raise RuntimeError("TabularPreprocessor must be fitted before transforming data.")
        X_mat = X[self.feature_cols].copy()
        return self.imputer.transform(X_mat)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fits on training data and returns imputed array."""
        return self.fit(X).transform(X)
