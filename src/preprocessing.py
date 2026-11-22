"""
Preprocessing module implementing scikit-learn ColumnTransformer and Pipeline
for imputing, scaling, and encoding numeric and categorical features.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

from src.config import PREPROCESSOR_JOB_PATH, VALID_SCALERS

logger = logging.getLogger(__name__)


def get_scaler(scaler_type: str = "standard"):
    """
    Returns an uninstantiated scaler based on requested type.
    """
    scaler_lower = str(scaler_type).lower()
    if scaler_lower == "standard":
        return StandardScaler()
    elif scaler_lower == "minmax":
        return MinMaxScaler()
    elif scaler_lower == "robust":
        return RobustScaler()
    else:
        raise ValueError(f"Unsupported scaler '{scaler_type}'. Choose from {VALID_SCALERS}")


class VRPPreprocessor(BaseEstimator, TransformerMixin):
    """
    Production-grade preprocessor combining imputing, scaling, and one-hot encoding.
    Ensures zero data leakage by fitting strictly on X_train.
    Preserves clean feature names in output DataFrames.
    """

    def __init__(self, scaler_type: str = "standard"):
        self.scaler_type = scaler_type
        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.feature_names_out_: List[str] = []
        self.pipeline_: Optional[ColumnTransformer] = None

    def _prepare_df(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Converts boolean types to strings and handles dtypes consistently.
        """
        df_out = X.copy()
        for col in df_out.columns:
            if pd.api.types.is_bool_dtype(df_out[col]):
                df_out[col] = df_out[col].astype(str)
            elif df_out[col].dtype == object:
                # Replace whitespace or null strings with NaN
                df_out[col] = df_out[col].replace(r"^\s*$", np.nan, regex=True)
        return df_out

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fits imputers, scalers, and encoders strictly on training data.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        df_prepared = self._prepare_df(X)

        # Detect column types
        self.numeric_cols_ = list(df_prepared.select_dtypes(include=[np.number]).columns)
        self.categorical_cols_ = [c for c in df_prepared.columns if c not in self.numeric_cols_]

        transformers = []

        # Numeric transformer
        if self.numeric_cols_:
            num_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", get_scaler(self.scaler_type))
            ])
            transformers.append(("num", num_pipe, self.numeric_cols_))

        # Categorical transformer
        if self.categorical_cols_:
            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])
            transformers.append(("cat", cat_pipe, self.categorical_cols_))

        self.pipeline_ = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False
        )

        self.pipeline_.fit(df_prepared, y)

        # Retrieve feature names out
        try:
            self.feature_names_out_ = list(self.pipeline_.get_feature_names_out())
        except Exception:
            # Fallback for older scikit-learn versions
            names = list(self.numeric_cols_)
            if self.categorical_cols_:
                cat_encoder = self.pipeline_.named_transformers_["cat"].named_steps["encoder"]
                cat_names = cat_encoder.get_feature_names_out(self.categorical_cols_)
                names.extend(list(cat_names))
            self.feature_names_out_ = names

        logger.info(
            f"Fitted VRPPreprocessor: {len(self.numeric_cols_)} numeric cols, "
            f"{len(self.categorical_cols_)} categorical cols -> {len(self.feature_names_out_)} total features."
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms input DataFrame using fitted transformers and returns
        a clean pandas DataFrame with legible column names.
        """
        if self.pipeline_ is None:
            raise RuntimeError("Preprocessor has not been fitted yet! Call .fit() first.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        df_prepared = self._prepare_df(X)

        # Ensure all columns expected by pipeline exist (missing columns filled with NaN)
        expected_cols = self.numeric_cols_ + self.categorical_cols_
        for col in expected_cols:
            if col not in df_prepared.columns:
                df_prepared[col] = np.nan

        arr = self.pipeline_.transform(df_prepared)
        
        # Build pandas DataFrame with preserved feature names
        df_transformed = pd.DataFrame(arr, columns=self.feature_names_out_, index=X.index)
        return df_transformed

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fits and transforms in a single call.
        """
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return list(self.feature_names_out_)

    def save(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """
        Serializes the fitted preprocessor to disk using joblib.
        """
        save_path = Path(filepath or PREPROCESSOR_JOB_PATH)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, save_path)
        logger.info(f"Saved fitted preprocessor to {save_path}")
        return save_path

    @classmethod
    def load(cls, filepath: Optional[Union[str, Path]] = None) -> "VRPPreprocessor":
        """
        Loads a serialized preprocessor from disk.
        """
        load_path = Path(filepath or PREPROCESSOR_JOB_PATH)
        if not load_path.exists():
            raise FileNotFoundError(f"Preprocessor file not found at {load_path}")
        loaded = joblib.load(load_path)
        logger.info(f"Loaded preprocessor from {load_path}")
        return loaded
