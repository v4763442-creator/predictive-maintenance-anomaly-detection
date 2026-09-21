"""
feature_engineering.py
======================
Domain-driven physical feature engineering for the AI4I 2020 Predictive Maintenance dataset.

Calculates mechanically and thermodynamically justified features derived from first principles:
1. Temperature Difference (Process Temp - Air Temp): Captures heat dissipation gradient (HDF mechanism).
2. Temperature Ratio (Process Temp / Air Temp): Captures relative thermal stress.
3. Mechanical Power Proxy (Torque * Rotational Speed * 2*pi/60): Direct measure of motor power in Watts (PWF mechanism).
4. Overstrain Proxy (Torque * Tool Wear): Compound stress index combining mechanical load and tool degradation (OSF mechanism).

Strictly isolates and drops identifier columns (UDI, Product ID) and failure mode flags (TWF, HDF, PWF, OSF, RNF)
to prevent target leakage.
"""

import logging
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Mandatory raw operational features required for feature engineering
BASE_FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Engineered feature names
ENGINEERED_FEATURE_NAMES = [
    "Temp_Difference_K",
    "Temp_Ratio",
    "Power_W",
    "Overstrain_Torque_Wear",
]

COLUMNS_TO_DROP_LEAKAGE = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]


class MachineFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that constructs domain-specific physical features
    from raw machine operating parameters.
    """

    def __init__(self, drop_leakage_cols: bool = True):
        self.drop_leakage_cols = drop_leakage_cols
        self.feature_names_out_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y=None):
        """Fit method (no-op as physical features have static deterministic formulas)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies domain-justified engineering transformations to the input dataframe.

        Args:
            X: Input dataframe containing machine operating parameters.

        Returns:
            pd.DataFrame: Augmented dataframe with engineered features.
        """
        # Create a shallow copy to prevent modifying original data in-place
        df = X.copy()

        # Check for presence of essential base features
        missing = [col for col in BASE_FEATURE_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing mandatory base features for feature engineering: {missing}")

        # 1. Temperature Difference (Process - Air) in Kelvin
        # Physical rationale: Quantifies heat dissipation gradient. Heat Dissipation Failure (HDF)
        # occurs when ambient air is high and dissipation delta falls below critical threshold (~8.9K).
        df["Temp_Difference_K"] = df["Process temperature [K]"] - df["Air temperature [K]"]

        # 2. Temperature Ratio (Process / Air)
        # Physical rationale: Dimensionless thermodynamic ratio indicating relative thermal expansion.
        df["Temp_Ratio"] = df["Process temperature [K]"] / df["Air temperature [K]"]

        # 3. Mechanical Power Proxy in Watts (P = tau * omega)
        # Rotational speed (rpm) converted to angular velocity (rad/s): omega = rpm * 2 * pi / 60
        # Power in Watts = Torque [Nm] * omega [rad/s]
        # Physical rationale: Power Failure (PWF) occurs when motor power falls outside normal operational envelope
        # (< 2000 W or > 9000 W).
        omega = df["Rotational speed [rpm]"] * (2 * np.pi / 60.0)
        df["Power_W"] = df["Torque [Nm]"] * omega

        # 4. Overstrain Proxy (Torque [Nm] * Tool Wear [min])
        # Physical rationale: Overstrain Failure (OSF) occurs under heavy cutting load when tool wear
        # has substantially accumulated, causing mechanical shear failure.
        df["Overstrain_Torque_Wear"] = df["Torque [Nm]"] * df["Tool wear [min]"]

        # Drop leakage columns and identifier columns if present
        if self.drop_leakage_cols:
            cols_to_drop = [col for col in COLUMNS_TO_DROP_LEAKAGE if col in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)

        # Drop target if present in input feature matrix
        if "Machine failure" in df.columns:
            df = df.drop(columns=["Machine failure"])

        self.feature_names_out_ = list(df.columns)
        return df

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Returns feature names after transformation."""
        if self.feature_names_out_ is None:
            return BASE_FEATURE_COLUMNS + ENGINEERED_FEATURE_NAMES
        return self.feature_names_out_


def engineer_features(df: pd.DataFrame, drop_leakage_cols: bool = True) -> pd.DataFrame:
    """
    Convenience function to engineer features on a DataFrame.
    """
    fe = MachineFeatureEngineer(drop_leakage_cols=drop_leakage_cols)
    return fe.transform(df)


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.data_loader import load_data
    raw_df = load_data()
    engineered_df = engineer_features(raw_df)
    print("Engineered DataFrame shape:", engineered_df.shape)
    print("Features:", list(engineered_df.columns))
    print("\nSample engineered statistics:")
    print(engineered_df[ENGINEERED_FEATURE_NAMES].describe().T[["mean", "std", "min", "50%", "max"]])
