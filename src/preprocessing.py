"""
preprocessing.py
================
Preprocessing, stratified splitting, and transformation pipelines for Predictive Maintenance.

Key components:
1. Stratified Train/Test Split: Maintains exact 3.39% minority failure proportion in both folds.
2. Leakage-Proof Pipelines: All scalers and encoders fit strictly on X_train.
3. Model-specific transforms:
   - Scaled Pipeline: OneHotEncoder + StandardScaler (for Logistic Regression).
   - Tree Pipeline: OneHotEncoder + Passthrough for continuous physical units (for Random Forest, XGBoost, Isolation Forest).
4. Full Inference Pipeline: Bundles MachineFeatureEngineer + ColumnTransformer so raw 6-feature inputs
   can be passed directly during deployment/Streamlit inference.
"""

import os
import sys
import json
import logging
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Ensure package importability
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.feature_engineering import MachineFeatureEngineer, ENGINEERED_FEATURE_NAMES, BASE_FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CATEGORICAL_FEATURES = ["Type"]
NUMERICAL_FEATURES_BASE = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
ALL_NUMERICAL_FEATURES = NUMERICAL_FEATURES_BASE + ENGINEERED_FEATURE_NAMES

MODELS_DIR = "models"
OUTPUT_METRICS_DIR = "outputs/metrics"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_METRICS_DIR, exist_ok=True)


def split_data(
    df: pd.DataFrame,
    target_col: str = "Machine failure",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
    """
    Performs a stratified train-test split to strictly preserve the target class ratio.

    Args:
        df: Raw input DataFrame.
        target_col: Target column name.
        test_size: Proportion of dataset allocated to the hold-out test set (default 0.2).
        random_state: Fixed random seed for complete reproducibility.

    Returns:
        X_train, X_test, y_train, y_test, split_metadata
    """
    logger.info("Executing stratified train-test split...")

    # Drop identifiers and failure mode leakage flags from X
    feature_cols = [
        col for col in df.columns
        if col not in ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF", target_col]
    ]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Compute audit metadata
    n_train = len(y_train)
    n_test = len(y_test)
    train_fail_cnt = int(y_train.sum())
    train_norm_cnt = n_train - train_fail_cnt
    test_fail_cnt = int(y_test.sum())
    test_norm_cnt = n_test - test_fail_cnt

    train_fail_rate = round(float(train_fail_cnt / n_train * 100), 2)
    test_fail_rate = round(float(test_fail_cnt / n_test * 100), 2)

    split_metadata = {
        "random_state": random_state,
        "test_size": test_size,
        "training_set": {
            "total_samples": n_train,
            "normal_count": train_norm_cnt,
            "failure_count": train_fail_cnt,
            "failure_rate_pct": train_fail_rate,
            "imbalance_ratio": f"{round(train_norm_cnt / train_fail_cnt, 2)}:1",
        },
        "test_set": {
            "total_samples": n_test,
            "normal_count": test_norm_cnt,
            "failure_count": test_fail_cnt,
            "failure_rate_pct": test_fail_rate,
            "imbalance_ratio": f"{round(test_norm_cnt / test_fail_cnt, 2)}:1",
        },
        "stratification_rationale": (
            "Because machine failures account for only 3.39% of observations (imbalance ratio 28.5:1), "
            "standard random splitting risks severe sampling variance where the test set could have significantly "
            "fewer or more failure events. Stratified splitting enforces an identical 3.39% failure incidence in both "
            "training and test partitions, ensuring realistic, unbiased evaluation."
        ),
    }

    metadata_path = os.path.join(OUTPUT_METRICS_DIR, "train_test_split_info.json")
    with open(metadata_path, "w") as f:
        json.dump(split_metadata, f, indent=4)
    logger.info(f"Saved train/test split metadata to: {metadata_path}")

    return X_train, X_test, y_train, y_test, split_metadata


def build_column_transformer(scale_numerical: bool = False) -> ColumnTransformer:
    """
    Builds a ColumnTransformer that encodes categorical 'Type' and handles numerical features.

    Args:
        scale_numerical: If True, uses StandardScaler on numerical features (for linear models).
                         If False, passes numerical features through unscaled (for tree models).
    """
    categorical_transformer = OneHotEncoder(
        categories=[["H", "L", "M"]],  # Fixed category ordering for absolute reproducibility
        handle_unknown="ignore",
        sparse_output=False,
    )

    if scale_numerical:
        numerical_transformer = StandardScaler()
    else:
        numerical_transformer = "passthrough"

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("num", numerical_transformer, ALL_NUMERICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def build_full_preprocessing_pipeline(scale_numerical: bool = False) -> Pipeline:
    """
    Constructs an end-to-end preprocessing pipeline combining:
    1. MachineFeatureEngineer (computes physical features from raw 6 features)
    2. ColumnTransformer (encodes Type and scales/passes numericals)
    """
    col_transformer = build_column_transformer(scale_numerical=scale_numerical)
    pipeline = Pipeline(
        steps=[
            ("feature_engineer", MachineFeatureEngineer(drop_leakage_cols=True)),
            ("preprocessor", col_transformer),
        ]
    )
    return pipeline


def fit_and_save_preprocessors(X_train: pd.DataFrame) -> Tuple[Pipeline, Pipeline]:
    """
    Fits both scaled and unscaled end-to-end preprocessing pipelines on X_train only,
    and serializes the primary preprocessing artifact to models/preprocessing_pipeline.pkl.
    """
    logger.info("Fitting preprocessing pipelines on training set...")

    # Tree preprocessor (primary for RF, XGBoost, Isolation Forest)
    tree_pipeline = build_full_preprocessing_pipeline(scale_numerical=False)
    tree_pipeline.fit(X_train)

    # Linear preprocessor (for Logistic Regression)
    linear_pipeline = build_full_preprocessing_pipeline(scale_numerical=True)
    linear_pipeline.fit(X_train)

    # Save primary preprocessing pipeline
    primary_pipeline_path = os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl")
    joblib.dump(tree_pipeline, primary_pipeline_path)
    logger.info(f"Saved primary preprocessing pipeline to: {primary_pipeline_path}")

    # Also save linear preprocessor for Logistic Regression
    linear_pipeline_path = os.path.join(MODELS_DIR, "preprocessing_pipeline_scaled.pkl")
    joblib.dump(linear_pipeline, linear_pipeline_path)
    logger.info(f"Saved scaled preprocessing pipeline to: {linear_pipeline_path}")

    return tree_pipeline, linear_pipeline


if __name__ == "__main__":
    from src.data_loader import load_data

    df = load_data()
    X_train, X_test, y_train, y_test, meta = split_data(df)
    tree_pipe, linear_pipe = fit_and_save_preprocessors(X_train)

    X_train_trans = tree_pipe.transform(X_train)
    print("\n--- Preprocessing Verification ---")
    print("X_train shape (raw input):", X_train.shape)
    print("X_train transformed shape:", X_train_trans.shape)
    print("Split metadata:")
    print(json.dumps(meta, indent=2))
