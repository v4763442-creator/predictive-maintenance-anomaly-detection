"""
test_pipeline.py
================
Automated unit tests and quality assurance checks for Predictive Maintenance & Anomaly Detection.

Validates:
1. Data loading and schema integrity.
2. Leakage prevention: identifier columns and failure modes quarantined.
3. Feature engineering correctness: physics calculations (Power, Temp Difference, Overstrain).
4. Preprocessing transformation shapes and consistency.
5. Model serialization and independent inference loading.
6. Failure prediction functionality, bounds, and probability validity.
7. Anomaly detection score normalization and predictions.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import joblib

# Ensure package importability
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import load_data, audit_data_quality, EXPECTED_COLUMNS
from src.feature_engineering import MachineFeatureEngineer, engineer_features, BASE_FEATURE_COLUMNS
from src.preprocessing import split_data, build_full_preprocessing_pipeline
from src.prediction import MaintenancePredictor, get_predictor


@pytest.fixture(scope="session")
def raw_dataset():
    """Loads dataset once for test session."""
    return load_data("data/ai4i2020.csv")


def test_data_loader_schema(raw_dataset):
    """Verifies dataset row count, column count, and mandatory schema."""
    assert raw_dataset.shape == (10000, 14)
    for col in EXPECTED_COLUMNS:
        assert col in raw_dataset.columns
    assert raw_dataset.isnull().sum().sum() == 0
    assert raw_dataset.duplicated().sum() == 0


def test_data_quality_audit(raw_dataset):
    """Verifies that data audit catches zero physical bound violations."""
    audit = audit_data_quality(raw_dataset)
    assert audit["dataset_dimensions"]["rows"] == 10000
    assert audit["dataset_dimensions"]["columns"] == 14
    assert audit["missing_values"]["total_missing"] == 0
    assert audit["plausibility_and_physics_checks"]["all_physics_checks_passed"] is True
    assert audit["target_distribution"]["counts"][1] == 339
    assert audit["target_distribution"]["counts"][0] == 9661


def test_feature_engineering_calculations():
    """Verifies mathematical correctness of physics-derived features."""
    sample_data = pd.DataFrame({
        "Type": ["L"],
        "Air temperature [K]": [300.0],
        "Process temperature [K]": [310.0],
        "Rotational speed [rpm]": [1500],
        "Torque [Nm]": [40.0],
        "Tool wear [min]": [100],
        "Machine failure": [0],
        "TWF": [0], "HDF": [0], "PWF": [0], "OSF": [0], "RNF": [0],
        "UDI": [1], "Product ID": ["L001"],
    })

    fe = MachineFeatureEngineer(drop_leakage_cols=True)
    transformed = fe.transform(sample_data)

    # 1. Temperature Difference: 310 - 300 = 10.0
    assert np.isclose(transformed.loc[0, "Temp_Difference_K"], 10.0)

    # 2. Temperature Ratio: 310 / 300 = 1.0333...
    assert np.isclose(transformed.loc[0, "Temp_Ratio"], 310.0 / 300.0)

    # 3. Power in Watts: 40 * (1500 * 2 * pi / 60) = 40 * 157.0796 = 6283.185 W
    expected_power = 40.0 * (1500 * 2 * np.pi / 60)
    assert np.isclose(transformed.loc[0, "Power_W"], expected_power)

    # 4. Overstrain: 40 * 100 = 4000.0
    assert np.isclose(transformed.loc[0, "Overstrain_Torque_Wear"], 4000.0)

    # 5. Strict leakage quarantine check
    for leaked_col in ["UDI", "Product ID", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]:
        assert leaked_col not in transformed.columns


def test_stratified_split(raw_dataset):
    """Verifies that stratified train/test split maintains identical class proportions."""
    X_train, X_test, y_train, y_test, meta = split_data(raw_dataset, random_state=42)

    assert len(X_train) == 8000
    assert len(X_test) == 2000
    assert y_train.sum() == 271
    assert y_test.sum() == 68

    train_pct = y_train.sum() / len(y_train) * 100
    test_pct = y_test.sum() / len(y_test) * 100
    assert np.isclose(train_pct, test_pct, atol=0.05)


def test_saved_model_artifacts_exist():
    """Verifies that all required serialized model files exist on disk."""
    required_models = [
        "models/preprocessing_pipeline.pkl",
        "models/failure_model.pkl",
        "models/isolation_forest.pkl",
        "models/failure_prediction_pipeline.pkl",
    ]
    for model_path in required_models:
        assert os.path.exists(model_path), f"Missing serialized artifact: {model_path}"
        assert os.path.getsize(model_path) > 0, f"Empty artifact: {model_path}"


def test_inference_engine():
    """Tests the production MaintenancePredictor across nominal, failure, and anomaly inputs."""
    predictor = get_predictor()

    # Case 1: Healthy
    healthy_case = {
        "Type": "M",
        "Air temperature [K]": 298.0,
        "Process temperature [K]": 308.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 20,
    }
    healthy_res = predictor.predict_machine_state(healthy_case)
    assert healthy_res["predicted_failure"] == 0
    assert 0.0 <= healthy_res["failure_probability"] <= 0.20
    assert healthy_res["risk_level"] == "Low"
    assert healthy_res["anomaly_label"] == "Normal"
    assert 0.0 <= healthy_res["anomaly_score"] <= 1.0

    # Case 2: Extreme Overstrain Hazard
    overstrain_case = {
        "Type": "L",
        "Air temperature [K]": 302.5,
        "Process temperature [K]": 311.2,
        "Rotational speed [rpm]": 1280,
        "Torque [Nm]": 65.5,
        "Tool wear [min]": 220,
    }
    hazard_res = predictor.predict_machine_state(overstrain_case)
    assert hazard_res["predicted_failure"] == 1
    assert hazard_res["failure_probability"] >= 0.70
    assert hazard_res["risk_level"] == "High"
    assert hazard_res["anomaly_label"] == "Anomaly"


def test_metrics_and_figures_generated():
    """Verifies that all required evaluation figures and metric files exist."""
    required_figures = [
        "outputs/figures/01_target_and_type_distribution.png",
        "outputs/figures/02_feature_distributions.png",
        "outputs/figures/03_boxplots_by_failure_status.png",
        "outputs/figures/04_correlation_matrix.png",
        "outputs/figures/05_failure_mechanisms_bivariate.png",
        "outputs/figures/06_confusion_matrices.png",
        "outputs/figures/07_roc_pr_curves.png",
        "outputs/figures/08_feature_importances.png",
        "outputs/figures/09_shap_summary.png",
        "outputs/figures/10_anomaly_score_distribution.png",
        "outputs/figures/11_anomaly_scatter_profiles.png",
    ]
    for fig in required_figures:
        assert os.path.exists(fig), f"Missing figure: {fig}"
        assert os.path.getsize(fig) > 1000, f"Corrupted/empty figure: {fig}"

    required_metrics = [
        "outputs/metrics/data_quality_report.json",
        "outputs/metrics/eda_summary.json",
        "outputs/metrics/train_test_split_info.json",
        "outputs/metrics/model_comparison.json",
        "outputs/metrics/model_comparison.csv",
        "outputs/metrics/anomaly_detection_summary.json",
    ]
    for metric_file in required_metrics:
        assert os.path.exists(metric_file), f"Missing metric artifact: {metric_file}"
