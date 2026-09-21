"""
prediction.py
=============
Unified inference engine for Predictive Maintenance & Anomaly Detection.

Loads serialized models independently to generate:
1. Supervised Machine Failure Prediction:
   - Binary class (0: Normal, 1: Failure)
   - Failure probability [0.0 - 1.0]
   - Operational risk tier (Low, Medium, High)
   - Rule-assisted physical diagnostic interpretation
2. Unsupervised Anomaly Detection:
   - Anomaly label ('Normal' vs. 'Anomaly')
   - Normalized anomaly score [0.0 - 1.0]

Accepts either single observation dictionaries or batch DataFrames.
"""

import os
import sys
import logging
from typing import Dict, Any, Union, List
import pandas as pd
import numpy as np
import joblib

# Ensure package importability
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.feature_engineering import BASE_FEATURE_COLUMNS, MachineFeatureEngineer
from src.anomaly_detection import MachineAnomalyDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
DEFAULT_FAILURE_PIPELINE = os.path.join(MODELS_DIR, "failure_prediction_pipeline.pkl")
DEFAULT_ANOMALY_MODEL = os.path.join(MODELS_DIR, "isolation_forest.pkl")


class MaintenancePredictor:
    """
    Inference service that coordinates the supervised failure classifier
    and the unsupervised anomaly detector.
    """

    def __init__(
        self,
        failure_pipeline_path: str = DEFAULT_FAILURE_PIPELINE,
        anomaly_model_path: str = DEFAULT_ANOMALY_MODEL,
    ):
        self.failure_pipeline_path = failure_pipeline_path
        self.anomaly_model_path = anomaly_model_path

        self.failure_pipeline = None
        self.anomaly_detector = None
        self._load_models()

    def _load_models(self):
        """Loads serialized models from disk with fallback path resolution."""
        # Failure pipeline loading
        if not os.path.exists(self.failure_pipeline_path):
            raise FileNotFoundError(f"Failure prediction pipeline not found at '{self.failure_pipeline_path}'")
        self.failure_pipeline = joblib.load(self.failure_pipeline_path)
        logger.info(f"Loaded failure prediction pipeline from: {self.failure_pipeline_path}")

        # Anomaly detector loading (standard IsolationForest estimator)
        if not os.path.exists(self.anomaly_model_path):
            raise FileNotFoundError(f"Anomaly detector model not found at '{self.anomaly_model_path}'")
        self.anomaly_model = joblib.load(self.anomaly_model_path)
        logger.info(f"Loaded anomaly detector model from: {self.anomaly_model_path}")

    def _validate_and_format_input(self, data: Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]]) -> pd.DataFrame:
        """
        Validates that required base operational features exist and formats into DataFrame.
        """
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError("Input data must be a dictionary, list of dictionaries, or a pandas DataFrame.")

        missing_cols = [col for col in BASE_FEATURE_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Input data is missing mandatory features: {missing_cols}")

        # Validate categorical 'Type' values
        valid_types = {"L", "M", "H"}
        invalid_types = set(df["Type"].unique()) - valid_types
        if invalid_types:
            raise ValueError(f"Invalid Product Type detected: {invalid_types}. Must be one of {valid_types}.")

        # Enforce numerical dtypes
        num_cols = [c for c in BASE_FEATURE_COLUMNS if c != "Type"]
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            if df[c].isnull().any():
                raise ValueError(f"Non-numeric values encountered in continuous feature '{c}'.")

        return df[BASE_FEATURE_COLUMNS]

    def _generate_physical_interpretation(self, row: pd.Series, failure_prob: float, is_anomaly: bool) -> str:
        """
        Generates a concise, domain-grounded diagnostic explanation of the prediction.
        """
        insights = []
        torque = row["Torque [Nm]"]
        rpm = row["Rotational speed [rpm]"]
        wear = row["Tool wear [min]"]
        temp_diff = row["Process temperature [K]"] - row["Air temperature [K]"]
        power_w = torque * (rpm * 2 * np.pi / 60)
        overstrain = torque * wear

        if failure_prob >= 0.50:
            if overstrain > 9000 or (wear > 180 and torque > 50):
                insights.append("Critical overstrain risk: High accumulated tool wear combined with elevated torque.")
            if power_w < 2200 or power_w > 8500:
                insights.append(f"Power operating anomaly: Spindle power ({power_w:.0f} W) outside safe envelope.")
            if temp_diff < 8.8 and row["Air temperature [K]"] > 300:
                insights.append(f"Heat dissipation hazard: Low thermal gradient ({temp_diff:.1f} K) under high ambient temperature.")
            if not insights:
                insights.append("Elevated multivariate operational risk detected across physical parameters.")
            status_prefix = "HIGH FAILURE RISK: "
        elif failure_prob >= 0.20:
            status_prefix = "ELEVATED RISK: "
            if wear > 150:
                insights.append("Tool wear approaching preventive maintenance threshold.")
            if torque > 55:
                insights.append("Torque operating near upper design margin.")
            if not insights:
                insights.append("Operating within marginal bounds; monitor closely.")
        else:
            status_prefix = "NORMAL: "
            insights.append("All mechanical and thermal telemetry within standard nominal operating envelope.")

        if is_anomaly:
            insights.append("Telemetry flagged as an atypical statistical outlier by Isolation Forest.")

        return status_prefix + " ".join(insights)

    def predict_failure(self, data: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
        """
        Runs supervised machine failure prediction.
        """
        df = self._validate_and_format_input(data)
        preds = self.failure_pipeline.predict(df)
        probs = self.failure_pipeline.predict_proba(df)[:, 1]

        risk_levels = []
        for p in probs:
            if p >= 0.60:
                risk_levels.append("High")
            elif p >= 0.20:
                risk_levels.append("Medium")
            else:
                risk_levels.append("Low")

        result = pd.DataFrame({
            "predicted_failure": preds,
            "failure_probability": np.round(probs, 4),
            "risk_level": risk_levels,
        })
        return result

    def detect_anomalies(self, data: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
        """
        Runs unsupervised anomaly detection using Isolation Forest.
        """
        df = self._validate_and_format_input(data)
        from src.feature_engineering import engineer_features
        from src.anomaly_detection import ANOMALY_FEATURE_COLUMNS

        df_eng = engineer_features(df, drop_leakage_cols=True)
        X_anom = df_eng[ANOMALY_FEATURE_COLUMNS]

        raw_preds = self.anomaly_model.predict(X_anom)
        is_anomaly = raw_preds == -1
        labels = np.where(is_anomaly, "Anomaly", "Normal")

        raw_scores = -self.anomaly_model.decision_function(X_anom)
        score_min = getattr(self.anomaly_model, "score_min_", float(raw_scores.min()))
        score_max = getattr(self.anomaly_model, "score_max_", float(raw_scores.max()))
        denom = max((score_max - score_min), 1e-6)
        scores = np.clip((raw_scores - score_min) / denom, 0.0, 1.0)

        result = pd.DataFrame({
            "is_anomaly": is_anomaly,
            "anomaly_label": labels,
            "anomaly_score": np.round(scores, 4),
        })
        return result

    def predict_machine_state(self, data: Union[Dict[str, Any], pd.DataFrame]) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Full unified prediction: evaluates both failure probability and anomaly status,
        accompanied by automated physical diagnostic interpretation.
        """
        is_single = isinstance(data, dict)
        df_input = self._validate_and_format_input(data)

        failure_res = self.predict_failure(df_input)
        anomaly_res = self.detect_anomalies(df_input)

        combined = pd.concat([df_input.reset_index(drop=True), failure_res, anomaly_res], axis=1)

        # Generate interpretations
        interpretations = []
        for idx in range(len(combined)):
            row = combined.iloc[idx]
            interp = self._generate_physical_interpretation(
                row, row["failure_probability"], row["is_anomaly"]
            )
            interpretations.append(interp)
        combined["diagnostic_interpretation"] = interpretations

        if is_single:
            return combined.iloc[0].to_dict()
        return combined


# Global helper instance for simple direct imports
_DEFAULT_PREDICTOR = None

def get_predictor() -> MaintenancePredictor:
    """Singleton getter for the inference predictor."""
    global _DEFAULT_PREDICTOR
    if _DEFAULT_PREDICTOR is None:
        _DEFAULT_PREDICTOR = MaintenancePredictor()
    return _DEFAULT_PREDICTOR


def predict(input_data: Union[Dict[str, Any], pd.DataFrame]) -> Union[Dict[str, Any], pd.DataFrame]:
    """Convenience functional wrapper for inference."""
    return get_predictor().predict_machine_state(input_data)


if __name__ == "__main__":
    predictor = MaintenancePredictor()

    print("=" * 75)
    print("DEMO INFERENCE ON REPRESENTATIVE OPERATIONAL SCENARIOS")
    print("=" * 75)

    # Scenario 1: Nominal healthy machine
    healthy_case = {
        "Type": "M",
        "Air temperature [K]": 298.2,
        "Process temperature [K]": 308.7,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 30,
    }

    # Scenario 2: Overstrain failure scenario (High wear + high torque)
    overstrain_case = {
        "Type": "L",
        "Air temperature [K]": 302.5,
        "Process temperature [K]": 311.2,
        "Rotational speed [rpm]": 1280,
        "Torque [Nm]": 65.5,
        "Tool wear [min]": 210,
    }

    # Scenario 3: High-speed / low-torque power anomaly
    anomaly_case = {
        "Type": "H",
        "Air temperature [K]": 300.5,
        "Process temperature [K]": 310.2,
        "Rotational speed [rpm]": 2750,
        "Torque [Nm]": 8.5,
        "Tool wear [min]": 95,
    }

    test_batch = [healthy_case, overstrain_case, anomaly_case]
    batch_results = predictor.predict_machine_state(test_batch)

    display_cols = [
        "Type", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
        "predicted_failure", "failure_probability", "risk_level",
        "anomaly_label", "anomaly_score", "diagnostic_interpretation"
    ]
    for i, r in batch_results[display_cols].iterrows():
        print(f"\n[Test Case {i+1}]")
        for c in display_cols:
            print(f"  {c}: {r[c]}")
    print("=" * 75)
