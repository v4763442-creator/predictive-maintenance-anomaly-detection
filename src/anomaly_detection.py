"""
anomaly_detection.py
====================
Unsupervised Anomaly Detection module using Isolation Forest for the AI4I 2020 dataset.

Core Principles:
1. Purely Unsupervised: Zero label inputs. Does NOT use Machine failure, TWF, HDF, PWF, OSF, or RNF.
2. Operating Features: Uses physical machine state variables and domain-engineered features.
3. Decoupled Objective: Identifies atypical operating states (statistical outliers), distinct
   from supervised failure classification.
4. Serializes models/isolation_forest.pkl and outputs/metrics/anomaly_detection_summary.json.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
import joblib

# Ensure package importability
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import load_data
from src.feature_engineering import engineer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
OUTPUT_METRICS_DIR = "outputs/metrics"
OUTPUT_FIG_DIR = "outputs/figures"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_METRICS_DIR, exist_ok=True)
os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)

# Continuous physical features used for multidimensional isolation trees
ANOMALY_FEATURE_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Temp_Difference_K",
    "Temp_Ratio",
    "Power_W",
    "Overstrain_Torque_Wear",
]


class MachineAnomalyDetector:
    """
    Wrapper for Isolation Forest tailored to continuous industrial machine telemetry.
    """

    def __init__(self, contamination: float = 0.04, n_estimators: int = 150, random_state: int = 42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self.feature_names = ANOMALY_FEATURE_COLUMNS
        self.score_min_ = None
        self.score_max_ = None

    def fit(self, X: pd.DataFrame):
        """Fits Isolation Forest on continuous operating features."""
        X_clean = self._extract_features(X)
        logger.info(f"Fitting Isolation Forest on {X_clean.shape[0]} samples with {X_clean.shape[1]} features...")
        self.model.fit(X_clean)

        raw_scores = -self.model.decision_function(X_clean)
        self.score_min_ = float(raw_scores.min())
        self.score_max_ = float(raw_scores.max())
        self.model.score_min_ = self.score_min_
        self.model.score_max_ = self.score_max_
        return self

    def _extract_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensures engineered features exist and filters only anomaly features."""
        df = X.copy()
        if "Power_W" not in df.columns or "Temp_Difference_K" not in df.columns:
            df = engineer_features(df, drop_leakage_cols=True)
        missing = [c for c in self.feature_names if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required anomaly detection features: {missing}")
        return df[self.feature_names]

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Inference routine.
        Returns:
            - is_anomaly (bool): True if observation is an outlier.
            - labels (str array): 'Anomaly' or 'Normal'.
            - anomaly_scores (float array): Normalized anomaly score in [0, 1] where 1 is extreme outlier.
        """
        X_clean = self._extract_features(X)
        raw_preds = self.model.predict(X_clean)  # 1 for inliers, -1 for outliers
        is_anomaly = raw_preds == -1
        labels = np.where(is_anomaly, "Anomaly", "Normal")

        raw_scores = -self.model.decision_function(X_clean)
        # Normalize score into [0, 1] range based on fitted boundaries
        denom = max((self.score_max_ - self.score_min_), 1e-6)
        normalized_scores = np.clip((raw_scores - self.score_min_) / denom, 0.0, 1.0)

        return is_anomaly, labels, normalized_scores


def train_and_evaluate_anomaly_detection(contamination: float = 0.04) -> Dict[str, Any]:
    """
    Trains Isolation Forest on operating features, computes outlier rates,
    generates exploratory comparisons against Machine failure, and saves figures.
    """
    logger.info("Executing Unsupervised Anomaly Detection pipeline...")
    raw_df = load_data()

    # Engineer physical features
    df_engineered = engineer_features(raw_df, drop_leakage_cols=False)

    detector = MachineAnomalyDetector(contamination=contamination, random_state=42)
    detector.fit(df_engineered)

    # Serialize trained anomaly detector (standard IsolationForest estimator)
    model_save_path = os.path.join(MODELS_DIR, "isolation_forest.pkl")
    joblib.dump(detector.model, model_save_path)
    logger.info(f"Saved Isolation Forest model to: {model_save_path}")

    # Generate predictions and scores
    is_anomaly, labels, scores = detector.predict(df_engineered)

    df_results = df_engineered.copy()
    df_results["Is_Anomaly"] = is_anomaly
    df_results["Anomaly_Label"] = labels
    df_results["Anomaly_Score"] = scores

    # Summary Statistics
    total_obs = len(df_results)
    anomaly_cnt = int(is_anomaly.sum())
    anomaly_rate_pct = round(anomaly_cnt / total_obs * 100, 2)

    # Exploratory overlap with Machine Failure (Strictly exploratory - NOT the training target)
    overlap_ct = pd.crosstab(
        df_results["Anomaly_Label"],
        raw_df["Machine failure"].map({0: "Normal Machine", 1: "Failure Event"}),
        margins=True,
    )

    # Detailed overlap counts
    true_failures = raw_df["Machine failure"] == 1
    failures_caught_by_anomaly = int((is_anomaly & true_failures).sum())
    failures_missed_by_anomaly = int((~is_anomaly & true_failures).sum())
    anomalies_without_failure = int((is_anomaly & ~true_failures).sum())

    recall_as_proxy = round(failures_caught_by_anomaly / int(true_failures.sum()) * 100, 2)
    precision_as_proxy = round(failures_caught_by_anomaly / anomaly_cnt * 100, 2)

    logger.info(f"Detected {anomaly_cnt} anomalies ({anomaly_rate_pct}%) out of {total_obs} machines.")
    logger.info(f"Exploratory overlap: {failures_caught_by_anomaly} of {true_failures.sum()} failures were anomalous.")

    # 1. Visualization: Anomaly Score Distribution & Exploratory Failure Overlay
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(
        df_results[~is_anomaly]["Anomaly_Score"],
        bins=30,
        color="#2b5c8f",
        label="Normal Operating State",
        kde=True,
        stat="density",
        alpha=0.6,
    )
    sns.histplot(
        df_results[is_anomaly]["Anomaly_Score"],
        bins=20,
        color="#d95f02",
        label="Detected Anomaly",
        kde=True,
        stat="density",
        alpha=0.8,
    )
    plt.axvline(
        df_results[is_anomaly]["Anomaly_Score"].min(),
        color="red",
        linestyle="--",
        label="Anomaly Threshold",
    )
    plt.title("Isolation Forest Anomaly Score Distribution", pad=12)
    plt.xlabel("Normalized Anomaly Score [0 = Typical, 1 = Outlier]")
    plt.legend(loc="upper right")

    # Exploratory comparison with true failures
    plt.subplot(1, 2, 2)
    sns.kdeplot(
        data=df_results,
        x="Anomaly_Score",
        hue=raw_df["Machine failure"].map({0: "Actual Normal", 1: "Actual Failure"}),
        palette={"Actual Normal": "#2b5c8f", "Actual Failure": "#e41a1c"},
        common_norm=False,
        fill=True,
        alpha=0.4,
    )
    plt.title("Exploratory Distribution: Anomaly Score by Machine Failure", pad=12)
    plt.xlabel("Normalized Anomaly Score")

    plt.tight_layout()
    fig1_path = os.path.join(OUTPUT_FIG_DIR, "10_anomaly_score_distribution.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    logger.info(f"Saved anomaly distribution figure to: {fig1_path}")

    # 2. Visualization: Physical Scatter Plots Showing Isolated Outliers
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # Spindle envelope (Speed vs Torque)
    sns.scatterplot(
        data=df_results[~is_anomaly],
        x="Rotational speed [rpm]",
        y="Torque [Nm]",
        color="#2b5c8f",
        alpha=0.3,
        s=20,
        label="Normal Operation",
        ax=axes[0],
    )
    sns.scatterplot(
        data=df_results[is_anomaly],
        x="Rotational speed [rpm]",
        y="Torque [Nm]",
        color="#d95f02",
        alpha=0.9,
        s=35,
        marker="^",
        label="Detected Outlier / Anomaly",
        ax=axes[0],
    )
    axes[0].set_title("Mechanical Envelope: Detected Operating Outliers", pad=12)
    axes[0].legend(loc="upper right")

    # Thermal envelope (Temp Difference vs Power)
    sns.scatterplot(
        data=df_results[~is_anomaly],
        x="Temp_Difference_K",
        y="Power_W",
        color="#2b5c8f",
        alpha=0.3,
        s=20,
        label="Normal Operation",
        ax=axes[1],
    )
    sns.scatterplot(
        data=df_results[is_anomaly],
        x="Temp_Difference_K",
        y="Power_W",
        color="#d95f02",
        alpha=0.9,
        s=35,
        marker="^",
        label="Detected Outlier / Anomaly",
        ax=axes[1],
    )
    axes[1].set_title("Thermo-Mechanical Envelope: Detected Operating Outliers", pad=12)
    axes[1].set_xlabel("Process Temp - Air Temp [K]")
    axes[1].set_ylabel("Mechanical Power [W]")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    fig2_path = os.path.join(OUTPUT_FIG_DIR, "11_anomaly_scatter_profiles.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    logger.info(f"Saved anomaly scatter profiles to: {fig2_path}")

    # Save summary report
    anomaly_summary = {
        "model_type": "Isolation Forest (Unsupervised)",
        "hyperparameters": {
            "contamination": contamination,
            "n_estimators": 150,
            "random_state": 42,
        },
        "features_used": ANOMALY_FEATURE_COLUMNS,
        "features_excluded_for_unsupervised_integrity": [
            "Machine failure (Target)",
            "TWF", "HDF", "PWF", "OSF", "RNF (Failure Modes)",
            "UDI", "Product ID (Identifiers)",
        ],
        "detection_results": {
            "total_observations": total_obs,
            "anomalous_observations": anomaly_cnt,
            "normal_observations": total_obs - anomaly_cnt,
            "anomaly_rate_percentage": anomaly_rate_pct,
        },
        "exploratory_overlap_with_machine_failure": {
            "note": (
                "Isolation Forest was trained without target labels. This cross-tabulation is strictly "
                "an exploratory analytical comparison, NOT a supervised validation metric."
            ),
            "actual_failures_in_dataset": int(true_failures.sum()),
            "actual_failures_flagged_as_anomalies": failures_caught_by_anomaly,
            "actual_failures_unflagged_as_normal": failures_missed_by_anomaly,
            "anomalies_operating_normally_no_failure": anomalies_without_failure,
            "exploratory_failure_capture_rate_pct": recall_as_proxy,
            "exploratory_anomaly_precision_pct": precision_as_proxy,
        },
        "conceptual_distinction_interview_guide": {
            "failure_prediction_vs_anomaly_detection": (
                "Failure prediction is a supervised task mapping operational telemetry to known historic breakdown events. "
                "It excels when historic failure labels are plentiful and future failures replicate past mechanisms. "
                "Anomaly detection is an unsupervised task isolating statistical outliers in multi-dimensional feature space. "
                "It is essential when novel failure modes occur, operating regimes drift, or labels are unrecorded. "
                "In real production, they are complementary: Anomaly detection flags unusual drift before an explicit failure pattern forms."
            )
        },
    }

    summary_path = os.path.join(OUTPUT_METRICS_DIR, "anomaly_detection_summary.json")
    with open(summary_path, "w") as f:
        json.dump(anomaly_summary, f, indent=4)
    logger.info(f"Saved anomaly detection summary to: {summary_path}")

    return anomaly_summary


if __name__ == "__main__":
    summary = train_and_evaluate_anomaly_detection(contamination=0.04)
    print("\n" + "=" * 70)
    print("UNSUPERVISED ANOMALY DETECTION (ISOLATION FOREST) RESULTS")
    print("=" * 70)
    print(f"Total Observations: {summary['detection_results']['total_observations']:,}")
    print(f"Flagged Anomalies: {summary['detection_results']['anomalous_observations']:,} ({summary['detection_results']['anomaly_rate_percentage']}%)")
    print("\n[Exploratory Comparison with Machine Failure - NOT Training Objective]")
    exp = summary["exploratory_overlap_with_machine_failure"]
    print(f"  Total Actual Failures: {exp['actual_failures_in_dataset']}")
    print(f"  Failures Falling in Anomaly Regions: {exp['actual_failures_flagged_as_anomalies']} ({exp['exploratory_failure_capture_rate_pct']}%)")
    print(f"  Failures in Normal Operating Space: {exp['actual_failures_unflagged_as_normal']}")
    print(f"  Anomalous Operating States with No Failure: {exp['anomalies_operating_normally_no_failure']}")
    print("=" * 70)
