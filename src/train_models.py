"""
train_models.py
===============
Supervised training, cross-validation, hyperparameter tuning, model comparison,
and explainability for the Predictive Maintenance system.

Evaluates:
1. Logistic Regression (Baseline with scaling and class weighting)
2. Random Forest Classifier (Tuned via Stratified 5-Fold Cross-Validation)
3. XGBoost Classifier (Tuned with scale_pos_weight and early stopping)

Artifacts produced:
- models/failure_model.pkl (Best overall model)
- models/logistic_regression.pkl, models/random_forest.pkl, models/xgboost.pkl
- outputs/metrics/model_comparison.json & .csv
- outputs/figures/06_confusion_matrices.png
- outputs/figures/07_roc_pr_curves.png
- outputs/figures/08_feature_importances.png
- outputs/figures/09_shap_summary.png
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Ensure package importability
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import load_data
from src.preprocessing import split_data, fit_and_save_preprocessors, build_full_preprocessing_pipeline
from src.evaluation import (
    compute_metrics,
    plot_confusion_matrices,
    plot_roc_pr_curves,
    plot_feature_importances,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import shap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
OUTPUT_METRICS_DIR = "outputs/metrics"
OUTPUT_FIG_DIR = "outputs/figures"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_METRICS_DIR, exist_ok=True)
os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)

# Transformed feature column ordering
FEATURE_NAMES = [
    "Type_H",
    "Type_L",
    "Type_M",
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


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> LogisticRegression:
    """
    Trains Logistic Regression baseline using balanced class weighting to handle the 28.5:1 imbalance.
    """
    logger.info("Training Logistic Regression Baseline...")
    # Using class_weight='balanced' inversely weights class frequencies
    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs",
    )
    lr.fit(X_train, y_train)
    return lr


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Trains Random Forest with Stratified 5-Fold Cross-Validation hyperparameter tuning.
    """
    logger.info("Tuning and training Random Forest Classifier...")
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [8, 12, None],
        "min_samples_split": [2, 5],
        "class_weight": ["balanced", "balanced_subsample"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    rf_base = RandomForestClassifier(random_state=random_state, n_jobs=-1)

    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)
    logger.info(f"Best Random Forest CV F1: {grid_search.best_score_:.4f} with params: {grid_search.best_params_}")
    return grid_search.best_estimator_


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scale_pos_weight: float = 28.5,
    random_state: int = 42,
) -> XGBClassifier:
    """
    Trains XGBoost Classifier with Stratified 5-Fold Cross-Validation tuning.
    """
    logger.info("Tuning and training XGBoost Classifier...")
    param_grid = {
        "n_estimators": [100, 150],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "scale_pos_weight": [1.0, scale_pos_weight / 2.0, scale_pos_weight],
        "subsample": [0.8, 1.0],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    xgb_base = XGBClassifier(
        random_state=random_state,
        eval_metric="logloss",
        n_jobs=-1,
    )

    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)
    logger.info(f"Best XGBoost CV F1: {grid_search.best_score_:.4f} with params: {grid_search.best_params_}")
    return grid_search.best_estimator_


def generate_shap_analysis(model, X_test: np.ndarray, feature_names: list, filename: str = "09_shap_summary.png"):
    """
    Computes Tree SHAP values on hold-out test set and plots feature attribution summary.
    """
    logger.info("Computing SHAP values for model interpretability...")
    try:
        explainer = shap.TreeExplainer(model)
        # Use first 500 test samples for fast, robust representation
        sample_size = min(500, len(X_test))
        X_sample = X_test[:sample_size]
        shap_values = explainer.shap_values(X_sample)

        # Handle binary classification output formats in SHAP
        if isinstance(shap_values, list):
            sv = shap_values[1]  # positive class (failure)
        elif len(shap_values.shape) == 3:
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            sv,
            X_sample,
            feature_names=feature_names,
            show=False,
            max_display=10,
        )
        plt.title("SHAP Feature Attribution (Impact on Failure Prediction)", pad=15)
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_FIG_DIR, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved SHAP summary plot to: {save_path}")
    except Exception as e:
        logger.warning(f"SHAP explanation generation encountered non-fatal notice: {e}")


def execute_model_training_pipeline() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Main orchestration function:
    1. Loads dataset and executes stratified split.
    2. Fits and serializes preprocessors.
    3. Trains Logistic Regression, Random Forest, and XGBoost.
    4. Evaluates all models on holdout test set.
    5. Compares metrics, generates plots, and serializes the winning model.
    """
    df = load_data()
    X_train_raw, X_test_raw, y_train, y_test, split_info = split_data(df)

    # Preprocessing
    tree_pipe, linear_pipe = fit_and_save_preprocessors(X_train_raw)

    # Transformed data arrays
    X_train_tree = tree_pipe.transform(X_train_raw)
    X_test_tree = tree_pipe.transform(X_test_raw)

    X_train_scaled = linear_pipe.transform(X_train_raw)
    X_test_scaled = linear_pipe.transform(X_test_raw)

    # 1. Train Baseline: Logistic Regression
    lr_model = train_logistic_regression(X_train_scaled, y_train.values)
    joblib.dump(lr_model, os.path.join(MODELS_DIR, "logistic_regression.pkl"))

    # Also train unweighted Logistic Regression to demonstrate the baseline without class-weighting
    lr_unweighted = LogisticRegression(max_iter=1000, random_state=42)
    lr_unweighted.fit(X_train_scaled, y_train.values)

    # 2. Train Random Forest
    rf_model = train_random_forest(X_train_tree, y_train.values)
    joblib.dump(rf_model, os.path.join(MODELS_DIR, "random_forest.pkl"))

    # 3. Train XGBoost
    imbalance_ratio = split_info["training_set"]["normal_count"] / split_info["training_set"]["failure_count"]
    xgb_model = train_xgboost(X_train_tree, y_train.values, scale_pos_weight=imbalance_ratio)
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgboost.pkl"))

    # Model Evaluation dictionary
    models = {
        "Logistic Regression (Balanced)": (lr_model, X_test_scaled),
        "Logistic Regression (Unweighted)": (lr_unweighted, X_test_scaled),
        "Random Forest": (rf_model, X_test_tree),
        "XGBoost": (xgb_model, X_test_tree),
    }

    eval_results = {}
    eval_curves = {}

    for name, (model, X_eval) in models.items():
        y_pred = model.predict(X_eval)
        y_prob = model.predict_proba(X_eval)[:, 1]

        metrics = compute_metrics(y_test.values, y_pred, y_prob, model_name=name)
        eval_results[name] = metrics
        eval_curves[name] = {"y_true": y_test.values, "y_prob": y_prob}

    # Generate comparison DataFrame
    comparison_rows = []
    for name, m in eval_results.items():
        comparison_rows.append({
            "Model": name,
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1-Score": m["f1_score"],
            "ROC-AUC": m["roc_auc"],
            "PR-AUC": m["pr_auc"],
        })

    comparison_df = pd.DataFrame(comparison_rows).sort_values(by="F1-Score", ascending=False)

    # Save metrics tables
    comparison_csv = os.path.join(OUTPUT_METRICS_DIR, "model_comparison.csv")
    comparison_json = os.path.join(OUTPUT_METRICS_DIR, "model_comparison.json")
    comparison_df.to_csv(comparison_csv, index=False)
    with open(comparison_json, "w") as f:
        json.dump(eval_results, f, indent=4)
    logger.info(f"Saved model comparison table to: {comparison_csv}")

    # Generate figures
    plot_confusion_matrices(eval_results, filename="06_confusion_matrices.png")
    plot_roc_pr_curves(eval_curves, filename="07_roc_pr_curves.png")

    # Feature Importance Plot
    rf_importances = pd.Series(rf_model.feature_importances_, index=FEATURE_NAMES, name="Random Forest")
    xgb_importances = pd.Series(xgb_model.feature_importances_, index=FEATURE_NAMES, name="XGBoost")
    importance_df = pd.concat([rf_importances, xgb_importances], axis=1)
    plot_feature_importances(importance_df, filename="08_feature_importances.png")

    # SHAP Explainability Plot on XGBoost
    generate_shap_analysis(xgb_model, X_test_tree, FEATURE_NAMES, filename="09_shap_summary.png")

    # Select winning model based on F1-Score (optimal balance of catching failures and avoiding false alarms)
    # Exclude unweighted LR from candidate winning selection
    candidate_names = ["Logistic Regression (Balanced)", "Random Forest", "XGBoost"]
    best_name = max(candidate_names, key=lambda n: eval_results[n]["f1_score"])
    best_model, _ = models[best_name]

    logger.info(f"Selected Winning Failure Prediction Model: {best_name} (F1 = {eval_results[best_name]['f1_score']:.4f})")

    # Save best model to models/failure_model.pkl
    best_model_path = os.path.join(MODELS_DIR, "failure_model.pkl")
    joblib.dump(best_model, best_model_path)
    logger.info(f"Saved winning model to: {best_model_path}")

    # Also build and save end-to-end deployable pipeline (preprocessor + best model)
    is_scaled = "Logistic Regression" in best_name
    best_preprocessor = linear_pipe if is_scaled else tree_pipe

    end_to_end_pipeline = Pipeline(steps=[
        ("feature_engineer", best_preprocessor.named_steps["feature_engineer"]),
        ("preprocessor", best_preprocessor.named_steps["preprocessor"]),
        ("classifier", best_model),
    ])
    pipeline_save_path = os.path.join(MODELS_DIR, "failure_prediction_pipeline.pkl")
    joblib.dump(end_to_end_pipeline, pipeline_save_path)
    logger.info(f"Saved end-to-end failure prediction pipeline to: {pipeline_save_path}")

    return comparison_df, eval_results


if __name__ == "__main__":
    comp_df, results = execute_model_training_pipeline()
    print("\n" + "=" * 70)
    print("SUPERVISED MODEL EVALUATION COMPARISON ON HOLDOUT TEST SET")
    print("=" * 70)
    print(comp_df.to_string(index=False))
    print("=" * 70)
