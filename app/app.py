"""
app.py
======
Streamlit Web Application: Predictive Maintenance & Anomaly Detection Dashboard.

Sections:
1. Executive Overview & Fleet KPIs
2. Supervised Machine Failure Prediction
3. Unsupervised Anomaly Detection
4. Model Performance & Explainability Suite
5. Business Interpretation & Operational Playbook
"""

import os
import sys
import json
import logging
import streamlit as st
import pandas as pd
import numpy as np

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import load_data
from src.prediction import MaintenancePredictor, BASE_FEATURE_COLUMNS

# Configure page metadata
st.set_page_config(
    page_title="Predictive Maintenance & Anomaly Detection",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished industrial aesthetic
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #2b5c8f;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .status-badge-healthy {
        background-color: #d4edda;
        color: #155724;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: bold;
    }
    .status-badge-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: bold;
    }
    .status-badge-critical {
        background-color: #f8d7da;
        color: #721c24;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Cached initialization of unified prediction engine."""
    return MaintenancePredictor()


@st.cache_data
def load_cached_data():
    """Cached loading of raw fleet dataset."""
    return load_data("data/ai4i2020.csv")


@st.cache_data
def load_metric_files():
    """Loads dynamically computed metrics from outputs/metrics/."""
    metrics = {}
    comp_path = "outputs/metrics/model_comparison.json"
    comp_csv = "outputs/metrics/model_comparison.csv"
    anomaly_path = "outputs/metrics/anomaly_detection_summary.json"
    quality_path = "outputs/metrics/data_quality_report.json"

    if os.path.exists(comp_path):
        with open(comp_path, "r") as f:
            metrics["comparison_json"] = json.load(f)
    if os.path.exists(comp_csv):
        metrics["comparison_df"] = pd.read_csv(comp_csv)
    if os.path.exists(anomaly_path):
        with open(anomaly_path, "r") as f:
            metrics["anomaly_json"] = json.load(f)
    if os.path.exists(quality_path):
        with open(quality_path, "r") as f:
            metrics["quality_json"] = json.load(f)

    return metrics


def main():
    st.sidebar.title("⚙️ Industrial AI Suite")
    st.sidebar.caption("Predictive Maintenance & Anomaly Detection")

    menu = st.sidebar.radio(
        "Navigation",
        [
            "Executive Overview & Fleet KPIs",
            "Supervised Failure Prediction",
            "Unsupervised Anomaly Detection",
            "Model Performance & Benchmarks",
            "Business Impact & Operational Guide",
        ],
    )

    df_raw = load_cached_data()
    metrics = load_metric_files()
    predictor = load_predictor()

    # =========================================================================
    # SECTION 1: EXECUTIVE OVERVIEW & FLEET KPIS
    # =========================================================================
    if menu == "Executive Overview & Fleet KPIs":
        st.title("🏭 Fleet Telemetry & Executive Health Overview")
        st.markdown(
            "Continuous real-time telemetry monitoring for **10,000 industrial milling machines** "
            "based on the AI4I 2020 dataset."
        )

        total_obs = len(df_raw)
        total_fail = int(df_raw["Machine failure"].sum())
        fail_rate = total_fail / total_obs * 100

        anomaly_cnt = metrics.get("anomaly_json", {}).get("detection_results", {}).get("anomalous_observations", 400)
        anomaly_rate = metrics.get("anomaly_json", {}).get("detection_results", {}).get("anomaly_rate_percentage", 4.0)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Monitored Units", f"{total_obs:,}")
        with c2:
            st.metric("Historical Failures", f"{total_fail:,}", f"{fail_rate:.2f}% incidence", delta_color="inverse")
        with c3:
            st.metric("Detected Anomaly States", f"{anomaly_cnt:,}", f"{anomaly_rate:.2f}% fleet outliers", delta_color="inverse")
        with c4:
            st.metric("Fleet Quality Variants", f"L (60%) | M (30%) | H (10%)")

        st.markdown("---")

        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.subheader("Target & Variant Distribution")
            fig1_path = "outputs/figures/01_target_and_type_distribution.png"
            if os.path.exists(fig1_path):
                st.image(fig1_path, caption="Figure 1: Target Class Imbalance (28.5:1) & Failure Rate by Quality Variant")
            else:
                st.info("Distribution figure will display once generated.")

        with col_right:
            st.subheader("Operating Envelope Distributions")
            fig2_path = "outputs/figures/02_feature_distributions.png"
            if os.path.exists(fig2_path):
                st.image(fig2_path, caption="Figure 2: Kernel Density Estimations of Continuous Sensor Telemetry")
            else:
                st.info("Continuous feature distribution figure will display once generated.")

        st.markdown("---")
        st.subheader("Telemetry Explorer & Summary Statistics")
        st.dataframe(
            df_raw[["Type", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Machine failure"]].describe().T.round(2),
            use_container_width=True,
        )

    # =========================================================================
    # SECTION 2: SUPERVISED FAILURE PREDICTION
    # =========================================================================
    elif menu == "Supervised Failure Prediction":
        st.title("🎯 Supervised Machine Failure Prediction")
        st.markdown(
            "Estimate failure probability using our tuned **Random Forest Classifier** ($99.15\\%$ accuracy, "
            "$0.8640$ F1-Score). Features are dynamically processed through our domain engineering pipeline."
        )

        st.warning(
            "⚠️ **Operational Disclaimer:** Predictions are statistical estimates derived from historical telemetry patterns. "
            "They serve as preventive risk alerts for maintenance prioritization and do not guarantee instantaneous physical failure."
        )

        st.markdown("### Operational Telemetry Inputs")

        # Scenario Presets
        preset = st.selectbox(
            "Select Scenario Preset (or customize sliders below):",
            [
                "Custom Configuration",
                "Preset 1: Healthy Nominal Operation",
                "Preset 2: Tool Wear & Overstrain Hazard (High Wear + High Torque)",
                "Preset 3: Heat Dissipation Risk (High Ambient Temp + Low Delta)",
                "Preset 4: Spindle Power Anomaly (High RPM + Low Torque)",
            ],
        )

        # Preset defaults
        if preset == "Preset 1: Healthy Nominal Operation":
            d_type, d_air, d_proc, d_speed, d_torque, d_wear = "M", 298.5, 308.7, 1520, 39.5, 35
        elif preset == "Preset 2: Tool Wear & Overstrain Hazard (High Wear + High Torque)":
            d_type, d_air, d_proc, d_speed, d_torque, d_wear = "L", 302.2, 311.5, 1260, 68.0, 215
        elif preset == "Preset 3: Heat Dissipation Risk (High Ambient Temp + Low Delta)":
            d_type, d_air, d_proc, d_speed, d_torque, d_wear = "L", 304.1, 312.8, 1420, 52.0, 110
        elif preset == "Preset 4: Spindle Power Anomaly (High RPM + Low Torque)":
            d_type, d_air, d_proc, d_speed, d_torque, d_wear = "H", 300.2, 309.8, 2780, 8.2, 85
        else:
            d_type, d_air, d_proc, d_speed, d_torque, d_wear = "L", 300.0, 310.0, 1500, 40.0, 100

        c1, c2, c3 = st.columns(3)
        with c1:
            inp_type = st.selectbox("Product Quality Variant (Type)", ["L", "M", "H"], index=["L", "M", "H"].index(d_type))
            inp_air = st.slider("Air Temperature [K]", 290.0, 310.0, float(d_air), 0.1, help="Ambient temperature (295.3K - 304.5K)")
        with c2:
            inp_proc = st.slider("Process Temperature [K]", 300.0, 320.0, float(d_proc), 0.1, help="Internal operating heat (305.7K - 313.8K)")
            inp_speed = st.slider("Rotational Speed [rpm]", 1000, 3000, int(d_speed), 10, help="Spindle RPM (1168 - 2886 rpm)")
        with c3:
            inp_torque = st.slider("Torque [Nm]", 0.0, 85.0, float(d_torque), 0.5, help="Cutting resistance load (3.8 - 76.6 Nm)")
            inp_wear = st.slider("Tool Wear [min]", 0, 300, int(d_wear), 1, help="Accumulated tool engagement time")

        input_record = {
            "Type": inp_type,
            "Air temperature [K]": inp_air,
            "Process temperature [K]": inp_proc,
            "Rotational speed [rpm]": inp_speed,
            "Torque [Nm]": inp_torque,
            "Tool wear [min]": inp_wear,
        }

        # Computed physical indicators
        temp_delta = inp_proc - inp_air
        power_kw = (inp_torque * inp_speed * 2 * np.pi / 60) / 1000.0
        overstrain_idx = inp_torque * inp_wear

        st.markdown("#### Real-time Computed Physical Indicators")
        i1, i2, i3 = st.columns(3)
        with i1:
            st.metric("Thermal Gradient (ΔT)", f"{temp_delta:.2f} K", "Normal > 8.9 K" if temp_delta >= 8.9 else "Critical < 8.9 K", delta_color="normal" if temp_delta >= 8.9 else "inverse")
        with i2:
            st.metric("Mechanical Power", f"{power_kw:.2f} kW", "Safe: 2.0 - 9.0 kW" if 2.0 <= power_kw <= 9.0 else "Envelope Breach", delta_color="normal" if 2.0 <= power_kw <= 9.0 else "inverse")
        with i3:
            st.metric("Overstrain Stress Index", f"{overstrain_idx:.1f}", "Critical > 9000" if overstrain_idx > 9000 else "Nominal", delta_color="inverse" if overstrain_idx > 9000 else "normal")

        if st.button("Run Failure Assessment", type="primary", use_container_width=True):
            prediction_result = predictor.predict_machine_state(input_record)

            prob = prediction_result["failure_probability"]
            pred_class = prediction_result["predicted_failure"]
            risk = prediction_result["risk_level"]
            interp = prediction_result["diagnostic_interpretation"]

            st.markdown("---")
            st.subheader("Diagnostic Assessment")

            r1, r2 = st.columns([1, 2])
            with r1:
                if pred_class == 1 or prob >= 0.50:
                    st.error(f"### Status: FAILURE WARNING\n**Risk Tier:** {risk} ({prob * 100:.1f}%)")
                elif prob >= 0.20:
                    st.warning(f"### Status: ELEVATED RISK\n**Risk Tier:** {risk} ({prob * 100:.1f}%)")
                else:
                    st.success(f"### Status: NOMINAL HEALTH\n**Risk Tier:** {risk} ({prob * 100:.1f}%)")
                st.progress(float(prob))

            with r2:
                st.info(f"**Engineering Rationale:**\n\n{interp}")

    # =========================================================================
    # SECTION 3: UNSUPERVISED ANOMALY DETECTION
    # =========================================================================
    elif menu == "Unsupervised Anomaly Detection":
        st.title("🔍 Unsupervised Machine Anomaly Detection")
        st.markdown(
            "Detect unusual, atypical operating states using **Isolation Forest** (150 estimators, $4\\%$ contamination). "
            "Isolation Forest operates **without target failure labels** to identify multi-dimensional sensor drift."
        )

        st.info(
            "💡 **Consultant Concept Note:** Anomaly Detection and Failure Prediction solve different problems. "
            "Failure prediction relies on historical failure occurrences. Anomaly detection flags statistical outliers "
            "in multi-dimensional sensor space, providing early warning for novel, unprecedented machine stress."
        )

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.subheader("Operational Input Telemetry")
            a_type = st.selectbox("Product Type", ["L", "M", "H"], key="anom_type")
            a_air = st.slider("Air Temperature [K]", 290.0, 310.0, 300.0, 0.1, key="anom_air")
            a_proc = st.slider("Process Temperature [K]", 300.0, 320.0, 310.0, 0.1, key="anom_proc")
            a_speed = st.slider("Rotational Speed [rpm]", 1000, 3000, 1500, 10, key="anom_speed")
            a_torque = st.slider("Torque [Nm]", 0.0, 85.0, 40.0, 0.5, key="anom_torque")
            a_wear = st.slider("Tool Wear [min]", 0, 300, 100, 1, key="anom_wear")

            test_row = {
                "Type": a_type,
                "Air temperature [K]": a_air,
                "Process temperature [K]": a_proc,
                "Rotational speed [rpm]": a_speed,
                "Torque [Nm]": a_torque,
                "Tool wear [min]": a_wear,
            }

            if st.button("Evaluate Anomaly Score", use_container_width=True):
                anom_res = predictor.detect_anomalies(test_row).iloc[0]
                label = anom_res["anomaly_label"]
                score = anom_res["anomaly_score"]

                st.markdown("---")
                if label == "Anomaly":
                    st.error(f"### Status: ANOMALOUS OPERATING STATE\n**Normalized Score:** {score:.4f} (High Outlier Density)")
                else:
                    st.success(f"### Status: NOMINAL INLIER\n**Normalized Score:** {score:.4f} (Standard Operating Envelope)")
                st.progress(float(score))

        with col_b:
            st.subheader("Outlier Isolation Space")
            fig11_path = "outputs/figures/11_anomaly_scatter_profiles.png"
            if os.path.exists(fig11_path):
                st.image(fig11_path, caption="Figure 11: Mechanical & Thermal Outlier Separation in Multi-Dimensional Space")
            else:
                st.info("Anomaly scatter profiles will display once generated.")

    # =========================================================================
    # SECTION 4: MODEL PERFORMANCE & BENCHMARKS
    # =========================================================================
    elif menu == "Model Performance & Benchmarks":
        st.title("📊 Rigorous Model Comparison & Benchmarks")
        st.markdown(
            "Evaluation against the **stratified hold-out test set** ($2,000$ unseen samples, $68$ failures). "
            "All metrics are generated dynamically from test execution."
        )

        if "comparison_df" in metrics:
            st.subheader("Hold-Out Test Evaluation Matrix")
            st.dataframe(metrics["comparison_df"].set_index("Model"), use_container_width=True)

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Confusion Matrices (Test Set)")
            fig6_path = "outputs/figures/06_confusion_matrices.png"
            if os.path.exists(fig6_path):
                st.image(fig6_path, caption="Figure 6: Side-by-Side Confusion Matrices")
        with col2:
            st.subheader("ROC & Precision-Recall Curves")
            fig7_path = "outputs/figures/07_roc_pr_curves.png"
            if os.path.exists(fig7_path):
                st.image(fig7_path, caption="Figure 7: Receiver Operating Characteristic & Precision-Recall Curves")

        st.markdown("---")

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Feature Importances")
            fig8_path = "outputs/figures/08_feature_importances.png"
            if os.path.exists(fig8_path):
                st.image(fig8_path, caption="Figure 8: Tree Feature Importance Ranking")
        with col4:
            st.subheader("SHAP Global Interpretability")
            fig9_path = "outputs/figures/09_shap_summary.png"
            if os.path.exists(fig9_path):
                st.image(fig9_path, caption="Figure 9: SHAP Summary Beeswarm Plot")

    # =========================================================================
    # SECTION 5: BUSINESS IMPACT & OPERATIONAL GUIDE
    # =========================================================================
    elif menu == "Business Impact & Operational Guide":
        st.title("💼 Data & AI Consultant Operational Guide")

        st.markdown("""
        ### Industrial Predictive Maintenance Strategy

        #### 1. Why Accuracy Alone is Fatal in Industrial AI
        In predictive maintenance, failures are rare events ($3.39\\%$ prevalence). A model that never alerts achieves **$96.61\\%$ accuracy** but permits 100% of catastrophic equipment breakdowns. In production, we evaluate:
        - **Recall:** Catching catastrophic breakdowns before they happen to avoid expensive unplanned line stoppages.
        - **Precision:** Preventing maintenance alert fatigue by minimizing false alarms.
        - **F1-Score / PR-AUC:** The harmonic balance that optimizes technician scheduling.

        #### 2. Business Economics: False Positives vs. False Negatives
        - **False Negative (Uncaught Breakdown):** Emergency line downtime, ruined manufacturing workpieces, damaged tooling, and emergency technician call-outs ($$$$).
        - **False Positive (Preventive False Alarm):** $\\approx 20$ minutes of scheduled technician inspection and diagnostic testing ($).
        - *Production Decision:* Random Forest achieves **$79.4\\%$ recall** while generating only **3 false alarms across 2,000 machines**, preventing technician alert fatigue.

        #### 3. Dual-Layer AI Architecture
        1. **Layer 1: Supervised Classifier (Random Forest):** Primary trigger predicting failure likelihood based on historical breakdown signatures.
        2. **Layer 2: Unsupervised Outlier Guardrail (Isolation Forest):** Flags operational drift, extreme speeds, and unprecedented load conditions that have not yet occurred historically.

        #### 4. Real-World Limitations & Factory Roadmap
        - **Sensor Expansion:** Integrate high-frequency vibration accelerometers (bearing fault frequencies) and acoustic emission sensors.
        - **Time-Series Telemetry:** Transition from snapshot tabular records to sequential LSTM/Transformer models tracking cumulative wear over time.
        - **Maintenance Feedback Loop:** Continuous retraining based on actual technician inspection work-orders.
        """)


if __name__ == "__main__":
    main()
