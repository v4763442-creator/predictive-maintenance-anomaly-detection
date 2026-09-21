"""
data_loader.py
==============
Data loading and quality assurance module for the AI4I 2020 Predictive Maintenance dataset.

Responsibilities:
1. Load dataset with path resolution and schema verification.
2. Perform comprehensive data auditing (null checks, duplicate detection, physical plausibility checks).
3. Export structured audit reports to outputs/metrics/.
"""

import os
import json
import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

# Configure module logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Expected canonical columns
EXPECTED_COLUMNS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

# Physical plausibility boundaries
PHYSICAL_BOUNDS = {
    "Air temperature [K]": (273.15, 373.15),      # 0°C to 100°C
    "Process temperature [K]": (273.15, 373.15),  # 0°C to 100°C
    "Rotational speed [rpm]": (100, 10000),       # realistic mechanical spindle speed
    "Torque [Nm]": (0.1, 500.0),                  # positive non-zero torque
    "Tool wear [min]": (0, 1000),                 # tool wear duration
}

FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def load_data(filepath: str = "data/ai4i2020.csv") -> pd.DataFrame:
    """
    Loads the AI4I 2020 dataset from the specified filepath and verifies schema integrity.

    Args:
        filepath: Relative or absolute path to the CSV file.

    Returns:
        pd.DataFrame: Validated dataframe.

    Raises:
        FileNotFoundError: If the CSV file cannot be located.
        ValueError: If mandatory columns are missing from the file.
    """
    # Check fallback paths if default relative path is not found directly
    resolved_path = filepath
    if not os.path.exists(resolved_path):
        # Check root directory or current working directory fallback
        alt_path = os.path.basename(filepath)
        if os.path.exists(alt_path):
            resolved_path = alt_path
        else:
            raise FileNotFoundError(f"Dataset not found at '{filepath}' or '{alt_path}'.")

    logger.info(f"Loading dataset from: {resolved_path}")
    df = pd.read_csv(resolved_path)

    # Schema integrity check
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset schema mismatch. Missing mandatory columns: {missing_cols}")

    logger.info(f"Successfully loaded dataset with shape: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def audit_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs a thorough data quality audit covering:
    - Dimensions and missing values
    - Exact and identifier duplicates
    - Categorical variant distributions
    - Target imbalance and failure mode decomposition
    - Physical and mechanical boundary plausibility
    - Statistical summaries of continuous operating features

    Args:
        df: Input dataframe to audit.

    Returns:
        Dict[str, Any]: Comprehensive audit results dictionary.
    """
    logger.info("Executing comprehensive data quality audit...")

    # 1. Dimensions and Missingness
    total_rows, total_cols = df.shape
    missing_per_col = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())

    # 2. Duplicate Detection
    exact_duplicates = int(df.duplicated().sum())
    udi_duplicates = int(df["UDI"].duplicated().sum())
    product_id_duplicates = int(df["Product ID"].duplicated().sum())

    # 3. Categorical distribution ('Type')
    type_counts = df["Type"].value_counts().to_dict()
    type_proportions = (df["Type"].value_counts(normalize=True) * 100).round(2).to_dict()

    # 4. Target distribution ('Machine failure')
    target_counts = {int(k): int(v) for k, v in df["Machine failure"].value_counts().to_dict().items()}
    target_proportions = {
        int(k): float(v)
        for k, v in (df["Machine failure"].value_counts(normalize=True) * 100).round(2).to_dict().items()
    }
    imbalance_ratio = round(target_counts.get(0, 0) / max(target_counts.get(1, 1), 1), 2)

    # 5. Failure modes breakdown and consistency
    failure_mode_counts = {col: int(df[col].sum()) for col in FAILURE_MODES}
    failure_mode_rates = {col: round(float(df[col].mean() * 100), 2) for col in FAILURE_MODES}
    any_mode_active = (df[FAILURE_MODES].sum(axis=1) > 0).astype(int)
    rows_with_any_mode = int(any_mode_active.sum())
    rows_with_target_failure = int(df["Machine failure"].sum())

    # Discrepancy analysis
    discrepancy_rnf_only = int(
        ((df["Machine failure"] == 0) & (df["RNF"] == 1) & (df[["TWF", "HDF", "PWF", "OSF"]].sum(axis=1) == 0)).sum()
    )
    discrepancy_target_no_modes = int(
        ((df["Machine failure"] == 1) & (df[FAILURE_MODES].sum(axis=1) == 0)).sum()
    )

    # 6. Physical boundary & plausibility checks
    bound_violations = {}
    for col, (lower_b, upper_b) in PHYSICAL_BOUNDS.items():
        if col in df.columns:
            out_of_bounds = int(((df[col] < lower_b) | (df[col] > upper_b)).sum())
            bound_violations[col] = out_of_bounds

    # Thermal physics check: Process temperature should exceed ambient Air temperature
    temp_inversion_count = int((df["Process temperature [K]"] <= df["Air temperature [K]"]).sum())

    # 7. Descriptive statistics for numerical variables
    num_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    numerical_summary = {}
    for col in num_cols:
        numerical_summary[col] = {
            "mean": round(float(df[col].mean()), 4),
            "std": round(float(df[col].std()), 4),
            "min": round(float(df[col].min()), 4),
            "p25": round(float(df[col].quantile(0.25)), 4),
            "median": round(float(df[col].median()), 4),
            "p75": round(float(df[col].quantile(0.75)), 4),
            "max": round(float(df[col].max()), 4),
        }

    audit_results = {
        "dataset_dimensions": {"rows": total_rows, "columns": total_cols},
        "missing_values": {"total_missing": total_missing, "by_column": missing_per_col},
        "duplicate_records": {
            "exact_duplicate_rows": exact_duplicates,
            "duplicate_udi": udi_duplicates,
            "duplicate_product_id": product_id_duplicates,
        },
        "type_distribution": {"counts": type_counts, "percentages": type_proportions},
        "target_distribution": {
            "counts": target_counts,
            "percentages": target_proportions,
            "imbalance_ratio": f"{imbalance_ratio}:1 (Normal:Failure)",
        },
        "failure_modes_summary": {
            "counts": failure_mode_counts,
            "percentages": failure_mode_rates,
            "rows_with_any_mode": rows_with_any_mode,
            "rows_with_target_failure": rows_with_target_failure,
            "discrepancies": {
                "rnf_active_but_no_machine_failure": discrepancy_rnf_only,
                "machine_failure_active_with_zero_failure_modes": discrepancy_target_no_modes,
            },
        },
        "plausibility_and_physics_checks": {
            "boundary_violations": bound_violations,
            "temperature_inversions_count": temp_inversion_count,
            "all_physics_checks_passed": (
                sum(bound_violations.values()) == 0 and temp_inversion_count == 0
            ),
        },
        "numerical_feature_statistics": numerical_summary,
    }

    logger.info("Data quality audit completed successfully.")
    return audit_results


def save_data_quality_report(audit_results: Dict[str, Any], output_path: str = "outputs/metrics/data_quality_report.json") -> None:
    """
    Serializes audit results to JSON for reproducibility and documentation.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(audit_results, f, indent=4)
    logger.info(f"Saved data quality report to: {output_path}")


def print_data_quality_summary(audit_results: Dict[str, Any]) -> None:
    """
    Prints a clean human-readable summary of the audit findings to stdout.
    """
    print("=" * 70)
    print("DATA QUALITY & INTEGRITY AUDIT SUMMARY")
    print("=" * 70)
    dims = audit_results["dataset_dimensions"]
    print(f"Dimensions: {dims['rows']:,} rows | {dims['columns']} columns")
    print(f"Total Missing Values: {audit_results['missing_values']['total_missing']}")
    dups = audit_results["duplicate_records"]
    print(f"Duplicate Rows: {dups['exact_duplicate_rows']} | Duplicate UDI: {dups['duplicate_udi']} | Duplicate Product ID: {dups['duplicate_product_id']}")

    print("\n[Product Type Distribution]")
    for k, v in audit_results["type_distribution"]["counts"].items():
        pct = audit_results["type_distribution"]["percentages"][k]
        print(f"  Type {k}: {v:,} ({pct:.2f}%)")

    print("\n[Target Variable: Machine failure]")
    for k, v in audit_results["target_distribution"]["counts"].items():
        label = "Normal (0)" if k == 0 else "Failure (1)"
        pct = audit_results["target_distribution"]["percentages"][k]
        print(f"  {label}: {v:,} ({pct:.2f}%)")
    print(f"  Class Imbalance: {audit_results['target_distribution']['imbalance_ratio']}")

    print("\n[Failure Modes Breakdown]")
    for mode, cnt in audit_results["failure_modes_summary"]["counts"].items():
        pct = audit_results["failure_modes_summary"]["percentages"][mode]
        print(f"  {mode}: {cnt} ({pct:.2f}%)")

    discrep = audit_results["failure_modes_summary"]["discrepancies"]
    print("\n[Target Leakage & Discrepancy Findings]")
    print(f"  RNF=1 but Machine failure=0: {discrep['rnf_active_but_no_machine_failure']} occurrences")
    print(f"  Machine failure=1 but all modes=0: {discrep['machine_failure_active_with_zero_failure_modes']} occurrences")
    print("  Note: Failure modes (TWF, HDF, PWF, OSF, RNF) must be strictly EXCLUDED")
    print("        from model features to prevent severe target leakage.")

    phys = audit_results["plausibility_and_physics_checks"]
    print(f"\n[Physical & Engineering Plausibility]")
    print(f"  Process Temp <= Air Temp Inversions: {phys['temperature_inversions_count']}")
    print(f"  Physical Boundary Violations: {sum(phys['boundary_violations'].values())}")
    print(f"  Status: {'PASSED (Zero physical contradictions)' if phys['all_physics_checks_passed'] else 'FLAGGED'}")
    print("=" * 70)


if __name__ == "__main__":
    df = load_data()
    audit = audit_data_quality(df)
    save_data_quality_report(audit)
    print_data_quality_summary(audit)
