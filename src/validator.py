import pandas as pd
from typing import List, Tuple, Any, Optional


def validate_predictions_df(df: pd.DataFrame, expected_ids: List[Any] = None) -> List[str]:
    """
    Validates submission prediction dataframe schema, non-null values, and positive pricing.
    """
    errors = []

    # Check required columns
    required_cols = ["sample_id", "price"]
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing required column: '{col}'")

    if errors:
        return errors

    # Check row count
    if len(df) == 0:
        errors.append("Prediction dataframe is empty (0 rows).")

    # Check for NaN / null values
    if df["price"].isnull().any():
        null_count = int(df["price"].isnull().sum())
        errors.append(f"Found {null_count} null/NaN values in 'price' column.")

    # Check strictly positive price values
    negative_count = int((df["price"] <= 0).sum())
    if negative_count > 0:
        errors.append(f"Found {negative_count} non-positive prices (prices must be > 0.0).")

    # Check sample_id integrity if expected IDs provided
    if expected_ids is not None:
        if len(df) != len(expected_ids):
            errors.append(f"Row count mismatch: expected {len(expected_ids)}, got {len(df)}.")
        mismatched = set(expected_ids) - set(df["sample_id"].tolist())
        if mismatched:
            errors.append(f"Found {len(mismatched)} missing sample_ids in predictions.")

    return errors
