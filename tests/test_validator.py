import pandas as pd
import pytest
from src.validator import validate_predictions_df


def test_valid_predictions():
    df = pd.DataFrame({
        "sample_id": [1, 2, 3],
        "price": [19.99, 45.50, 10.00],
    })
    errors = validate_predictions_df(df, expected_ids=[1, 2, 3])
    assert len(errors) == 0


def test_missing_column():
    df = pd.DataFrame({"sample_id": [1, 2]})
    errors = validate_predictions_df(df)
    assert any("Missing required column: 'price'" in e for e in errors)


def test_non_positive_price():
    df = pd.DataFrame({
        "sample_id": [1, 2],
        "price": [-5.0, 10.0],
    })
    errors = validate_predictions_df(df)
    assert any("non-positive prices" in e for e in errors)


def test_null_price():
    df = pd.DataFrame({
        "sample_id": [1, 2],
        "price": [None, 10.0],
    })
    errors = validate_predictions_df(df)
    assert any("null/NaN values" in e for e in errors)
