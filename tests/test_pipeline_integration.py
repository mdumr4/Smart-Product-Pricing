from pathlib import Path
import pandas as pd
from src.predictor import PricePredictor
from src.config import PricingConfig
from src.validator import validate_predictions_df


def test_fast_mode_prediction_pipeline():
    repo_root = Path(__file__).resolve().parent.parent
    sample_file = repo_root / "data" / "sample_products.csv"
    tmp_out = repo_root / "tests" / "tmp_test_predictions.csv"

    if tmp_out.exists():
        tmp_out.unlink()

    try:
        predictor = PricePredictor()
        df_pred = predictor.predict(
            input_data=sample_file,
            output_csv=tmp_out,
            fast_mode=True,
            validate=True,
        )

        assert isinstance(df_pred, pd.DataFrame)
        assert len(df_pred) == 50
        assert list(df_pred.columns) == ["sample_id", "price"]
        assert (df_pred["price"] > 0).all()
        assert tmp_out.exists()

        errors = validate_predictions_df(df_pred)
        assert len(errors) == 0
    finally:
        if tmp_out.exists():
            tmp_out.unlink()
