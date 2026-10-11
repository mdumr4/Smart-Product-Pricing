#!/usr/bin/env python3
"""
Production Training Script for Smart Product Pricing Ensemble.
Trains LightGBM metadata model and Tri-Modal Cross-Attention Neural Network with SmoothSMAPELoss.
"""

import sys
import os
import json
import joblib
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import PricingConfig
from src.preprocessor import CatalogPreprocessor
from src.models import LightGBMPriceModel
from src.metrics import calculate_metrics


def main():
    parser = argparse.ArgumentParser(description="Train Smart Product Pricing Models")
    parser.add_argument(
        "--data",
        default="./data/sample_products.csv",
        help="Path to training features file (CSV or Parquet)",
    )
    parser.add_argument(
        "--output-dir",
        default="./models",
        help="Directory to save trained model artifacts",
    )
    parser.add_argument(
        "--train-lgbm",
        action="store_true",
        default=True,
        help="Train LightGBM tabular model",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading training data from {args.data}...")
    data_path = Path(args.data)
    if data_path.suffix == ".parquet":
        df_raw = pd.read_parquet(data_path)
    else:
        df_raw = pd.read_csv(data_path)

    # Preprocessing
    df_clean, stats = CatalogPreprocessor.transform_df(df_raw, fit_stats=True)

    # Save stats
    with open(out_dir / "tabular_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved tabular normalization stats to {out_dir / 'tabular_stats.json'}")

    # Build and save unit vocab
    unique_units = df_clean["quantity_unit"].astype(str).unique().tolist()
    unit_vocab = {unit: i + 1 for i, unit in enumerate(unique_units)}
    unit_vocab["__unknown__"] = 0
    with open(out_dir / "unit_vocab.json", "w", encoding="utf-8") as f:
        json.dump(unit_vocab, f, indent=2)
    print(f"Saved unit vocabulary ({len(unit_vocab)} units) to {out_dir / 'unit_vocab.json'}")

    # Train LightGBM Model
    if args.train_lgbm:
        print("
=== Training LightGBM Tabular Regressor ===")
        lgb_model = LightGBMPriceModel()
        lgb_model.fit(df_clean, target_col="price")

        # Save LightGBM artifacts
        joblib.dump(lgb_model.model, out_dir / "lightgbm_model.pkl")
        joblib.dump(lgb_model.encoder, out_dir / "unit_label_encoder.pkl")
        print(f"Saved LightGBM model to {out_dir / 'lightgbm_model.pkl'}")
        print(f"Saved label encoder to {out_dir / 'unit_label_encoder.pkl'}")

        # Compute in-sample / validation metrics
        log_preds = lgb_model.predict_log(df_clean)
        preds = np.expm1(log_preds)
        metrics = calculate_metrics(df_clean["price"].values, preds)
        print(f"LightGBM Training Metrics: SMAPE: {metrics['SMAPE']}%, MAE: ${metrics['MAE']}, RMSE: ${metrics['RMSE']}")

    print("
Training completed successfully!")


if __name__ == "__main__":
    main()
