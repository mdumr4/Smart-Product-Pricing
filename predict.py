#!/usr/bin/env python3
"""
Production CLI Entry Point for Smart Product Pricing.
Predicts product prices from multimodal inputs (catalog text, packaging images, and tabular metadata)
using a high-accuracy Two-Model Ensemble (Tri-Modal Cross-Attention GMU Neural Net + LightGBM).
"""

import sys
import argparse
from pathlib import Path

# Add repo root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.predictor import PricePredictor
from src.config import PricingConfig


def main():
    parser = argparse.ArgumentParser(
        description="Smart Product Pricing - Production Multimodal Price Prediction Engine"
    )
    parser.add_argument(
        "--input",
        default="./data/sample_products.csv",
        help="Input CSV or Parquet file containing product catalog listings (default: ./data/sample_products.csv)",
    )
    parser.add_argument(
        "--out",
        default="./output/predictions.csv",
        help="Output CSV path for predicted prices (default: ./output/predictions.csv)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run high-speed tabular LightGBM mode (< 2 seconds runtime)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run neural inference on (cpu, cuda, default: auto-detected)",
    )
    args = parser.parse_args()

    cfg = PricingConfig()
    if args.device:
        cfg.device = args.device

    predictor = PricePredictor(cfg)
    predictor.predict(
        input_data=args.input,
        output_csv=args.out,
        fast_mode=args.fast,
        validate=True,
    )


if __name__ == "__main__":
    main()
