import os
import json
import logging
from pathlib import Path
from typing import Optional, Union, Dict
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.config import PricingConfig, default_config
from src.preprocessor import CatalogPreprocessor
from src.dataset import MultimodalProductDataset
from src.models import TriModalCrossAttentionModel, LightGBMPriceModel
from src.validator import validate_predictions_df

logger = logging.getLogger("predictor")


class PricePredictor:
    """
    Production Multimodal Inference Engine for Smart Product Pricing.
    Supports Fast Mode (Tabular LightGBM < 2s) and Full Tri-Modal Ensemble Mode.
    """

    def __init__(self, config: Optional[PricingConfig] = None):
        self.config = config or default_config
        self.lgbm_model = LightGBMPriceModel(
            model_path=self.config.lgbm_model_path,
            encoder_path=self.config.unit_encoder_path,
        )
        self.nn_model: Optional[TriModalCrossAttentionModel] = None
        self.tokenizer = None
        self.unit_vocab = {}
        self.tabular_stats = {}

    def load_neural_network(self) -> bool:
        """Loads Tri-Modal Neural Network weights if available."""
        if not os.path.exists(self.config.nn_checkpoint_path):
            logger.info("Neural network checkpoint not found on disk. Operating in LightGBM-only mode.")
            return False

        try:
            logger.info(f"Loading Neural Network checkpoint: {self.config.nn_checkpoint_path}")
            if os.path.exists(self.config.unit_vocab_path):
                with open(self.config.unit_vocab_path, "r", encoding="utf-8") as f:
                    self.unit_vocab = json.load(f)

            if os.path.exists(self.config.tabular_stats_path):
                with open(self.config.tabular_stats_path, "r", encoding="utf-8") as f:
                    self.tabular_stats = json.load(f)

            self.tokenizer = AutoTokenizer.from_pretrained(self.config.text_model_name)
            self.nn_model = TriModalCrossAttentionModel(
                unit_vocab_size=len(self.unit_vocab) or 50,
                text_model_name=self.config.text_model_name,
                image_model_name=self.config.image_model_name,
            )
            state_dict = torch.load(self.config.nn_checkpoint_path, map_location=self.config.device)
            self.nn_model.load_state_dict(state_dict)
            self.nn_model.to(self.config.device)
            self.nn_model.eval()
            logger.info("Neural network successfully initialized.")
            return True
        except Exception as e:
            logger.warning(f"Failed to load neural network weights ({e}). Reverting to LightGBM mode.")
            self.nn_model = None
            return False

    def predict(
        self,
        input_data: Union[str, Path, pd.DataFrame],
        output_csv: Optional[Union[str, Path]] = None,
        fast_mode: bool = False,
        validate: bool = True,
    ) -> pd.DataFrame:
        """
        Runs price prediction pipeline on input dataframe or CSV file.
        """
        # 1. Load Data
        if isinstance(input_data, (str, Path)):
            input_path = Path(input_data)
            if input_path.suffix == ".parquet":
                df_raw = pd.read_parquet(input_path)
            else:
                df_raw = pd.read_csv(input_path)
        else:
            df_raw = input_data.copy()

        logger.info(f"Loaded {len(df_raw)} product rows for prediction.")

        # 2. Preprocessing
        df_clean, stats = CatalogPreprocessor.transform_df(df_raw)

        # 3. LightGBM Predictions (Log Space)
        log_preds_lgb = self.lgbm_model.predict_log(df_clean)

        # 4. Neural Network Predictions (Optional / Full Mode)
        if not fast_mode and self.nn_model is not None:
            logger.info("Executing Tri-Modal Neural Network forward inference...")
            dataset = MultimodalProductDataset(
                dataframe=df_clean,
                tokenizer=self.tokenizer,
                image_dir=self.config.images_dir,
                split="test",
                unit_vocab=self.unit_vocab,
                tabular_stats=self.tabular_stats or stats,
            )
            loader = DataLoader(
                dataset,
                batch_size=self.config.eval_batch_size,
                shuffle=False,
                num_workers=0,
            )

            nn_preds_list = []
            with torch.no_grad():
                for batch in loader:
                    input_ids = batch["input_ids"].to(self.config.device)
                    attention_mask = batch["attention_mask"].to(self.config.device)
                    pixel_values = batch["pixel_values"].to(self.config.device)
                    tabular_features = batch["tabular_features"].to(self.config.device)
                    unit_idx = batch["unit_idx"].to(self.config.device)

                    pred = self.nn_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        tabular_features=tabular_features,
                        unit_idx=unit_idx,
                    )
                    nn_preds_list.extend(pred.cpu().numpy().tolist())

            log_preds_nn = np.array(nn_preds_list)
            # Ensemble Blend in Log Space
            final_log_preds = (
                self.config.weight_nn * log_preds_nn + self.config.weight_lgbm * log_preds_lgb
            )
        else:
            logger.info("Running in Fast Mode (Pure Tabular LightGBM)...")
            final_log_preds = log_preds_lgb

        # 5. Inverse Log-Transform & Positive Pricing Safeguard
        final_prices = np.expm1(final_log_preds)
        final_prices = np.maximum(final_prices, self.config.min_price_floor)

        # 6. Assembly & Validation
        sample_ids = df_raw["sample_id"].values if "sample_id" in df_raw.columns else list(range(len(df_raw)))
        df_out = pd.DataFrame({
            "sample_id": sample_ids,
            "price": np.round(final_prices, 2),
        })

        if validate:
            errors = validate_predictions_df(df_out, expected_ids=sample_ids)
            if errors:
                raise ValueError(f"Prediction output validation failed: {errors[0]}")
            logger.info("Predictions passed all schema and positivity validation gates!")

        if output_csv:
            out_p = Path(output_csv)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            df_out.to_csv(out_p, index=False)
            logger.info(f"Saved predictions to: {out_p.resolve()}")

        return df_out
