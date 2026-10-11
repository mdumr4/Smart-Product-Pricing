import os
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from transformers import AutoModel, BertModel, RobertaModel
import lightgbm as lgb


class GatedMultimodalUnit(nn.Module):
    """Gated Multimodal Unit (GMU) dynamically weighting multi-source feature representations."""

    def __init__(self, input_dim_1: int, input_dim_2: int, output_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(input_dim_1, output_dim)
        self.linear_2 = nn.Linear(input_dim_2, output_dim)
        self.linear_gate_1 = nn.Linear(input_dim_1, output_dim)
        self.linear_gate_2 = nn.Linear(input_dim_2, output_dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        h1 = torch.tanh(self.linear_1(x1))
        h2 = torch.tanh(self.linear_2(x2))
        z = torch.sigmoid(self.linear_gate_1(x1) + self.linear_gate_2(x2))
        return z * h1 + (1.0 - z) * h2


class TriModalCrossAttentionModel(nn.Module):
    """
    Deep Tri-Modal Neural Network combining:
    1. Text Tower (BERT/RoBERTa)
    2. Vision Tower (DINOv2 ViT)
    3. Tabular Tower (Unit Embedding + MLP)
    4. Text-Image Cross-Attention
    5. Gated Multimodal Unit (GMU) Fusion
    6. Regression Head predicting log1p(price)
    """

    def __init__(
        self,
        unit_vocab_size: int = 50,
        num_numerical_features: int = 2,
        unit_embedding_dim: int = 16,
        text_model_name: str = "bert-base-uncased",
        image_model_name: str = "facebook/dinov2-base",
        tabular_hidden_dim: int = 128,
        pretrained: bool = True,
    ):
        super().__init__()

        # Text Encoder
        if pretrained:
            if "roberta" in text_model_name.lower():
                self.text_encoder = RobertaModel.from_pretrained(text_model_name)
            else:
                self.text_encoder = BertModel.from_pretrained(text_model_name)
            text_hidden_size = self.text_encoder.config.hidden_size
        else:
            text_hidden_size = 768
            self.text_encoder = nn.Linear(text_hidden_size, text_hidden_size)

        # Image Encoder
        if pretrained:
            self.image_encoder = AutoModel.from_pretrained(image_model_name)
            image_hidden_size = self.image_encoder.config.hidden_size
        else:
            image_hidden_size = 768
            self.image_encoder = nn.Linear(image_hidden_size, image_hidden_size)

        # Vision Projection to match text dimension
        if text_hidden_size != image_hidden_size:
            self.image_proj = nn.Linear(image_hidden_size, text_hidden_size)
        else:
            self.image_proj = nn.Identity()

        # Tabular Tower
        self.unit_embedding = nn.Embedding(max(unit_vocab_size, 10), unit_embedding_dim)
        self.tabular_mlp = nn.Sequential(
            nn.Linear(num_numerical_features + unit_embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, tabular_hidden_dim),
        )

        # Multihead Cross-Attention (Query = Text, Key/Value = Image)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=text_hidden_size, num_heads=8, batch_first=True
        )
        self.text_image_proj = nn.Linear(text_hidden_size * 2, tabular_hidden_dim)

        # Gated Multimodal Unit
        self.gmu = GatedMultimodalUnit(
            input_dim_1=tabular_hidden_dim,
            input_dim_2=tabular_hidden_dim,
            output_dim=tabular_hidden_dim,
        )

        # Regression Head
        self.regression_head = nn.Sequential(
            nn.Linear(tabular_hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        tabular_features: torch.Tensor,
        unit_idx: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Text Embeddings
        if hasattr(self.text_encoder, "config"):
            text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_embeds = text_out.last_hidden_state
        else:
            text_embeds = input_ids.float()

        # 2. Image Embeddings
        if hasattr(self.image_encoder, "config"):
            img_out = self.image_encoder(pixel_values=pixel_values)
            img_embeds = self.image_proj(img_out.last_hidden_state)
        else:
            img_embeds = pixel_values.float()

        # 3. Tabular Embeddings
        unit_emb = self.unit_embedding(unit_idx)
        tab_inp = torch.cat([tabular_features, unit_emb], dim=-1)
        tab_embeds = self.tabular_mlp(tab_inp)

        # 4. Cross-Attention Text + Image
        attn_out, _ = self.cross_attention(query=text_embeds, key=img_embeds, value=img_embeds)
        fused_text_img = torch.cat([text_embeds[:, 0, :], attn_out[:, 0, :]], dim=-1)
        proj_text_img = self.text_image_proj(fused_text_img)

        # 5. Gated Multimodal Unit Fusion
        final_rep = self.gmu(proj_text_img, tab_embeds)

        # 6. Predict log1p price
        pred_log_price = self.regression_head(final_rep).squeeze(-1)
        return pred_log_price


class LightGBMPriceModel:
    """Production Tabular Gradient Boosting Model for high-speed price prediction."""

    def __init__(self, model_path: Optional[str] = None, encoder_path: Optional[str] = None):
        self.model = None
        self.encoder = None
        self.features = ["pack_size", "quantity_value", "quantity_unit"]

        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
        if encoder_path and os.path.exists(encoder_path):
            self.encoder = joblib.load(encoder_path)

    def fit(self, df: pd.DataFrame, target_col: str = "price"):
        """Trains LightGBM regressor on log1p prices."""
        df_train = df.copy()
        y = np.log1p(df_train[target_col].clip(lower=0.01))

        # Encode quantity_unit
        from sklearn.preprocessing import LabelEncoder
        self.encoder = LabelEncoder()
        df_train["quantity_unit"] = self.encoder.fit_transform(df_train["quantity_unit"].astype(str))

        X = df_train[self.features]
        self.model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X, y)

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        """Predicts prices in log space."""
        if self.model is None:
            # Fallback heuristic if untrained
            return np.log1p(np.maximum(df["pack_size"].fillna(1.0).values * 10.0, 1.0))

        df_feat = df.copy()
        if self.encoder:
            # Map unseen units to first known class
            known_classes = set(self.encoder.classes_)
            df_feat["quantity_unit"] = df_feat["quantity_unit"].astype(str).apply(
                lambda u: u if u in known_classes else self.encoder.classes_[0]
            )
            df_feat["quantity_unit"] = self.encoder.transform(df_feat["quantity_unit"])
        else:
            df_feat["quantity_unit"] = 0

        X = df_feat[self.features]
        return self.model.predict(X)
