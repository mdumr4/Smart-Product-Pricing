import os
from pathlib import Path
import logging
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

logger = logging.getLogger("dataset")


class MultimodalProductDataset(Dataset):
    """
    PyTorch Dataset providing aligned Text, Vision, Tabular, and Log-Price tensors.
    Safely falls back to zero-tensors if product images are missing or unavailable.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        image_dir: str,
        split: str = "train",
        unit_vocab: dict = None,
        tabular_stats: dict = None,
        max_text_len: int = 128,
    ):
        self.df = dataframe.copy().reset_index(drop=True)
        self.tokenizer = tokenizer
        self.image_dir = Path(image_dir)
        self.split = split
        self.max_text_len = max_text_len

        # Numerical Features
        self.numerical_cols = ["pack_size", "quantity_value"]
        for col in self.numerical_cols:
            if col not in self.df.columns:
                self.df[col] = 1.0
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(1.0)

        # Tabular Z-Score Stats
        if tabular_stats is None:
            self.tabular_stats = {
                "mean": {c: float(self.df[c].mean()) for c in self.numerical_cols},
                "std": {c: float(self.df[c].std()) if self.df[c].std() > 0 else 1.0 for c in self.numerical_cols},
            }
        else:
            self.tabular_stats = tabular_stats

        # Categorical Unit Vocabulary
        if unit_vocab is None:
            unique_units = self.df["quantity_unit"].astype(str).unique().tolist()
            self.unit_vocab = {unit: i + 1 for i, unit in enumerate(unique_units)}
            self.unit_vocab["__unknown__"] = 0
        else:
            self.unit_vocab = unit_vocab
        self.unit_to_idx = self.unit_vocab

        # Image Transforms
        self.train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.eval_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # 1. Text Modality
        text = str(row.get("item_name", "product"))
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_text_len,
            truncation=True,
            padding="max_length",
        )

        # 2. Vision Modality (Zero Tensor Fallback)
        pixel_values = torch.zeros((3, 224, 224), dtype=torch.float32)
        img_link = str(row.get("image_link", ""))
        if img_link and img_link.startswith("http"):
            img_path = self.image_dir / Path(img_link).name
            if img_path.exists():
                try:
                    img = Image.open(img_path).convert("RGB")
                    pixel_values = self.train_transforms(img) if self.split == "train" else self.eval_transforms(img)
                except Exception:
                    pass

        # 3. Tabular Modality (Z-score normalized)
        tab_vals = []
        for col in self.numerical_cols:
            mean = self.tabular_stats["mean"].get(col, 1.0)
            std = self.tabular_stats["std"].get(col, 1.0)
            norm_val = (float(row[col]) - mean) / (std + 1e-8)
            tab_vals.append(norm_val)
        tabular_features = torch.tensor(tab_vals, dtype=torch.float32)

        unit = str(row.get("quantity_unit", "__unknown__"))
        unit_idx = torch.tensor(self.unit_to_idx.get(unit, 0), dtype=torch.long)

        # 4. Target Price (log1p space)
        if "price" in row and not pd.isna(row["price"]):
            raw_p = float(row["price"])
            price_tensor = torch.tensor(np.log1p(max(raw_p, 0.01)), dtype=torch.float32)
        else:
            price_tensor = torch.tensor(0.0, dtype=torch.float32)

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": pixel_values,
            "tabular_features": tabular_features,
            "unit_idx": unit_idx,
            "price": price_tensor,
            "sample_id": row.get("sample_id", idx),
        }
