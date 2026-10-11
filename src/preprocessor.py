import re
import string
import logging
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger("preprocessor")


class CatalogPreprocessor:
    """
    Production preprocessor extracted from Amazon ML Challenge exploratory data analysis:
    Parses unstructured catalog_content, normalizes product units, cleans text, and normalizes tabular features.
    """

    UNIT_MAPPING = {
        "oz": "oz", "ounce": "oz", "ounces": "oz",
        "fl oz": "fl_oz", "fluid ounce": "fl_oz", "fluid ounces": "fl_oz", "fl ounce": "fl_oz", "fl": "fl_oz",
        "lb": "lb", "pound": "lb", "pounds": "lb", "lbs": "lb",
        "g": "g", "gram": "g", "grams": "g", "gramm": "g", "gr": "g", "kg": "kg", "k": "kg",
        "ml": "ml", "milliliter": "ml", "milliliters": "ml", "millilitre": "ml", "mililitro": "ml",
        "l": "l", "liter": "l", "liters": "l", "ltr": "l",
        "gallon": "gallon", "gallons": "gallon",
        "count": "count", "ct": "count", "each": "count", "unit": "count", "units": "count", "bottle": "count",
        "unknown": "na", "other": "na",
    }

    @staticmethod
    def clean_text(text: Any) -> str:
        """Removes emojis, punctuation, extra whitespace, and normalizes text."""
        if not isinstance(text, str) or not text.strip():
            return "other"

        # Remove emojis
        text = re.sub(r"[\U00010000-\U0010ffff]|[\u2600-\u26FF\u2700-\u27BF]", "", text)

        # Remove special bullet symbols and punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"[\•\–\—\✓]", "", text)

        # Remove extra whitespaces & lowercase
        text = re.sub(r"\s+", " ", text)
        return text.lower().strip() or "other"

    @staticmethod
    def parse_catalog_content(text: Any) -> Dict[str, str]:
        """
        Parses raw unstructured catalog text into structured item_name, description, value, and unit.
        """
        if not isinstance(text, str) or not text.strip():
            return {"item_name": "other", "description": "other", "value": "other", "unit": "na"}

        # Item Name
        match_name = re.search(r"Item Name:\s*(.*?)(?=Bullet Point 1:|Product Description:|Value:|$)", text, re.S)
        item_name = match_name.group(1).strip() if match_name else "other"

        # Product Description
        match_desc = re.search(r"Product Description:\s*(.*?)(?=Value:|$)", text, re.S)
        description = match_desc.group(1).strip() if match_desc else "other"

        # Value
        match_val = re.search(r"Value:\s*(.*?)(?=Unit:|$)", text, re.S)
        value = match_val.group(1).strip() if match_val else "other"

        # Unit
        match_unit = re.search(r"Unit:\s*(.*?)$", text, re.S)
        unit = match_unit.group(1).strip() if match_unit else "na"

        return {
            "item_name": item_name,
            "description": description,
            "value": value,
            "unit": unit,
        }

    @classmethod
    def normalize_unit(cls, unit_str: Any) -> str:
        """Maps diverse raw unit spellings to canonical unit tags."""
        if not isinstance(unit_str, str) or not unit_str.strip():
            return "na"
        u = unit_str.lower().strip()
        return cls.UNIT_MAPPING.get(u, u)

    @classmethod
    def transform_df(
        cls, df: pd.DataFrame, fit_stats: bool = False, stats: Optional[Dict] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        End-to-end preprocessing pipeline for a product dataframe.
        """
        df_out = df.copy()

        # If catalog_content is present but item_name isn't, parse it
        if "catalog_content" in df_out.columns and "item_name" not in df_out.columns:
            logger.info("Parsing catalog_content strings into structured fields...")
            parsed = df_out["catalog_content"].apply(cls.parse_catalog_content).tolist()
            df_parsed = pd.DataFrame(parsed, index=df_out.index)
            for col in ["item_name", "description", "value", "unit"]:
                if col not in df_out.columns:
                    df_out[col] = df_parsed[col]

        # Clean text
        if "item_name" in df_out.columns:
            df_out["item_name"] = df_out["item_name"].apply(cls.clean_text)

        # Standardize units
        if "quantity_unit" in df_out.columns:
            df_out["quantity_unit"] = df_out["quantity_unit"].apply(cls.normalize_unit)
        elif "unit" in df_out.columns:
            df_out["quantity_unit"] = df_out["unit"].apply(cls.normalize_unit)
        else:
            df_out["quantity_unit"] = "na"

        # Tabular numerical columns
        numerical_cols = ["pack_size", "quantity_value"]
        for col in numerical_cols:
            if col not in df_out.columns:
                df_out[col] = 1.0
            df_out[col] = pd.to_numeric(df_out[col], errors="coerce").fillna(1.0)

        # Compute or apply Z-score stats
        if fit_stats or stats is None:
            stats = {
                "mean": {col: float(df_out[col].mean()) for col in numerical_cols},
                "std": {col: float(df_out[col].std()) if df_out[col].std() > 0 else 1.0 for col in numerical_cols},
            }

        return df_out, stats
