import os
import sys
import torch
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("config")


def get_default_device() -> str:
    """Auto-detect CUDA GPU or fallback to CPU."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def find_first_existing(candidates: List[str], base_dir: Optional[Path] = None) -> str:
    """Finds the first existing path among candidates."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent

    for candidate in candidates:
        p = (base_dir / candidate).resolve()
        if p.exists():
            return str(p)
        p_lit = Path(candidate).resolve()
        if p_lit.exists():
            return str(p_lit)

    return str((base_dir / candidates[0]).resolve())


@dataclass
class PricingConfig:
    """Production configuration parameters for Smart Product Pricing Pipeline."""

    # Project Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    output_dir: Path = field(init=False)

    # Data File Paths
    train_data_path: str = field(default="")
    test_data_path: str = field(default="")
    sample_data_path: str = field(default="")
    images_dir: str = field(default="")

    # Model Checkpoint Paths
    nn_checkpoint_path: str = field(default="")
    lgbm_model_path: str = field(default="")
    unit_encoder_path: str = field(default="")
    unit_vocab_path: str = field(default="")
    tabular_stats_path: str = field(default="")

    # Hardware & Compute
    device: str = field(default_factory=get_default_device)
    num_workers: int = field(default_factory=lambda: min(os.cpu_count() or 2, 4))
    batch_size: int = 64
    eval_batch_size: int = 128

    # Model Hyperparameters
    text_model_name: str = "bert-base-uncased"
    image_model_name: str = "facebook/dinov2-base"
    unit_embedding_dim: int = 16
    tabular_hidden_dim: int = 128
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 10
    dropout: float = 0.30

    # Ensemble Weights (Log-Space Blend)
    weight_nn: float = 0.50
    weight_lgbm: float = 0.50
    min_price_floor: float = 0.01

    def __post_init__(self):
        self.data_dir = (self.base_dir / "data").resolve()
        self.models_dir = (self.base_dir / "models").resolve()
        self.output_dir = (self.base_dir / "output").resolve()

        # Resolve Data Paths
        if not self.train_data_path:
            self.train_data_path = find_first_existing([
                "data/train_features.parquet",
                "data/train_features.csv",
                "data/sample_products.csv",
            ], self.base_dir)

        if not self.test_data_path:
            self.test_data_path = find_first_existing([
                "data/test_features.parquet",
                "data/test_features.csv",
                "data/sample_products.csv",
            ], self.base_dir)

        if not self.sample_data_path:
            self.sample_data_path = str((self.data_dir / "sample_products.csv").resolve())

        if not self.images_dir:
            self.images_dir = str((self.data_dir / "images").resolve())

        # Resolve Model Paths
        if not self.nn_checkpoint_path:
            self.nn_checkpoint_path = find_first_existing([
                "models/best.pth",
                "best.pth",
            ], self.base_dir)

        if not self.lgbm_model_path:
            self.lgbm_model_path = find_first_existing([
                "models/lightgbm_model.pkl",
                "lightgbm_model.pkl",
            ], self.base_dir)

        if not self.unit_encoder_path:
            self.unit_encoder_path = find_first_existing([
                "models/unit_label_encoder.pkl",
                "unit_label_encoder.pkl",
            ], self.base_dir)

        if not self.unit_vocab_path:
            self.unit_vocab_path = find_first_existing([
                "models/unit_vocab.json",
                "unit_vocab.json",
            ], self.base_dir)

        if not self.tabular_stats_path:
            self.tabular_stats_path = find_first_existing([
                "models/tabular_stats.json",
                "tabular_stats.json",
            ], self.base_dir)


default_config = PricingConfig()
