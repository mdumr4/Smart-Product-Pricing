import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error


class SmoothSMAPELoss(nn.Module):
    """
    Differentiable approximation of Symmetric Mean Absolute Percentage Error (SMAPE)
    used for neural network regression optimization.
    """

    def __init__(self, epsilon: float = 1e-2):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Prevent negative predictions in log-space/linear space
        y_pred = torch.relu(y_pred)
        numerator = torch.abs(y_pred - y_true)
        denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2.0
        loss = torch.mean(numerator / (denominator + self.epsilon))
        return loss


def calculate_metrics(
    y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]
) -> Dict[str, float]:
    """
    Computes official competition metrics: SMAPE (percentage), MAE, and RMSE.
    Safeguards negative values by clamping to positive floor.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_pred = np.clip(y_pred, a_min=0.01, a_max=None)

    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape = float(np.mean(numerator / (denominator + 1e-8)) * 100.0)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    return {"SMAPE": round(smape, 4), "MAE": round(mae, 4), "RMSE": round(rmse, 4)}
