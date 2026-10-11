import torch
import numpy as np
from src.metrics import SmoothSMAPELoss, calculate_metrics


def test_calculate_metrics():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([10.0, 20.0, 30.0])
    res = calculate_metrics(y_true, y_pred)
    assert res["SMAPE"] < 0.01
    assert res["MAE"] == 0.0
    assert res["RMSE"] == 0.0


def test_calculate_metrics_divergence():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([20.0, 10.0])
    res = calculate_metrics(y_true, y_pred)
    assert res["SMAPE"] > 50.0
    assert res["MAE"] == 10.0


def test_smooth_smape_loss_gradient():
    loss_fn = SmoothSMAPELoss()
    y_pred = torch.tensor([12.0, 25.0], requires_grad=True)
    y_true = torch.tensor([10.0, 20.0])
    loss = loss_fn(y_pred, y_true)
    assert loss.item() > 0.0
    loss.backward()
    assert y_pred.grad is not None
