"""Sanity checks on the MAPE/WAPE/RMSE math used in compute_metrics."""
import numpy as np
import pandas as pd

import forecast_baseline as fb


def _toy_predictions():
    return pd.DataFrame({
        "sku_id":   ["A", "A", "B", "B"],
        "family":   ["F1", "F1", "F2", "F2"],
        "date":     pd.to_datetime(["2024-07-01", "2024-08-01"] * 2),
        "actual":   [100.0, 200.0, 50.0, 80.0],
        "forecast": [110.0, 180.0, 40.0, 100.0],
    })


def test_overall_metrics_match_formula():
    pred = _toy_predictions()
    _, _, overall = fb.compute_metrics(pred)
    # APE = [10/100, 20/200, 10/50, 20/80] = [0.10, 0.10, 0.20, 0.25]
    # MAPE = mean * 100 = 16.25
    assert overall["MAPE"] == 16.25
    # WAPE = sum(|err|) / sum(actual) * 100 = 60 / 430 * 100
    assert abs(overall["WAPE"] - 60 / 430 * 100) < 1e-9
    # RMSE = sqrt(mean([10^2, 20^2, 10^2, 20^2]))
    assert abs(overall["RMSE"] - np.sqrt((100 + 400 + 100 + 400) / 4)) < 1e-9


def test_metrics_skip_zero_actuals():
    pred = pd.DataFrame({
        "sku_id":   ["A", "A"],
        "family":   ["F1", "F1"],
        "date":     pd.to_datetime(["2024-07-01", "2024-08-01"]),
        "actual":   [0.0, 100.0],
        "forecast": [10.0, 110.0],
    })
    _, _, overall = fb.compute_metrics(pred)
    # Only the row with actual=100 contributes
    assert abs(overall["MAPE"] - 10.0) < 1e-9
    assert abs(overall["WAPE"] - 10.0) < 1e-9
    assert abs(overall["RMSE"] - 10.0) < 1e-9
