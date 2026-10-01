"""Inventory dollar-impact arithmetic must match the documented formula:
    stockout = max(actual - forecast, 0) * unit_cost * 0.35
    overstock = max(forecast - actual, 0) * unit_cost * 0.20
"""
import pandas as pd
import sys
import os

# Ensure benchmark module is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import benchmark


def test_stockout_and_overstock_formula():
    pred = pd.DataFrame({
        "sku_id":   ["A", "B", "C"],
        "family":   ["Engine_Parts", "Engine_Parts", "Body_Panels"],
        "date":     pd.to_datetime(["2024-07-01"] * 3),
        "actual":   [100.0, 100.0, 50.0],
        "forecast": [120.0, 80.0, 50.0],  # overstock, stockout, exact
    })
    out = benchmark.estimate_inventory_impact(pred, "test_model")
    # Engine_Parts cost = 85; Body_Panels = 95.
    # Row A: overstock 20 * 85 * 0.20 = 340; stockout 0
    # Row B: stockout 20 * 85 * 0.35 = 595; overstock 0
    # Row C: zero error -> zero
    engine = out[out["family"] == "Engine_Parts"].iloc[0]
    assert engine["overstock_cost"] == 340.0
    assert engine["stockout_cost"] == 595.0
    assert engine["total_impact"] == 935.0

    body = out[out["family"] == "Body_Panels"].iloc[0]
    assert body["overstock_cost"] == 0.0
    assert body["stockout_cost"] == 0.0
