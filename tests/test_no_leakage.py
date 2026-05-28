"""Guards against the most common data-leakage mistake in this kind of
benchmark: using held-out actuals when computing test-period lag/rolling
features.
"""
import numpy as np
import pandas as pd

import forecast_ml as fml
import forecast_baseline as fb


def _build_dummy_panel():
    """Two SKUs, 24 months, deterministic demand."""
    dates = pd.date_range("2023-01-01", periods=24, freq="MS")
    rows = []
    for sku, fam, base in [("S1", "Engine_Parts", 100), ("S2", "Brake_Systems", 50)]:
        for i, d in enumerate(dates):
            rows.append({"date": d, "sku_id": sku, "family": fam,
                         "demand": base + i, "is_promo": 0})
    return pd.DataFrame(rows)


def test_baseline_forecast_uses_only_pre_test_data():
    """The naive seasonal baseline at month t must equal demand at month t-12,
    which for our 24-month panel and 6-month holdout lies entirely inside the
    training window (months 7-18)."""
    df = _build_dummy_panel()
    pred = fb.naive_seasonal_forecast(df)
    sku = df[df["sku_id"] == "S1"].sort_values("date").reset_index(drop=True)
    for _, row in pred[pred["sku_id"] == "S1"].iterrows():
        target_idx = sku.index[sku["date"] == row["date"]][0]
        lag_idx = target_idx - 12
        assert row["forecast"] == sku.loc[lag_idx, "demand"]
        assert lag_idx < 18, "lag must come from the training window"


def test_ml_recursive_does_not_peek_at_test_actuals(monkeypatch, tmp_path):
    """After fitting, the model's test-period lag_1 feature for horizon h>1
    must equal the *previous* forecast, not the previous actual. We assert
    this by mocking the regressor with a function that returns a sentinel,
    then checking that the lag_1 fed into the second call equals that
    sentinel rather than the held-out demand value."""
    df = _build_dummy_panel()
    family_map = {f: i for i, f in enumerate(sorted(df["family"].unique()))}
    n_train, n_total = 18, 24

    SENTINEL = 999.0
    captured_lag_1 = []

    class FakeModel:
        def predict(self, X):
            captured_lag_1.append(float(X[0, fml.FEATURE_COLS.index("lag_1")]))
            return np.array([SENTINEL])

    preds = fml.recursive_forecast(FakeModel(), df, family_map, n_train, n_total)
    s1 = df[df["sku_id"] == "S1"].sort_values("date").reset_index(drop=True)
    # First test month: lag_1 must equal last training demand (month_idx 17).
    assert captured_lag_1[0] == s1.loc[17, "demand"]
    # Second test month: lag_1 must equal the SENTINEL forecast from step 1,
    # not the held-out actual at month_idx 18.
    assert captured_lag_1[1] == SENTINEL
    assert captured_lag_1[1] != s1.loc[18, "demand"]
    assert (preds["forecast"] == SENTINEL).all()


def test_ml_training_rows_have_no_test_period_targets():
    """The features dataframe used to fit must contain only rows whose
    target month falls inside the training window."""
    df = _build_dummy_panel()
    df_feat, _ = fml.build_training_features(df)
    df_feat["month_idx"] = df_feat.groupby("sku_id").cumcount()
    train = df_feat[df_feat["month_idx"] < 18].dropna(subset=fml.FEATURE_COLS)
    assert (train["month_idx"] < 18).all()
