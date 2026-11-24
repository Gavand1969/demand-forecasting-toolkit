"""
forecast_sarima.py
------------------
SARIMA (Seasonal AutoRegressive Integrated Moving Average) forecasting
using statsmodels. With only 18 training months (< 2 full seasonal cycles),
heavy seasonal differencing is numerically unstable. We use:
  - SARIMA(1,1,1)(1,0,0,12) — one seasonal AR term, no seasonal differencing
    (captures the annual pattern via the lag-12 AR term)
  - Falls back to Holt-Winters Exponential Smoothing per SKU if SARIMA fails

Fixed origin: train on the first 18 months and forecast all six holdout months at once.

Outputs: reports/sarima_predictions.csv, reports/sarima_sku_metrics.csv
"""

import pandas as pd
import numpy as np
import os
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "demand_panel.csv")
REPORTS_PATH = os.path.join(os.path.dirname(__file__), "..", "reports")
HOLDOUT_MONTHS = 6

# SARIMA order: (p,d,q)(P,D,Q,s)
# D=0: avoid seasonal over-differencing with short series
ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 0, 0, 12)


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values(["sku_id", "date"]).reset_index(drop=True)
    return df


def holtwinters_forecast(series_train, n_forecast):
    """Fallback: Holt-Winters additive seasonality."""
    try:
        model = ExponentialSmoothing(
            series_train,
            trend="add",
            seasonal="add",
            seasonal_periods=12,
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True)
        return np.maximum(0, fit.forecast(n_forecast).values)
    except Exception:
        # Last-resort: repeat last value
        return np.full(n_forecast, float(series_train.iloc[-1]))


def sarima_forecast_sku(series_train, n_forecast):
    """Fit SARIMA on training series and produce n_forecast steps ahead."""
    try:
        model = SARIMAX(
            series_train,
            order=ORDER,
            seasonal_order=SEASONAL_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False, maxiter=300, method="lbfgs")
        forecast = fit.forecast(steps=n_forecast)
        preds = np.maximum(0, forecast.values)
        # Sanity check: if any prediction is > 10x the training mean, fallback
        train_mean = float(series_train.mean())
        if train_mean > 0 and np.any(preds > 10 * train_mean):
            raise ValueError("Explosive forecast detected")
        return preds
    except Exception as e:
        # Fallback to Holt-Winters
        return holtwinters_forecast(series_train, n_forecast)


def main():
    df = load_data()
    n_total = df.groupby("sku_id")["date"].count().iloc[0]  # 24
    n_train = n_total - HOLDOUT_MONTHS  # 18

    results = []
    skus = df["sku_id"].unique()
    print(f"Fitting SARIMA for {len(skus)} SKUs (train={n_train}, test={HOLDOUT_MONTHS})...")

    sarima_count = 0
    hw_count = 0

    for idx, sku_id in enumerate(skus):
        grp = df[df["sku_id"] == sku_id].sort_values("date").reset_index(drop=True)
        family = grp["family"].iloc[0]

        train = grp.iloc[:n_train]
        test = grp.iloc[n_train:]

        series_train = train.set_index("date")["demand"].astype(float)
        series_train.index = pd.DatetimeIndex(series_train.index).to_period("M").to_timestamp()

        forecasts = sarima_forecast_sku(series_train, HOLDOUT_MONTHS)

        for i, row in enumerate(test.itertuples()):
            results.append(
                {
                    "sku_id": sku_id,
                    "family": family,
                    "date": row.date,
                    "actual": float(row.demand),
                    "forecast": float(forecasts[i]),
                    "model": "sarima",
                }
            )

        if (idx + 1) % 10 == 0:
            print(f"  Completed {idx + 1}/{len(skus)} SKUs")

    print(f"  Completed {len(skus)}/{len(skus)} SKUs")
    pred_df = pd.DataFrame(results)

    # Metrics
    pred_df = pred_df[pred_df["actual"] > 0].copy()
    pred_df["abs_error"] = (pred_df["actual"] - pred_df["forecast"]).abs()
    pred_df["pct_error"] = pred_df["abs_error"] / pred_df["actual"]

    sku_metrics = (
        pred_df.groupby(["sku_id", "family"])
        .apply(
            lambda g: pd.Series(
                {
                    "MAPE": g["pct_error"].mean() * 100,
                    "WAPE": g["abs_error"].sum() / g["actual"].sum() * 100,
                    "RMSE": np.sqrt((g["abs_error"] ** 2).mean()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    family_metrics = (
        pred_df.groupby("family")
        .apply(
            lambda g: pd.Series(
                {
                    "MAPE": g["pct_error"].mean() * 100,
                    "WAPE": g["abs_error"].sum() / g["actual"].sum() * 100,
                    "RMSE": np.sqrt((g["abs_error"] ** 2).mean()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    overall = {
        "MAPE": pred_df["pct_error"].mean() * 100,
        "WAPE": pred_df["abs_error"].sum() / pred_df["actual"].sum() * 100,
        "RMSE": np.sqrt((pred_df["abs_error"] ** 2).mean()),
    }

    os.makedirs(REPORTS_PATH, exist_ok=True)
    pred_df.to_csv(os.path.join(REPORTS_PATH, "sarima_predictions.csv"), index=False)
    sku_metrics.to_csv(os.path.join(REPORTS_PATH, "sarima_sku_metrics.csv"), index=False)
    family_metrics.to_csv(os.path.join(REPORTS_PATH, "sarima_family_metrics.csv"), index=False)

    print("\n=== SARIMA — Overall Metrics ===")
    print(f"  MAPE : {overall['MAPE']:.2f}%")
    print(f"  WAPE : {overall['WAPE']:.2f}%")
    print(f"  RMSE : {overall['RMSE']:.2f}")

    print("\n=== By Family ===")
    print(family_metrics.to_string(index=False))

    print(f"\nPredictions saved to {REPORTS_PATH}/sarima_predictions.csv")
    return overall


if __name__ == "__main__":
    main()
