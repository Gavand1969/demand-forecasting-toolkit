"""
forecast_baseline.py
--------------------
Naive Seasonal Baseline: forecast for month t = actual demand at month t-12
(same-period-last-year). For the walk-forward backtest, the last 6 months of
the panel are held out and predictions come from the preceding 12 months.

Outputs: reports/baseline_predictions.csv
"""

import pandas as pd
import numpy as np
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "demand_panel.csv")
REPORTS_PATH = os.path.join(os.path.dirname(__file__), "..", "reports")
HOLDOUT_MONTHS = 6


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values(["sku_id", "date"]).reset_index(drop=True)
    return df


def naive_seasonal_forecast(df):
    """
    Walk-forward: for each SKU, predict the last HOLDOUT_MONTHS months
    using the value from 12 months prior.
    """
    results = []

    for sku_id, grp in df.groupby("sku_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        n = len(grp)  # should be 24

        # Training window: first 18 months; test: last 6
        test_start_idx = n - HOLDOUT_MONTHS

        for i in range(test_start_idx, n):
            lag_idx = i - 12
            if lag_idx < 0:
                forecast = np.nan
            else:
                forecast = float(grp.loc[lag_idx, "demand"])

            results.append(
                {
                    "sku_id": sku_id,
                    "family": grp.loc[i, "family"],
                    "date": grp.loc[i, "date"],
                    "actual": float(grp.loc[i, "demand"]),
                    "forecast": forecast,
                    "model": "naive_seasonal",
                }
            )

    return pd.DataFrame(results)


def compute_metrics(pred_df):
    pred_df = pred_df.dropna(subset=["forecast"])
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

    return sku_metrics, family_metrics, overall


def main():
    df = load_data()
    print(f"Loaded {len(df)} rows for {df['sku_id'].nunique()} SKUs")

    pred_df = naive_seasonal_forecast(df)

    sku_metrics, family_metrics, overall = compute_metrics(pred_df)

    os.makedirs(REPORTS_PATH, exist_ok=True)
    pred_df.to_csv(os.path.join(REPORTS_PATH, "baseline_predictions.csv"), index=False)
    sku_metrics.to_csv(os.path.join(REPORTS_PATH, "baseline_sku_metrics.csv"), index=False)
    family_metrics.to_csv(os.path.join(REPORTS_PATH, "baseline_family_metrics.csv"), index=False)

    print("\n=== Naive Seasonal Baseline — Overall Metrics ===")
    print(f"  MAPE : {overall['MAPE']:.2f}%")
    print(f"  WAPE : {overall['WAPE']:.2f}%")
    print(f"  RMSE : {overall['RMSE']:.2f}")

    print("\n=== By Family ===")
    print(family_metrics.to_string(index=False))

    print(f"\nPredictions saved to {REPORTS_PATH}/baseline_predictions.csv")
    return overall


if __name__ == "__main__":
    main()
