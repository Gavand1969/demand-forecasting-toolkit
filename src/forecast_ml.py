"""
forecast_ml.py
--------------
LightGBM-based demand forecasting with lag and rolling window features.
Falls back to sklearn GradientBoostingRegressor if LightGBM is unavailable.

Features engineered per SKU-month observation:
  - lag_1, lag_2, lag_3       : demand 1-3 months prior
  - lag_12                    : same-month last year
  - rolling_mean_3, rolling_std_3  : 3-month trailing statistics
  - rolling_mean_6            : 6-month trailing mean
  - month (1-12)              : calendar month
  - family_enc                : label-encoded product family

Fit on the first 18 months; forecast each of the final six months using only
demand observed before that month. The fitted parameters remain unchanged.

Outputs: reports/ml_predictions.csv, reports/ml_sku_metrics.csv
"""

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "demand_panel.csv")
REPORTS_PATH = os.path.join(os.path.dirname(__file__), "..", "reports")
HOLDOUT_MONTHS = 6

try:
    import lightgbm as lgb
    MODEL_NAME = "lightgbm"
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    MODEL_NAME = "gradient_boosting"

print(f"ML model: {MODEL_NAME}")


FEATURE_COLS = [
    "lag_1", "lag_2", "lag_3", "lag_12",
    "rolling_mean_3", "rolling_std_3", "rolling_mean_6",
    "month", "family_enc",
]


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values(["sku_id", "date"]).reset_index(drop=True)
    return df


def build_features(df):
    """Build lag/rolling features for all SKUs."""
    df = df.sort_values(["sku_id", "date"]).reset_index(drop=True)
    family_map = {f: i for i, f in enumerate(sorted(df["family"].unique()))}
    df["family_enc"] = df["family"].map(family_map)
    df["month"] = df["date"].dt.month

    # Per-SKU lag and rolling features
    grp = df.groupby("sku_id")["demand"]

    df["lag_1"] = grp.shift(1)
    df["lag_2"] = grp.shift(2)
    df["lag_3"] = grp.shift(3)
    df["lag_12"] = grp.shift(12)
    df["rolling_mean_3"] = grp.shift(1).rolling(3).mean().reset_index(0, drop=True)
    df["rolling_std_3"] = grp.shift(1).rolling(3).std().reset_index(0, drop=True)
    df["rolling_mean_6"] = grp.shift(1).rolling(6).mean().reset_index(0, drop=True)

    return df, family_map


def train_predict(df_feat, n_train, n_total):
    """
    Single global model: train on all SKUs' first n_train months,
    predict the last HOLDOUT_MONTHS months for each SKU.
    """
    # Month index within each SKU (0-based)
    df_feat = df_feat.copy()
    df_feat["month_idx"] = df_feat.groupby("sku_id").cumcount()

    train_mask = df_feat["month_idx"] < n_train
    test_mask = df_feat["month_idx"] >= n_train

    train_df = df_feat[train_mask].dropna(subset=FEATURE_COLS)
    test_df = df_feat[test_mask].dropna(subset=FEATURE_COLS)

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["demand"].values
    X_test = test_df[FEATURE_COLS].values

    if MODEL_NAME == "lightgbm":
        model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
    else:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
        )

    model.fit(X_train, y_train)
    preds = np.maximum(0, model.predict(X_test))

    result_df = test_df[["sku_id", "family", "date", "demand"]].copy()
    result_df = result_df.rename(columns={"demand": "actual"})
    result_df["forecast"] = preds
    result_df["model"] = MODEL_NAME

    return result_df, model


def compute_metrics(pred_df):
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
    n_total = df.groupby("sku_id")["date"].count().iloc[0]
    n_train = n_total - HOLDOUT_MONTHS

    print(f"Building features for {df['sku_id'].nunique()} SKUs...")
    df_feat, family_map = build_features(df)

    print(f"Training {MODEL_NAME} on {n_train} months, forecasting {HOLDOUT_MONTHS}...")
    pred_df, model = train_predict(df_feat, n_train, n_total)

    sku_metrics, family_metrics, overall = compute_metrics(pred_df)

    os.makedirs(REPORTS_PATH, exist_ok=True)
    pred_df.to_csv(os.path.join(REPORTS_PATH, "ml_predictions.csv"), index=False)
    sku_metrics.to_csv(os.path.join(REPORTS_PATH, "ml_sku_metrics.csv"), index=False)
    family_metrics.to_csv(os.path.join(REPORTS_PATH, "ml_family_metrics.csv"), index=False)

    # Feature importance
    if MODEL_NAME == "lightgbm":
        importance = pd.DataFrame(
            {"feature": FEATURE_COLS, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)
        importance.to_csv(os.path.join(REPORTS_PATH, "ml_feature_importance.csv"), index=False)
        print("\n=== Feature Importance (LightGBM) ===")
        print(importance.to_string(index=False))

    print(f"\n=== {MODEL_NAME.upper()} — Overall Metrics ===")
    print(f"  MAPE : {overall['MAPE']:.2f}%")
    print(f"  WAPE : {overall['WAPE']:.2f}%")
    print(f"  RMSE : {overall['RMSE']:.2f}")

    print("\n=== By Family ===")
    print(family_metrics.to_string(index=False))

    print(f"\nPredictions saved to {REPORTS_PATH}/ml_predictions.csv")
    return overall, MODEL_NAME


if __name__ == "__main__":
    main()
