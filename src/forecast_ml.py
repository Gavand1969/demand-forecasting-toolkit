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

Walk-forward protocol:
  - Train on first 18 months
  - Forecast last 6 months *recursively* — at horizon h>1, the lag/rolling
    features that fall inside the test window use the model's own prior
    forecasts, not the held-out actuals. This avoids data leakage and
    matches how the model would be used in production (where future
    actuals are unknown).

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


def _compute_features_from_history(history, month_val, family_enc_val):
    """Compute one row of features from a 1-D array of past demand values.

    `history` must contain only observations strictly prior to the target
    month. For training rows this is real demand; for test rows beyond the
    first horizon step it is a mix of real (pre-test) demand and the model's
    own prior forecasts.
    """
    h = np.asarray(history, dtype=float)
    n = len(h)

    def lag(k):
        return h[-k] if n >= k else np.nan

    def roll_mean(k):
        return h[-k:].mean() if n >= k else np.nan

    def roll_std(k):
        return h[-k:].std(ddof=1) if n >= k else np.nan

    return {
        "lag_1": lag(1),
        "lag_2": lag(2),
        "lag_3": lag(3),
        "lag_12": lag(12),
        "rolling_mean_3": roll_mean(3),
        "rolling_std_3": roll_std(3),
        "rolling_mean_6": roll_mean(6),
        "month": month_val,
        "family_enc": family_enc_val,
    }


def build_training_features(df):
    """Build lag/rolling features for the full panel — used to assemble the
    training set only. Test-period rows produced here are discarded by the
    caller; test forecasts come from `recursive_forecast`.
    """
    family_map = {f: i for i, f in enumerate(sorted(df["family"].unique()))}
    df = df.copy()
    df["family_enc"] = df["family"].map(family_map)
    df["month"] = df["date"].dt.month

    df = df.sort_values(["sku_id", "date"])
    grp = df.groupby("sku_id")["demand"]

    df["lag_1"] = grp.shift(1)
    df["lag_2"] = grp.shift(2)
    df["lag_3"] = grp.shift(3)
    df["lag_12"] = grp.shift(12)
    df["rolling_mean_3"] = grp.shift(1).rolling(3).mean().reset_index(0, drop=True)
    df["rolling_std_3"] = grp.shift(1).rolling(3).std().reset_index(0, drop=True)
    df["rolling_mean_6"] = grp.shift(1).rolling(6).mean().reset_index(0, drop=True)

    return df, family_map


def fit_model(X_train, y_train):
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
    return model


def recursive_forecast(model, df, family_map, n_train, n_total):
    """For each SKU, produce horizon = n_total - n_train forecasts by
    feeding the model's prior predictions back into the lag/rolling features.

    This is the production-realistic protocol: at forecast time we do not
    know the actual demand for any month in the test window, so each
    horizon step's features must be derived from the train series plus the
    forecasts we've already made for earlier horizon steps.
    """
    horizon = n_total - n_train
    rows = []
    for sku_id, grp in df.groupby("sku_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        family = grp["family"].iloc[0]
        family_enc = family_map[family]
        train_demand = grp["demand"].iloc[:n_train].to_numpy(dtype=float)
        history = list(train_demand)

        for h in range(horizon):
            target_row = grp.iloc[n_train + h]
            feats = _compute_features_from_history(
                history,
                month_val=int(target_row["date"].month),
                family_enc_val=family_enc,
            )
            # If any required feature is NaN (e.g., lag_12 missing in
            # very short series) fall back to the last observed value.
            x = np.array([feats[c] for c in FEATURE_COLS], dtype=float).reshape(1, -1)
            if np.isnan(x).any():
                yhat = float(history[-1])
            else:
                yhat = float(max(0.0, model.predict(x)[0]))
            rows.append({
                "sku_id": sku_id,
                "family": family,
                "date": target_row["date"],
                "actual": float(target_row["demand"]),
                "forecast": yhat,
                "model": MODEL_NAME,
            })
            history.append(yhat)

    return pd.DataFrame(rows)


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
    df_feat, family_map = build_training_features(df)

    # Train only on rows whose target month is inside the training window.
    df_feat["month_idx"] = df_feat.groupby("sku_id").cumcount()
    train_df = df_feat[df_feat["month_idx"] < n_train].dropna(subset=FEATURE_COLS)
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["demand"].values

    print(f"Training {MODEL_NAME} on {len(train_df)} rows ({n_train}-month window), recursively forecasting {HOLDOUT_MONTHS} months...")
    model = fit_model(X_train, y_train)

    pred_df = recursive_forecast(model, df, family_map, n_train, n_total)

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
