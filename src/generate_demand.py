"""
generate_demand.py
------------------
Generates a synthetic 24-month SKU-level demand panel with:
  - 50 SKUs across 4 product families
  - Linear trend component per SKU
  - Annual seasonality (sine wave)
  - Promotional lift events (randomly placed)
  - Gaussian noise

Output: data/demand_panel.csv
"""

import numpy as np
import pandas as pd
import os

RNG_SEED = 42
N_SKUS = 50
N_MONTHS = 24
FAMILIES = ["Engine_Parts", "Brake_Systems", "Electrical", "Body_Panels"]

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "demand_panel.csv")


def generate_sku_demand(sku_id, family, rng):
    """Generate monthly demand for a single SKU over 24 months."""
    dates = pd.date_range(start="2023-01-01", periods=N_MONTHS, freq="MS")

    # Base demand per family
    base_map = {
        "Engine_Parts": 300,
        "Brake_Systems": 220,
        "Electrical": 180,
        "Body_Panels": 140,
    }
    base = base_map[family] + rng.uniform(-50, 50)

    # Trend: slight upward or downward
    trend_slope = rng.uniform(-2.0, 4.0)  # units/month
    trend = np.arange(N_MONTHS) * trend_slope

    # Seasonality: annual cycle (peak in summer/winter depending on family)
    phase_map = {
        "Engine_Parts": 0.0,
        "Brake_Systems": np.pi / 2,
        "Electrical": np.pi,
        "Body_Panels": 3 * np.pi / 2,
    }
    season_amp = rng.uniform(20, 60)
    t = np.arange(N_MONTHS)
    seasonality = season_amp * np.sin(2 * np.pi * t / 12 + phase_map[family])

    # Promotional lifts: 1-3 random months get a 15-40% uplift
    promo = np.zeros(N_MONTHS)
    n_promos = rng.integers(1, 4)
    promo_months = rng.choice(N_MONTHS, size=n_promos, replace=False)
    for m in promo_months:
        promo[m] = base * rng.uniform(0.15, 0.40)

    # Noise
    noise = rng.normal(0, base * 0.07, size=N_MONTHS)

    demand = np.maximum(0, base + trend + seasonality + promo + noise)
    demand = np.round(demand).astype(int)

    rows = []
    for i, (date, d) in enumerate(zip(dates, demand)):
        rows.append(
            {
                "date": date,
                "sku_id": sku_id,
                "family": family,
                "demand": d,
                "is_promo": int(i in promo_months),
            }
        )
    return rows


def main():
    rng = np.random.default_rng(RNG_SEED)
    # Distribute 50 SKUs across 4 families: 13, 13, 12, 12
    family_counts = [13, 13, 12, 12]

    all_rows = []
    sku_counter = 1
    for family, count in zip(FAMILIES, family_counts):
        for _ in range(count):
            sku_id = f"SKU_{sku_counter:03d}"
            rows = generate_sku_demand(sku_id, family, rng)
            all_rows.extend(rows)
            sku_counter += 1

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["sku_id", "date"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Generated demand panel: {df.shape[0]} rows, {df['sku_id'].nunique()} SKUs")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Families: {df['family'].unique().tolist()}")
    print(f"Saved to: {os.path.abspath(OUTPUT_PATH)}")
    print(df.groupby("family")["demand"].agg(["mean", "std"]).round(1))


if __name__ == "__main__":
    main()
