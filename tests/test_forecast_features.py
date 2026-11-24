import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from forecast_ml import FEATURE_COLS, build_features
from forecast_sarima import holtwinters_forecast


class ForecastFeatureTests(unittest.TestCase):
    def test_shuffling_rows_preserves_per_sku_lags_and_rolling_values(self):
        dates = pd.date_range("2023-01-01", periods=15, freq="MS")
        panel = pd.DataFrame(
            [(sku, "parts", date, offset + i) for sku, offset in [("A", 0), ("B", 100)]
             for i, date in enumerate(dates)],
            columns=["sku_id", "family", "date", "demand"],
        )
        shuffled = panel.sample(frac=1, random_state=42).reset_index(drop=True)
        expected, _ = build_features(panel)
        result, _ = build_features(shuffled)
        keys = ["sku_id", "date"]
        ref = expected.set_index(keys).sort_index()[FEATURE_COLS]
        got = result.set_index(keys).sort_index()[FEATURE_COLS]
        pd.testing.assert_frame_equal(got, ref)
        self.assertTrue(pd.isna(got.loc[("B", dates[0]), "rolling_mean_3"]))
        self.assertEqual(got.loc[("B", dates[3]), "rolling_mean_3"], 101)
        self.assertEqual(got.loc[("B", dates[6]), "rolling_mean_6"], 102.5)

    def test_seasonal_fallback_uses_history_when_two_cycles_are_available(self):
        months = pd.date_range("2023-01-01", periods=36, freq="MS")
        series = pd.Series([100 + 3 * i + i % 12 for i in range(36)], index=months)
        forecast = holtwinters_forecast(series, 2)
        self.assertAlmostEqual(forecast[0], 208, delta=0.5)
        self.assertAlmostEqual(forecast[1], 212, delta=0.5)


if __name__ == "__main__":
    unittest.main()
