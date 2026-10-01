"""Reproducibility and shape checks on the synthetic demand panel."""
import numpy as np
import pandas as pd
import generate_demand


def test_reproducible_seed(tmp_path, monkeypatch):
    out1 = tmp_path / "panel1.csv"
    monkeypatch.setattr(generate_demand, "OUTPUT_PATH", str(out1))
    generate_demand.main()
    df1 = pd.read_csv(out1)

    out2 = tmp_path / "panel2.csv"
    monkeypatch.setattr(generate_demand, "OUTPUT_PATH", str(out2))
    generate_demand.main()
    df2 = pd.read_csv(out2)

    pd.testing.assert_frame_equal(df1, df2)


def test_panel_shape(tmp_path, monkeypatch):
    out = tmp_path / "panel.csv"
    monkeypatch.setattr(generate_demand, "OUTPUT_PATH", str(out))
    generate_demand.main()
    df = pd.read_csv(out, parse_dates=["date"])

    assert df["sku_id"].nunique() == 50
    assert df.groupby("sku_id").size().eq(24).all()
    assert set(df["family"].unique()) == {
        "Engine_Parts", "Brake_Systems", "Electrical", "Body_Panels",
    }
    assert (df["demand"] >= 0).all()
