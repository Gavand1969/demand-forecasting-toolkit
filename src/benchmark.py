"""
benchmark.py
------------
Orchestrates all three forecasting models, collects metrics, estimates
inventory dollar impact, and produces publication-quality charts.

Run:
    python src/benchmark.py

Outputs:
    reports/benchmark_summary.csv
    reports/inventory_impact.csv
    reports/charts/01_model_comparison_mape.png
    reports/charts/02_family_heatmap.png
    reports/charts/03_forecast_vs_actual_sample.png
    reports/charts/04_error_distribution.png
    reports/charts/05_inventory_impact.png
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

# Add src/ to path so we can import sibling modules
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

DATA_PATH = os.path.join(ROOT_DIR, "data", "demand_panel.csv")
REPORTS_PATH = os.path.join(ROOT_DIR, "reports")
CHARTS_PATH = os.path.join(REPORTS_PATH, "charts")

HOLDOUT_MONTHS = 6

# Assumed unit cost per family (USD) for inventory impact estimation
UNIT_COST = {
    "Engine_Parts": 85,
    "Brake_Systems": 55,
    "Electrical": 40,
    "Body_Panels": 95,
}

# Chart style
PALETTE = {
    "naive_seasonal": "#4C72B0",
    "sarima":         "#DD8452",
    "lightgbm":       "#55A868",
    "gradient_boosting": "#55A868",
}
FAMILY_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


# ---------------------------------------------------------------------------
# Step 1: Run all models
# ---------------------------------------------------------------------------

def run_all_models():
    print("=" * 60)
    print("STEP 1: Generating synthetic demand data")
    print("=" * 60)
    import generate_demand
    generate_demand.main()

    print("\n" + "=" * 60)
    print("STEP 2: Naive Seasonal Baseline")
    print("=" * 60)
    import forecast_baseline
    baseline_overall = forecast_baseline.main()

    print("\n" + "=" * 60)
    print("STEP 3: SARIMA")
    print("=" * 60)
    import forecast_sarima
    sarima_overall = forecast_sarima.main()

    print("\n" + "=" * 60)
    print("STEP 4: LightGBM / GradientBoosting")
    print("=" * 60)
    import forecast_ml
    ml_overall, ml_model_name = forecast_ml.main()

    return baseline_overall, sarima_overall, ml_overall, ml_model_name


# ---------------------------------------------------------------------------
# Step 2: Load all predictions
# ---------------------------------------------------------------------------

def load_predictions():
    baseline = pd.read_csv(os.path.join(REPORTS_PATH, "baseline_predictions.csv"), parse_dates=["date"])
    sarima = pd.read_csv(os.path.join(REPORTS_PATH, "sarima_predictions.csv"), parse_dates=["date"])
    ml = pd.read_csv(os.path.join(REPORTS_PATH, "ml_predictions.csv"), parse_dates=["date"])
    return baseline, sarima, ml


def load_metrics(model_tag):
    path = os.path.join(REPORTS_PATH, f"{model_tag}_family_metrics.csv")
    df = pd.read_csv(path)
    df["model"] = model_tag
    return df


# ---------------------------------------------------------------------------
# Step 3: Inventory impact estimation
# ---------------------------------------------------------------------------

def estimate_inventory_impact(pred_df, model_name):
    """
    Stockout: actual > forecast  -> lost sales / emergency procurement
    Excess:   forecast > actual  -> holding cost (overstock)
    Dollar impact = |error| * unit_cost * factor
    """
    df = pred_df.copy()
    df["unit_cost"] = df["family"].map(UNIT_COST)

    # Overstock factor: 20% of unit cost as carrying cost
    df["overstock_cost"] = np.where(
        df["forecast"] > df["actual"],
        (df["forecast"] - df["actual"]) * df["unit_cost"] * 0.20,
        0,
    )
    # Stockout factor: 35% of unit cost as lost margin / premium sourcing
    df["stockout_cost"] = np.where(
        df["actual"] > df["forecast"],
        (df["actual"] - df["forecast"]) * df["unit_cost"] * 0.35,
        0,
    )
    df["total_impact"] = df["overstock_cost"] + df["stockout_cost"]

    summary = df.groupby("family")[["overstock_cost", "stockout_cost", "total_impact"]].sum()
    summary["model"] = model_name
    return summary.reset_index()


# ---------------------------------------------------------------------------
# Step 4: Charts
# ---------------------------------------------------------------------------

def chart1_model_comparison(summary_df, ml_model_name):
    """Bar chart: MAPE and WAPE side-by-side per model."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Model Comparison — Backtest Error Metrics\n(6-Month Walk-Forward Hold-Out)", fontsize=13, fontweight="bold")

    models = summary_df["model"].tolist()
    labels = [m.replace("_", " ").title() for m in models]
    colors = [PALETTE.get(m, "#888888") for m in models]
    x = np.arange(len(models))
    width = 0.55

    for ax, metric, title in zip(axes, ["MAPE", "WAPE"], ["MAPE (%)", "WAPE (%)"]):
        vals = summary_df[metric].values
        bars = ax.bar(x, vals, width=width, color=colors, edgecolor="white", linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f"{val:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )
        ax.set_ylim(0, max(vals) * 1.25)

    plt.tight_layout()
    path = os.path.join(CHARTS_PATH, "01_model_comparison_mape.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def chart2_family_heatmap(baseline_fam, sarima_fam, ml_fam, ml_model_name):
    """Heatmap of MAPE by model and product family."""
    dfs = []
    for df, label in [(baseline_fam, "Naive Seasonal"), (sarima_fam, "SARIMA"), (ml_fam, ml_model_name.replace("_", " ").title())]:
        tmp = df[["family", "MAPE"]].copy()
        tmp["model"] = label
        dfs.append(tmp)

    combined = pd.concat(dfs)
    pivot = combined.pivot(index="model", columns="family", values="MAPE")

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=pivot.values.max() * 1.1)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.replace("_", " ") for c in pivot.columns], rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if val > pivot.values.max() * 0.6 else "black")

    plt.colorbar(im, ax=ax, label="MAPE (%)", shrink=0.8)
    ax.set_title("MAPE by Model and Product Family", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    path = os.path.join(CHARTS_PATH, "02_family_heatmap.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def chart3_forecast_vs_actual(baseline_df, sarima_df, ml_df, ml_model_name):
    """Line chart: actual vs. all three forecasts for a sample SKU."""
    # Pick SKU_001 — known to be Engine_Parts
    target_sku = "SKU_001"
    raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
    actual_all = raw[raw["sku_id"] == target_sku].sort_values("date")

    fig, ax = plt.subplots(figsize=(12, 5))

    # Full actual series
    ax.plot(actual_all["date"], actual_all["demand"],
            color="black", linewidth=2, label="Actual", zorder=5)

    # Shading for test period
    test_start = actual_all["date"].iloc[-HOLDOUT_MONTHS]
    ax.axvspan(test_start, actual_all["date"].iloc[-1],
               alpha=0.08, color="gray", label="Holdout period")

    # Forecasts (test period only)
    for pred_df, model_label, color in [
        (baseline_df, "Naive Seasonal", PALETTE["naive_seasonal"]),
        (sarima_df, "SARIMA", PALETTE["sarima"]),
        (ml_df, ml_model_name.replace("_", " ").title(), PALETTE.get(ml_model_name, "#55A868")),
    ]:
        sku_pred = pred_df[pred_df["sku_id"] == target_sku].sort_values("date")
        ax.plot(sku_pred["date"], sku_pred["forecast"],
                linestyle="--", linewidth=1.8, color=color, label=model_label)

    ax.set_title(f"Forecast vs. Actual — {target_sku} (Engine Parts)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Units Demanded")
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%Y"))
    plt.tight_layout()
    path = os.path.join(CHARTS_PATH, "03_forecast_vs_actual_sample.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def chart4_error_distribution(baseline_df, sarima_df, ml_df, ml_model_name):
    """Box plot of absolute percentage error by model."""
    import matplotlib.patches as mpatches

    all_dfs = []
    for pred_df, label in [
        (baseline_df, "Naive\nSeasonal"),
        (sarima_df, "SARIMA"),
        (ml_df, ml_model_name.replace("_", " ").title()),
    ]:
        tmp = pred_df[pred_df["actual"] > 0].copy()
        tmp["APE"] = (tmp["actual"] - tmp["forecast"]).abs() / tmp["actual"] * 100
        tmp["label"] = label
        all_dfs.append(tmp)

    combined = pd.concat(all_dfs)

    labels_order = ["Naive\nSeasonal", "SARIMA", ml_model_name.replace("_", " ").title()]
    data_by_model = [combined[combined["label"] == l]["APE"].values for l in labels_order]
    colors_list = [PALETTE["naive_seasonal"], PALETTE["sarima"], PALETTE.get(ml_model_name, "#55A868")]

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(
        data_by_model,
        patch_artist=True,
        notch=False,
        widths=0.5,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )
    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels_order, fontsize=10)
    ax.set_ylabel("Absolute Percentage Error (%)")
    ax.set_title("Error Distribution by Model\n(All SKUs, 6-Month Hold-Out)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, None)
    plt.tight_layout()
    path = os.path.join(CHARTS_PATH, "04_error_distribution.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def chart5_inventory_impact(impact_dfs):
    """Stacked bar: overstock vs. stockout dollar impact by model and family."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    model_labels = [d["model"].iloc[0] for d in impact_dfs]
    nice_labels = [l.replace("_", " ").title() for l in model_labels]

    for ax, impact_df, nice_label in zip(axes, impact_dfs, nice_labels):
        families = impact_df["family"].tolist()
        x = np.arange(len(families))
        over = impact_df["overstock_cost"].values / 1000
        stock = impact_df["stockout_cost"].values / 1000

        bars1 = ax.bar(x, stock, label="Stockout Cost", color="#C44E52", alpha=0.85)
        bars2 = ax.bar(x, over, bottom=stock, label="Overstock Cost", color="#4878CF", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([f.replace("_", "\n") for f in families], fontsize=8, rotation=0)
        ax.set_title(nice_label, fontsize=11, fontweight="bold")
        ax.set_ylabel("Estimated Cost ($K)" if ax == axes[0] else "")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:.0f}K"))

        total = impact_df["total_impact"].sum() / 1000
        ax.text(0.5, 0.97, f"Total: ${total:.1f}K", transform=ax.transAxes,
                ha="center", va="top", fontsize=9, fontweight="bold")

    # Shared legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#C44E52", alpha=0.85, label="Stockout Cost"),
        plt.Rectangle((0, 0), 1, 1, color="#4878CF", alpha=0.85, label="Overstock Cost"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Estimated Inventory Dollar Impact by Model and Product Family\n(6-Month Hold-Out, Assumed Unit Costs)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    path = os.path.join(CHARTS_PATH, "05_inventory_impact.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(CHARTS_PATH, exist_ok=True)

    # Run all models
    baseline_overall, sarima_overall, ml_overall, ml_model_name = run_all_models()

    print("\n" + "=" * 60)
    print("STEP 5: Generating benchmark charts")
    print("=" * 60)

    # Load predictions
    baseline_df, sarima_df, ml_df = load_predictions()

    # Load family metrics
    baseline_fam = pd.read_csv(os.path.join(REPORTS_PATH, "baseline_family_metrics.csv"))
    sarima_fam = pd.read_csv(os.path.join(REPORTS_PATH, "sarima_family_metrics.csv"))
    ml_fam = pd.read_csv(os.path.join(REPORTS_PATH, "ml_family_metrics.csv"))

    # Build overall summary table
    summary_df = pd.DataFrame([
        {"model": "naive_seasonal", **baseline_overall},
        {"model": "sarima",         **sarima_overall},
        {"model": ml_model_name,    **ml_overall},
    ])
    summary_df.to_csv(os.path.join(REPORTS_PATH, "benchmark_summary.csv"), index=False)

    print("\n=== BENCHMARK SUMMARY ===")
    print(summary_df.to_string(index=False))

    # Inventory impact
    impact_baseline = estimate_inventory_impact(baseline_df, "naive_seasonal")
    impact_sarima   = estimate_inventory_impact(sarima_df, "sarima")
    impact_ml       = estimate_inventory_impact(ml_df, ml_model_name)
    impact_all = pd.concat([impact_baseline, impact_sarima, impact_ml])
    impact_all.to_csv(os.path.join(REPORTS_PATH, "inventory_impact.csv"), index=False)

    # Charts
    chart1_model_comparison(summary_df, ml_model_name)
    chart2_family_heatmap(baseline_fam, sarima_fam, ml_fam, ml_model_name)
    chart3_forecast_vs_actual(baseline_df, sarima_df, ml_df, ml_model_name)
    chart4_error_distribution(baseline_df, sarima_df, ml_df, ml_model_name)
    chart5_inventory_impact([impact_baseline, impact_sarima, impact_ml])

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"\nCharts: {CHARTS_PATH}/")
    print(f"Summary: {REPORTS_PATH}/benchmark_summary.csv")

    # Print final clean summary for README
    print("\n=== FINAL METRICS TABLE ===")
    for _, row in summary_df.iterrows():
        print(f"  {row['model']:25s}  MAPE={row['MAPE']:.2f}%  WAPE={row['WAPE']:.2f}%  RMSE={row['RMSE']:.2f}")

    return summary_df, ml_model_name


if __name__ == "__main__":
    main()
