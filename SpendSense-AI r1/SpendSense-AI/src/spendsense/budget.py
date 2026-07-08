"""Rule + ML hybrid budget recommendation engine."""
from __future__ import annotations

import pandas as pd

from .forecast import forecast, monthly_by_category


def recommend(df: pd.DataFrame, savings_target_pct: float = 0.15) -> pd.DataFrame:
    """Return per-category recommended monthly cap and projected savings."""
    monthly = monthly_by_category(df)
    if monthly.empty:
        return pd.DataFrame(
            columns=["category", "median_spend", "p75_spend", "recommended_cap",
                     "next_month_forecast", "projected_savings", "note"]
        )

    stats = (
        monthly.groupby("category")["amount"]
        .agg(median="median", p75=lambda s: s.quantile(0.75), mean="mean")
        .reset_index()
    )
    fc = forecast(df, horizon=1).forecast.rename(columns={"amount": "next_month_forecast"})
    fc = fc[["category", "next_month_forecast"]]
    out = stats.merge(fc, on="category", how="left")

    # Recommended cap: tighter of (median * 1.1) or p75, with a minimum 10% trim of forecast
    out["recommended_cap"] = out[["median", "p75"]].apply(
        lambda r: round(min(r["median"] * 1.1, r["p75"]), 2), axis=1
    )
    forecast_cap = out["next_month_forecast"].fillna(out["mean"]) * (1 - savings_target_pct)
    out["recommended_cap"] = out[["recommended_cap"]].join(forecast_cap.rename("alt")).min(axis=1).round(2)
    out["projected_savings"] = (out["next_month_forecast"].fillna(out["mean"]) - out["recommended_cap"]).clip(lower=0).round(2)

    def _note(row: pd.Series) -> str:
        if pd.isna(row["next_month_forecast"]):
            return "Insufficient history; cap based on past spending."
        if row["next_month_forecast"] > row["p75"]:
            return "Forecast above your 75th percentile — strong saving opportunity."
        if row["projected_savings"] < 50:
            return "Already on track."
        return "Modest trimming recommended."

    out["note"] = out.apply(_note, axis=1)
    out = out.rename(columns={"median": "median_spend", "p75": "p75_spend"}).drop(columns=["mean"])
    return out.sort_values("projected_savings", ascending=False).reset_index(drop=True)
