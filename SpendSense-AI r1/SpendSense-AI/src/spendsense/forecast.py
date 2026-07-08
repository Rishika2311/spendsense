"""Per-category monthly spend forecasting."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_SM = True
except Exception:  # pragma: no cover
    HAS_SM = False


@dataclass
class ForecastResult:
    history: pd.DataFrame   # columns: month, category, amount
    forecast: pd.DataFrame  # columns: month, category, amount, kind='forecast'


def monthly_by_category(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["month"] = pd.to_datetime(d["timestamp"]).dt.to_period("M").dt.to_timestamp()
    g = d.groupby(["month", "category"], as_index=False)["amount"].sum()
    return g


def _forecast_series(s: pd.Series, horizon: int) -> np.ndarray:
    s = s.astype(float)
    if len(s) >= 6 and HAS_SM:
        try:
            model = ExponentialSmoothing(s, trend="add", seasonal=None).fit(optimized=True)
            return np.clip(model.forecast(horizon).values, 0, None)
        except Exception:
            pass
    # fallback: linear trend, else mean
    if len(s) >= 2:
        x = np.arange(len(s))
        coef = np.polyfit(x, s.values, 1)
        future_x = np.arange(len(s), len(s) + horizon)
        return np.clip(np.polyval(coef, future_x), 0, None)
    return np.repeat(float(s.mean() if len(s) else 0.0), horizon)


def forecast(df: pd.DataFrame, horizon: int = 3) -> ForecastResult:
    hist = monthly_by_category(df)
    pieces = []
    for cat, g in hist.groupby("category"):
        g = g.sort_values("month")
        preds = _forecast_series(g["amount"], horizon)
        last = g["month"].max()
        future_months = pd.date_range(
            last + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"
        )
        pieces.append(
            pd.DataFrame({"month": future_months, "category": cat, "amount": preds})
        )
    fc = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(
        columns=["month", "category", "amount"]
    )
    return ForecastResult(history=hist, forecast=fc)
