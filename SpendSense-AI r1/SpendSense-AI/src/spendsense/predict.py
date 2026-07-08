"""Convenience inference helpers used by the dashboard."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import anomaly, categorize
from .preprocess import clean


def enrich(df: pd.DataFrame, models_dir: str | Path, anomaly_threshold: float = 0.98) -> pd.DataFrame:
    models_dir = Path(models_dir)
    df = clean(df)
    cat_pipe = categorize.load(models_dir / "categorizer.joblib")
    am = anomaly.load(models_dir / "anomaly.joblib")

    df["predicted_category"] = categorize.predict(cat_pipe, df)
    df["anomaly_score"] = am.score(df)
    df["anomaly_flag"] = (df["anomaly_score"] > df["anomaly_score"].quantile(anomaly_threshold)).astype(int)
    return df
