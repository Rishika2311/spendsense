"""Smoke tests for the SpendSense pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spendsense.anomaly import train_anomaly  # noqa: E402
from spendsense.budget import recommend  # noqa: E402
from spendsense.categorize import predict, train_categorizer  # noqa: E402
from spendsense.data_gen import GenConfig, generate  # noqa: E402
from spendsense.forecast import forecast  # noqa: E402
from spendsense.preprocess import clean  # noqa: E402


def _small_df():
    return clean(generate(GenConfig(months=6, txns_per_month=120, fraud_rate=0.02)))


def test_data_shape():
    df = _small_df()
    assert len(df) > 500
    assert {"timestamp", "source", "amount", "merchant", "category"}.issubset(df.columns)


def test_categorizer_learns():
    df = _small_df()
    res = train_categorizer(df)
    assert res.macro_f1 > 0.7
    preds = predict(res.pipeline, df.head(10))
    assert len(preds) == 10


def test_anomaly_runs():
    df = _small_df()
    am, auc = train_anomaly(df)
    scores = am.score(df.head(50))
    assert len(scores) == 50
    if auc is not None:
        assert 0.0 <= auc <= 1.0


def test_forecast_and_budget():
    df = _small_df()
    fc = forecast(df, horizon=2)
    assert not fc.forecast.empty
    rec = recommend(df)
    assert "recommended_cap" in rec.columns
