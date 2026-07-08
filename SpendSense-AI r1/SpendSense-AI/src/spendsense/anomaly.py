"""Fraud / anomaly detection with Isolation Forest."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .preprocess import add_time_features

FEATURES = ["amount", "hour", "dow", "is_weekend", "is_night"]


@dataclass
class AnomalyModel:
    scaler: StandardScaler
    model: IsolationForest

    def score(self, df: pd.DataFrame) -> np.ndarray:
        X = _features(df)
        Xs = self.scaler.transform(X)
        # Higher = more anomalous
        return -self.model.score_samples(Xs)


def _features(df: pd.DataFrame) -> pd.DataFrame:
    feat = add_time_features(df)
    # log-amount tames the heavy tail
    feat = feat.copy()
    feat["amount"] = np.log1p(feat["amount"])
    return feat[FEATURES]


def train_anomaly(df: pd.DataFrame, contamination: float = 0.02) -> tuple[AnomalyModel, float | None]:
    X = _features(df)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    ).fit(Xs)
    am = AnomalyModel(scaler, model)
    auc = None
    if "is_fraud" in df.columns and df["is_fraud"].nunique() > 1:
        scores = am.score(df)
        auc = float(roc_auc_score(df["is_fraud"], scores))
    return am, auc


def save(am: AnomalyModel, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(am, path)


def load(path: str | Path) -> AnomalyModel:
    return joblib.load(path)
