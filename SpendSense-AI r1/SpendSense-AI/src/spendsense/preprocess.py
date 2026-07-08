"""Unified schema + feature engineering for CC and UPI transactions."""
from __future__ import annotations

import pandas as pd

REQUIRED_COLS = ["timestamp", "source", "amount", "merchant", "raw_desc"]


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return clean(df)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "amount"])
    df["amount"] = df["amount"].astype(float)
    df["source"] = df["source"].str.lower().str.strip()
    df["merchant"] = df["merchant"].fillna("").astype(str)
    df["raw_desc"] = df["raw_desc"].fillna("").astype(str)
    return df.sort_values("timestamp").reset_index(drop=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.dayofweek
    df["day"] = ts.dt.day
    df["month"] = ts.dt.to_period("M").astype(str)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] > 22)).astype(int)
    return df


def text_feature(df: pd.DataFrame) -> pd.Series:
    """Combined text used by the categorizer."""
    return (df["merchant"].fillna("") + " " + df["raw_desc"].fillna("")).str.lower()
