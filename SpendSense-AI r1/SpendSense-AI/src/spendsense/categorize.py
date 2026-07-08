"""Spending categorization: TF-IDF + Logistic Regression."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .preprocess import text_feature


@dataclass
class CategorizerResult:
    pipeline: Pipeline
    macro_f1: float
    report: str


def train_categorizer(df: pd.DataFrame, label_col: str = "category") -> CategorizerResult:
    X = text_feature(df)
    y = df[label_col].astype(str)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    analyzer="word",
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            ("clf", LogisticRegression(max_iter=2000, n_jobs=None, C=2.0)),
        ]
    )
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_te)
    f1 = f1_score(y_te, preds, average="macro")
    return CategorizerResult(pipe, f1, classification_report(y_te, preds))


def save(pipe: Pipeline, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)


def load(path: str | Path) -> Pipeline:
    return joblib.load(path)


def predict(pipe: Pipeline, df: pd.DataFrame) -> pd.Series:
    return pd.Series(pipe.predict(text_feature(df)), index=df.index, name="predicted_category")
