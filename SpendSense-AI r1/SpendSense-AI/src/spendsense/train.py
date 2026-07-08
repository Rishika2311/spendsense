"""Train all models and persist them to disk."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import anomaly, categorize
from .preprocess import load


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="models")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load(args.data)
    print(f"Loaded {len(df):,} transactions")

    # 1) Categorizer
    cat_res = categorize.train_categorizer(df)
    categorize.save(cat_res.pipeline, out / "categorizer.joblib")
    print(f"Categorizer macro-F1: {cat_res.macro_f1:.3f}")

    # 2) Anomaly
    am, auc = anomaly.train_anomaly(df)
    anomaly.save(am, out / "anomaly.joblib")
    print(f"Anomaly ROC-AUC vs is_fraud: {auc:.3f}" if auc else "Anomaly trained (no labels)")

    metrics = {
        "categorizer_macro_f1": cat_res.macro_f1,
        "anomaly_roc_auc": auc,
        "n_rows": len(df),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "classification_report.txt").write_text(cat_res.report)
    print(f"Saved models -> {out.resolve()}")


if __name__ == "__main__":
    main()
