# SpendSense — Credit Card + UPI Spending ML

End-to-end ML project that unifies **credit card** and **UPI** transactions to deliver:

1. **Spending categorization** — TF-IDF + Logistic Regression on merchant / UPI VPA text.
2. **Fraud / anomaly detection** — Isolation Forest on transaction features.
3. **Spend forecasting** — Per-category monthly forecast (Holt-Winters / linear fallback).
4. **Budget recommendations** — Rule + ML hybrid that suggests savings per category.

Plus a **Streamlit dashboard** to explore everything interactively.

---

## Quickstart

```bash
# 1. Create env & install
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Generate synthetic data (credit card + UPI, 18 months, ~8k tx)
python -m spendsense.data_gen --out data/transactions.csv

# 3. Train all models
python -m spendsense.train --data data/transactions.csv --out models/

# 4. Launch dashboard
streamlit run app/dashboard.py
```

Open http://localhost:8501

---

## Project structure

```
spendsense/
├── app/dashboard.py            # Streamlit UI
├── src/spendsense/
│   ├── data_gen.py             # Synthetic CC + UPI transaction generator
│   ├── preprocess.py           # Unified schema + feature engineering
│   ├── categorize.py           # TF-IDF + LogReg classifier
│   ├── anomaly.py              # Isolation Forest fraud detector
│   ├── forecast.py             # Per-category monthly forecasting
│   ├── budget.py               # Budget recommendation engine
│   ├── train.py                # Trains & saves all models
│   └── predict.py              # Inference helpers
├── notebooks/eda.ipynb         # Exploratory analysis
├── data/                       # Generated CSVs (gitignored)
├── models/                     # Saved .joblib artifacts
└── tests/                      # pytest unit tests
```

---

## Unified transaction schema

| column        | type       | notes                                   |
|---------------|------------|-----------------------------------------|
| txn_id        | str        | unique                                  |
| timestamp     | datetime   | ISO 8601                                |
| source        | enum       | `credit_card` \| `upi`                  |
| amount        | float      | INR                                     |
| merchant      | str        | merchant name (CC) or VPA handle (UPI)  |
| raw_desc      | str        | free-text narration                     |
| city          | str        |                                         |
| category      | str        | label (food, travel, bills, ...)        |
| is_fraud      | int (0/1)  | ground truth for evaluation             |

---

## Models & metrics

- **Categorizer**: TF-IDF (char + word n-grams) → LogReg. Reports macro-F1.
- **Anomaly**: IsolationForest(contamination=0.02). Reports ROC-AUC vs `is_fraud`.
- **Forecast**: ExponentialSmoothing per category, falls back to linear trend if <6 months.
- **Budget**: Suggests cap = min(median × 1.1, p75) per category; flags overspend categories.

---

## License

MIT
