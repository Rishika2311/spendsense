"""Synthetic Credit Card + UPI transaction generator.

Produces a realistic unified dataset with category labels and a small
fraction of fraudulent / anomalous transactions for supervised evaluation.
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
random.seed(42)

CITIES = ["Mumbai", "Bengaluru", "Delhi", "Hyderabad", "Pune", "Chennai", "Kolkata", "Ahmedabad"]

# category -> (merchant samples, upi vpa samples, amount mean, amount std)
CATEGORIES = {
    "food": (
        ["Swiggy", "Zomato", "Dominos", "McDonalds", "Starbucks", "Cafe Coffee Day", "Haldirams"],
        ["swiggy@ybl", "zomato@paytm", "dominos@upi", "rest.ownername@okhdfc"],
        (320, 180),
    ),
    "groceries": (
        ["BigBasket", "Blinkit", "Zepto", "DMart", "Reliance Fresh", "More Supermarket"],
        ["bigbasket@ybl", "blinkit@paytm", "dmart@hdfc", "kirana.shop@okicici"],
        (850, 400),
    ),
    "travel": (
        ["IRCTC", "MakeMyTrip", "Uber", "Ola Cabs", "IndiGo", "Vistara", "RedBus"],
        ["uber.india@axisbank", "olacabs@paytm", "irctc@sbi", "makemytrip@hdfc"],
        (1800, 1500),
    ),
    "bills": (
        ["Airtel", "Jio", "Vi Postpaid", "Tata Power", "BESCOM", "Adani Electricity"],
        ["airtel@axis", "jio@paytm", "tatapower@hdfc", "bescom@upi"],
        (1200, 600),
    ),
    "shopping": (
        ["Amazon", "Flipkart", "Myntra", "Ajio", "Nykaa", "Decathlon", "Croma"],
        ["amazon@apl", "flipkart@ybl", "myntra@paytm", "nykaa@hdfc"],
        (2200, 1800),
    ),
    "entertainment": (
        ["Netflix", "Spotify", "BookMyShow", "PVR Cinemas", "Hotstar", "Prime Video"],
        ["bookmyshow@paytm", "netflix@hdfc", "spotify@ybl"],
        (450, 250),
    ),
    "health": (
        ["Apollo Pharmacy", "1mg", "PharmEasy", "Practo", "Cult.fit", "Healthians"],
        ["apollo@hdfc", "1mg@paytm", "pharmeasy@ybl", "cultfit@axis"],
        (700, 500),
    ),
    "transfers": (
        ["NEFT Transfer", "Self Transfer", "Friend Payment"],
        ["rahul.sharma@okaxis", "priya.k@okhdfc", "amit99@paytm", "self@ybl", "neha.r@oksbi"],
        (1500, 1200),
    ),
}

CATEGORY_WEIGHTS = {
    "food": 0.20,
    "groceries": 0.13,
    "travel": 0.10,
    "bills": 0.08,
    "shopping": 0.14,
    "entertainment": 0.08,
    "health": 0.07,
    "transfers": 0.20,
}


@dataclass
class GenConfig:
    months: int = 18
    txns_per_month: int = 450
    fraud_rate: float = 0.015
    upi_share: float = 0.55


def _sample_amount(mean: float, std: float) -> float:
    val = max(20.0, RNG.normal(mean, std))
    return round(float(val), 2)


def _make_row(ts: datetime, cfg: GenConfig) -> dict:
    cat = RNG.choice(list(CATEGORY_WEIGHTS), p=list(CATEGORY_WEIGHTS.values()))
    cc_merchants, upi_vpas, (mean, std) = CATEGORIES[cat]
    is_upi = RNG.random() < cfg.upi_share
    amount = _sample_amount(mean, std)
    if is_upi:
        merchant = RNG.choice(upi_vpas)
        raw = f"UPI/{merchant}/{RNG.integers(100000, 999999)}/payment"
        source = "upi"
    else:
        merchant = RNG.choice(cc_merchants)
        raw = f"POS {merchant.upper()} {RNG.choice(CITIES).upper()}"
        source = "credit_card"

    is_fraud = 0
    if RNG.random() < cfg.fraud_rate:
        is_fraud = 1
        amount = round(amount * RNG.uniform(8, 25), 2)  # unusually large
        ts = ts.replace(hour=int(RNG.integers(1, 5)))   # odd hour

    return {
        "txn_id": f"TX{RNG.integers(10**9, 10**10)}",
        "timestamp": ts.isoformat(),
        "source": source,
        "amount": amount,
        "merchant": merchant,
        "raw_desc": raw,
        "city": RNG.choice(CITIES),
        "category": cat,
        "is_fraud": is_fraud,
    }


def generate(cfg: GenConfig | None = None) -> pd.DataFrame:
    cfg = cfg or GenConfig()
    end = datetime.now().replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=30 * cfg.months)
    rows = []
    total = cfg.months * cfg.txns_per_month
    span = (end - start).total_seconds()
    for _ in range(total):
        offset = RNG.uniform(0, span)
        ts = start + timedelta(seconds=offset)
        # bias hours toward daytime
        ts = ts.replace(hour=int(np.clip(RNG.normal(14, 4), 0, 23)))
        rows.append(_make_row(ts, cfg))
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/transactions.csv")
    p.add_argument("--months", type=int, default=18)
    p.add_argument("--per-month", type=int, default=450)
    args = p.parse_args()

    cfg = GenConfig(months=args.months, txns_per_month=args.per_month)
    df = generate(cfg)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} transactions -> {args.out}")
    print(df.head())


if __name__ == "__main__":
    main()
