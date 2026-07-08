"""Streamlit dashboard for SpendSense."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from spendsense.budget import recommend  # noqa: E402
from spendsense.forecast import forecast  # noqa: E402
from spendsense.predict import enrich  # noqa: E402
from spendsense.preprocess import clean  # noqa: E402

st.set_page_config(page_title="SpendSense", page_icon="💳", layout="wide")
st.title("💳 SpendSense — Credit Card + UPI Spending Intelligence")

DATA_DEFAULT = ROOT / "data" / "transactions.csv"
MODELS_DIR = ROOT / "models"

# ── Sidebar: data + anomaly ───────────────────────────────────────────────────
with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload transactions CSV", type=["csv"])
    use_default = st.checkbox("Use generated sample", value=True, disabled=uploaded is not None)
    st.caption("Required columns: timestamp, source, amount, merchant, raw_desc")

    st.divider()
    st.header("Anomaly Sensitivity")
    anomaly_threshold = st.slider(
        "Flag top N% as anomalies",
        min_value=90, max_value=99, value=98, step=1,
        format="%d%%",
        help="Lower = more transactions flagged; higher = only extreme outliers",
    ) / 100

# ── Load & enrich ─────────────────────────────────────────────────────────────
if uploaded is not None:
    df_raw = pd.read_csv(uploaded)
elif use_default and DATA_DEFAULT.exists():
    df_raw = pd.read_csv(DATA_DEFAULT)
else:
    st.warning("Upload a CSV or run `python -m spendsense.data_gen` first.")
    st.stop()

df_full = clean(df_raw)

if not (MODELS_DIR / "categorizer.joblib").exists():
    st.error("Models not found. Run `python -m spendsense.train --data data/transactions.csv --out models/`.")
    st.stop()

df_full = enrich(df_full, MODELS_DIR, anomaly_threshold=anomaly_threshold)

# ── Sidebar: filters (needs df_full to know available values) ─────────────────
with st.sidebar:
    st.divider()
    st.header("Filters")

    min_date = df_full["timestamp"].dt.date.min()
    max_date = df_full["timestamp"].dt.date.max()
    date_range = st.date_input("Date range", value=(min_date, max_date),
                               min_value=min_date, max_value=max_date)

    all_sources = sorted(df_full["source"].unique().tolist())
    sel_sources = st.multiselect("Payment source", options=all_sources, default=all_sources)

    all_cats = sorted(df_full["predicted_category"].unique().tolist())
    sel_cats = st.multiselect("Categories", options=all_cats, default=all_cats)

# ── Apply filters ─────────────────────────────────────────────────────────────
df = df_full.copy()
if len(date_range) == 2:
    df = df[(df["timestamp"].dt.date >= date_range[0]) & (df["timestamp"].dt.date <= date_range[1])]
if sel_sources:
    df = df[df["source"].isin(sel_sources)]
if sel_cats:
    df = df[df["predicted_category"].isin(sel_cats)]

if len(df) == 0:
    st.warning("No data matches the current filters — adjust the sidebar filters.")
    st.stop()

# ── Month-over-month helpers ──────────────────────────────────────────────────
latest_period = df["timestamp"].dt.to_period("M").max()
prev_period = latest_period - 1
this_month = df[df["timestamp"].dt.to_period("M") == latest_period]
last_month = df[df["timestamp"].dt.to_period("M") == prev_period]
has_prev = len(last_month) > 0


def _fmt_delta(val: float, prefix: str = "") -> str:
    sign = "+" if val >= 0 else ""
    return f"{prefix}{sign}{val:,.0f} vs last month"


# ── Smart Insights ────────────────────────────────────────────────────────────
with st.expander("Smart Insights", expanded=True):
    insights: list[tuple[str, str]] = []  # (icon, text)

    if len(this_month) > 0:
        cat_totals = this_month.groupby("predicted_category")["amount"].sum()
        top_cat = cat_totals.idxmax()
        insights.append(("🏆", f"Top category this month: **{top_cat}** (₹{cat_totals.max():,.0f})"))

    if has_prev and last_month["amount"].sum() > 0:
        pct = (this_month["amount"].sum() - last_month["amount"].sum()) / last_month["amount"].sum() * 100
        arrow = "📈" if pct > 0 else "📉"
        insights.append((arrow, f"Total spend is **{'up' if pct > 0 else 'down'} {abs(pct):.0f}%** vs last month"))

    weekend = df[df["timestamp"].dt.dayofweek >= 5]["amount"]
    weekday = df[df["timestamp"].dt.dayofweek < 5]["amount"]
    if len(weekend) > 0 and len(weekday) > 0:
        we_avg, wd_avg = weekend.mean(), weekday.mean()
        if we_avg > wd_avg * 1.15:
            insights.append(("🛍️", f"Weekend avg ₹{we_avg:.0f} vs weekday ₹{wd_avg:.0f} — you spend **{(we_avg/wd_avg - 1)*100:.0f}% more** on weekends"))
        else:
            insights.append(("✅", f"Spending is consistent — weekday avg ₹{wd_avg:.0f}, weekend avg ₹{we_avg:.0f}"))

    top_merch = df.groupby("merchant")["amount"].sum()
    insights.append(("🏪", f"Highest-spend merchant: **{top_merch.idxmax()}** (₹{top_merch.max():,.0f} total)"))

    n_anom = int(df["anomaly_flag"].sum())
    if n_anom > 0:
        insights.append(("⚠️", f"**{n_anom} unusual transactions** flagged — check the Anomalies tab"))

    upi_amt = df[df["source"] == "upi"]["amount"].sum()
    card_amt = df[df["source"] == "credit_card"]["amount"].sum()
    total_amt = upi_amt + card_amt
    if total_amt > 0:
        insights.append(("💳", f"UPI: ₹{upi_amt:,.0f} ({upi_amt/total_amt*100:.0f}%) · Card: ₹{card_amt:,.0f} ({card_amt/total_amt*100:.0f}%)"))

    peak_hour = df.groupby(df["timestamp"].dt.hour)["amount"].sum().idxmax()
    insights.append(("🕐", f"Peak spending hour: **{peak_hour}:00–{peak_hour+1}:00**"))

    cols = st.columns(2)
    for i, (icon, text) in enumerate(insights):
        cols[i % 2].info(f"{icon} {text}")

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions", f"{len(df):,}",
          delta=_fmt_delta(len(this_month) - len(last_month)) if has_prev else None)
c2.metric("Total spend (₹)", f"{df['amount'].sum():,.0f}",
          delta=_fmt_delta(this_month["amount"].sum() - last_month["amount"].sum(), "₹") if has_prev else None)
c3.metric("UPI share", f"{df['source'].eq('upi').mean() * 100:.0f}%")
c4.metric("Flagged anomalies", int(df["anomaly_flag"].sum()),
          delta=_fmt_delta(int(this_month["anomaly_flag"].sum()) - int(last_month["anomaly_flag"].sum()))
          if has_prev else None)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    ["Overview", "Categories", "Anomalies", "Forecast", "Budget", "Patterns", "Merchants", "Model Info"]
)

# ── Tab 1: Overview ───────────────────────────────────────────────────────────
with tab1:
    st.subheader("Spend over time")
    daily = df.assign(date=df["timestamp"].dt.date).groupby(["date", "source"])["amount"].sum().reset_index()
    st.plotly_chart(px.area(daily, x="date", y="amount", color="source"), use_container_width=True)

    # Recurring transaction detector
    st.subheader("Potential recurring transactions / subscriptions")
    recurring_rows = []
    for merchant, grp in df.groupby("merchant"):
        if len(grp) < 3:
            continue
        median_amt = grp["amount"].median()
        if median_amt == 0:
            continue
        similar = grp[abs(grp["amount"] - median_amt) / median_amt < 0.15]
        if len(similar) < 3:
            continue
        intervals = similar["timestamp"].sort_values().diff().dropna().dt.days
        if len(intervals) > 0 and intervals.std() < 12:
            avg_interval = intervals.mean()
            est_monthly = median_amt * 30 / max(avg_interval, 1)
            recurring_rows.append({
                "merchant": merchant,
                "avg_amount (₹)": round(median_amt),
                "occurrences": len(similar),
                "avg_interval_days": round(avg_interval),
                "est_monthly (₹)": round(est_monthly),
            })
    if recurring_rows:
        rec_df = pd.DataFrame(recurring_rows).sort_values("est_monthly (₹)", ascending=False)
        st.info(f"Found **{len(rec_df)} recurring charges** — est. ₹{rec_df['est_monthly (₹)'].sum():,.0f}/month total")
        st.dataframe(rec_df, use_container_width=True)
    else:
        st.caption("No strong recurring patterns detected in the current filtered data.")

    st.subheader("Recent transactions")
    st.dataframe(
        df.sort_values("timestamp", ascending=False)
        .head(50)[["timestamp", "source", "amount", "merchant", "predicted_category", "anomaly_flag"]],
        use_container_width=True,
    )
    st.download_button(
        "Download enriched CSV",
        data=df.to_csv(index=False).encode(),
        file_name="spendsense_enriched.csv",
        mime="text/csv",
    )

# ── Tab 2: Categories ─────────────────────────────────────────────────────────
with tab2:
    st.subheader("Spend by predicted category")
    by_cat = df.groupby("predicted_category", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
    st.plotly_chart(
        px.bar(by_cat, x="predicted_category", y="amount",
               color="amount", color_continuous_scale="Purples"),
        use_container_width=True,
    )

    st.subheader("Category × source")
    mix = df.groupby(["predicted_category", "source"], as_index=False)["amount"].sum()
    st.plotly_chart(
        px.bar(mix, x="predicted_category", y="amount", color="source", barmode="group"),
        use_container_width=True,
    )

    if "city" in df.columns and df["city"].notna().any():
        st.subheader("Spend by city (top 15)")
        by_city = (
            df.groupby("city", as_index=False)["amount"].sum()
            .sort_values("amount", ascending=False)
            .head(15)
        )
        st.plotly_chart(
            px.bar(by_city, x="city", y="amount", color="amount", color_continuous_scale="Purples"),
            use_container_width=True,
        )

        st.subheader("Category breakdown by city")
        city_cat = df.groupby(["city", "predicted_category"], as_index=False)["amount"].sum()
        top_cities = by_city["city"].tolist()
        city_cat = city_cat[city_cat["city"].isin(top_cities[:10])]
        st.plotly_chart(
            px.bar(city_cat, x="city", y="amount", color="predicted_category", barmode="stack"),
            use_container_width=True,
        )

# ── Tab 3: Anomalies ──────────────────────────────────────────────────────────
with tab3:
    st.subheader("Anomaly scores")
    st.plotly_chart(
        px.scatter(
            df, x="timestamp", y="amount",
            color=df["anomaly_flag"].map({0: "Normal", 1: "Anomaly"}),
            color_discrete_map={"Normal": "#7C3AED", "Anomaly": "#EF4444"},
            hover_data=["merchant", "source", "predicted_category", "anomaly_score"],
        ),
        use_container_width=True,
    )

    st.subheader("Top suspicious transactions")
    st.dataframe(
        df.sort_values("anomaly_score", ascending=False).head(25)[
            ["timestamp", "source", "amount", "merchant", "predicted_category", "anomaly_score"]
        ],
        use_container_width=True,
    )

# ── Tab 4: Forecast ───────────────────────────────────────────────────────────
with tab4:
    horizon = st.slider("Forecast horizon (months)", 1, 6, 3)
    res = forecast(df.assign(category=df["predicted_category"]), horizon=horizon)
    hist_df = res.history.assign(kind="history")
    fc_df = res.forecast.assign(kind="forecast")
    combined = pd.concat([hist_df, fc_df], ignore_index=True)
    st.plotly_chart(
        px.line(combined, x="month", y="amount", color="category", line_dash="kind"),
        use_container_width=True,
    )
    st.dataframe(
        fc_df.pivot(index="month", columns="category", values="amount").round(0),
        use_container_width=True,
    )

# ── Tab 5: Budget ─────────────────────────────────────────────────────────────
with tab5:
    target = st.slider("Savings target (%)", 5, 40, 15) / 100
    rec = recommend(df.assign(category=df["predicted_category"]), savings_target_pct=target)

    st.subheader("Budget vs. Actual spend")
    bdf = rec[["category", "median_spend", "recommended_cap"]].melt(
        id_vars="category", var_name="type", value_name="amount"
    )
    bdf["type"] = bdf["type"].map({"median_spend": "Actual (median)", "recommended_cap": "Recommended cap"})
    st.plotly_chart(
        px.bar(bdf, x="category", y="amount", color="type", barmode="group",
               color_discrete_map={"Actual (median)": "#F97316", "Recommended cap": "#7C3AED"}),
        use_container_width=True,
    )

    st.subheader("Recommended monthly caps")
    st.dataframe(rec, use_container_width=True)
    st.success(f"Projected monthly savings if you follow these caps: ₹{rec['projected_savings'].sum():,.0f}")

# ── Tab 6: Patterns ───────────────────────────────────────────────────────────
with tab6:
    st.subheader("Spending patterns")

    col_l, col_r = st.columns(2)
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    with col_l:
        st.markdown("**Spend by day of week**")
        dow_df = (
            df.assign(dow=df["timestamp"].dt.dayofweek)
            .groupby("dow")["amount"].sum().reset_index()
        )
        dow_df["day"] = dow_df["dow"].map(dict(enumerate(dow_labels)))
        st.plotly_chart(
            px.bar(dow_df, x="day", y="amount", color="amount",
                   color_continuous_scale="Purples",
                   category_orders={"day": dow_labels}),
            use_container_width=True,
        )

    with col_r:
        st.markdown("**Spend by hour of day**")
        hour_df = (
            df.assign(hour=df["timestamp"].dt.hour)
            .groupby("hour")["amount"].sum().reset_index()
        )
        st.plotly_chart(
            px.bar(hour_df, x="hour", y="amount", color="amount",
                   color_continuous_scale="Purples"),
            use_container_width=True,
        )

    st.markdown("**Daily spend heatmap**")
    cal_df = df.assign(date=df["timestamp"].dt.date).groupby("date")["amount"].sum().reset_index()
    cal_df["date"] = pd.to_datetime(cal_df["date"])
    cal_df["week"] = (cal_df["date"] - cal_df["date"].min()).dt.days // 7
    cal_df["dow"] = cal_df["date"].dt.dayofweek
    grid = cal_df.pivot_table(index="dow", columns="week", values="amount", fill_value=0)
    fig_cal = go.Figure(go.Heatmap(
        z=grid.values,
        y=[dow_labels[i] for i in grid.index],
        colorscale=[[0, "#1A1A2E"], [0.4, "#4C1D95"], [1, "#A78BFA"]],
        hovertemplate="Week %{x}, %{y}: ₹%{z:,.0f}<extra></extra>",
    ))
    fig_cal.update_layout(height=230, margin=dict(l=40, r=10, t=10, b=10),
                          xaxis=dict(showticklabels=False))
    st.plotly_chart(fig_cal, use_container_width=True)

    st.markdown("**Monthly category trends**")
    monthly_cat = (
        df.assign(month=df["timestamp"].dt.to_period("M").astype(str))
        .groupby(["month", "predicted_category"])["amount"].sum().reset_index()
    )
    st.plotly_chart(
        px.line(monthly_cat, x="month", y="amount", color="predicted_category", markers=True),
        use_container_width=True,
    )

    st.markdown("**Average transaction size by category**")
    avg_txn = df.groupby("predicted_category")["amount"].mean().reset_index().sort_values("amount", ascending=False)
    avg_txn.columns = ["category", "avg_amount"]
    st.plotly_chart(
        px.bar(avg_txn, x="category", y="avg_amount", color="avg_amount",
               color_continuous_scale="Purples"),
        use_container_width=True,
    )

# ── Tab 7: Merchants ──────────────────────────────────────────────────────────
with tab7:
    st.subheader("Top 20 merchants by total spend")
    top_m = (
        df.groupby("merchant").agg(
            total_spend=("amount", "sum"),
            transactions=("amount", "count"),
            avg_transaction=("amount", "mean"),
        )
        .reset_index()
        .sort_values("total_spend", ascending=False)
        .head(20)
    )
    top_m["avg_transaction"] = top_m["avg_transaction"].round(0)
    top_m["total_spend"] = top_m["total_spend"].round(0)

    st.plotly_chart(
        px.bar(top_m.head(15), x="merchant", y="total_spend",
               color="total_spend", color_continuous_scale="Purples",
               hover_data=["transactions", "avg_transaction"]),
        use_container_width=True,
    )
    st.dataframe(top_m, use_container_width=True)

    st.subheader("Merchants by category")
    selected_cat = st.selectbox("Select a category to drill down", options=all_cats)
    cat_m = (
        df[df["predicted_category"] == selected_cat]
        .groupby("merchant").agg(total=("amount", "sum"), count=("amount", "count"))
        .reset_index()
        .sort_values("total", ascending=False)
        .head(10)
    )
    if len(cat_m) > 0:
        col_pie, col_tbl = st.columns([1, 1])
        with col_pie:
            st.plotly_chart(
                px.pie(cat_m, names="merchant", values="total", hole=0.4,
                       color_discrete_sequence=px.colors.sequential.Purples_r),
                use_container_width=True,
            )
        with col_tbl:
            st.dataframe(cat_m, use_container_width=True)

    st.subheader("Search transactions by merchant")
    search = st.text_input("Merchant name", placeholder="e.g. Swiggy, Amazon, Uber...")
    if search:
        hits = df[df["merchant"].str.contains(search, case=False, na=False)]
        st.caption(f"{len(hits)} transactions found")
        st.dataframe(
            hits[["timestamp", "source", "amount", "merchant", "predicted_category", "anomaly_flag"]]
            .sort_values("timestamp", ascending=False),
            use_container_width=True,
        )

# ── Tab 8: Model Info ─────────────────────────────────────────────────────────
with tab8:
    st.subheader("Model performance")
    metrics_path = MODELS_DIR / "metrics.json"
    report_path = MODELS_DIR / "classification_report.txt"

    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        col1, col2, col3 = st.columns(3)
        col1.metric("Categorizer macro-F1", f"{m['categorizer_macro_f1']:.3f}")
        col2.metric("Anomaly ROC-AUC", f"{m['anomaly_roc_auc']:.3f}")
        col3.metric("Training rows", f"{m['n_rows']:,}")
    else:
        st.warning("metrics.json not found — retrain models to generate.")

    if report_path.exists():
        with st.expander("Full classification report"):
            st.code(report_path.read_text(), language="text")
