import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "processed" / "risk_results.csv"

st.set_page_config(
    page_title="FORESIGHT | Demand Intelligence",
    page_icon="📊",
    layout="wide"
)

st.title("FORESIGHT")
st.subheader("AI-Powered Demand & Inventory Intelligence")
st.caption(
    "Retail demand forecasting and inventory risk monitoring. "
    "Inventory values are illustrative assumptions, not actual stock records."
)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["week"] = pd.to_datetime(df["week"])
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
departments = ["All"] + sorted(df["id"].str.split("_").str[0:2].str.join("_").unique().tolist())
selected_department = st.sidebar.selectbox("Department / SKU group", departments)

if selected_department != "All":
    df = df[df["id"].str.startswith(selected_department)]

risk_options = ["All"] + sorted(df["risk_status"].unique().tolist())
selected_risk = st.sidebar.selectbox("Risk status", risk_options)

if selected_risk != "All":
    df = df[df["risk_status"] == selected_risk]

# KPI cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Forecast records", f"{len(df):,}")
c2.metric("Stockout-risk records", f"{(df['risk_status'] == 'Stockout Risk').sum():,}")
c3.metric("Overstock-risk records", f"{(df['risk_status'] == 'Overstock Risk').sum():,}")
c4.metric("Normal records", f"{(df['risk_status'] == 'Normal').sum():,}")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Weekly sales vs forecast")
    trend = df.groupby("week", as_index=False)[["weekly_sales", "forecast"]].sum()
    trend = trend.melt(
        id_vars="week",
        value_vars=["weekly_sales", "forecast"],
        var_name="Measure",
        value_name="Units"
    )
    fig = px.line(trend, x="week", y="Units", color="Measure")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Risk distribution")
    risk_counts = df["risk_status"].value_counts().rename_axis("Risk").reset_index(name="Records")
    fig = px.pie(risk_counts, names="Risk", values="Records", hole=0.45)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Risk records")
st.dataframe(
    df[[
        "id", "week", "weekly_sales", "forecast",
        "assumed_on_hand", "projected_lead_time_demand", "risk_status"
    ]].sort_values("week", ascending=False),
    use_container_width=True
)

st.download_button(
    "Download filtered risk data",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="foresight_risk_results.csv",
    mime="text/csv"
)