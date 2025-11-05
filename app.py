import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import feedparser
import requests

st.set_page_config(page_title="Pinterest Growing Trends: Insightboard", layout="wide")

# ---- HEADER ----
st.title("📈 Pinterest Growing Trends: Insightboard")
st.caption("Visualize Pinterest's Growing Trends by market, compare patterns, and connect search spikes to real-world events and news.")

# ---- FILE UPLOAD ----
uploaded_file = st.file_uploader("📤 Upload your Pinterest 'Growing Trends' CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    # Identify date columns (weekly columns like F–R)
    date_cols = [c for c in df.columns if any(x in c for x in ["/", "-", "202"])]
    meta_cols = [c for c in df.columns if c not in date_cols]

    # Melt into long format
    long_df = df.melt(id_vars=meta_cols, value_vars=date_cols, var_name="Week", value_name="Volume_norm")
    long_df["Week"] = pd.to_datetime(long_df["Week"], errors="coerce")

    # Detect key columns
    if "Trend" not in long_df.columns:
        trend_col = next((c for c in long_df.columns if "trend" in c.lower() or "keyword" in c.lower()), None)
        if trend_col:
            long_df.rename(columns={trend_col: "Trend"}, inplace=True)
        else:
            st.error("No 'Trend' or 'Keyword' column found. Please rename appropriately.")
            st.stop()

    if "Market" not in long_df.columns:
        st.error("Missing 'Market' column. Please add one (e.g., US, JP, IN).")
        st.stop()

    long_df["Volume_norm"] = pd.to_numeric(long_df["Volume_norm"], errors="coerce")
    long_df = long_df.dropna(subset=["Week", "Volume_norm"])

    # ---- SIDEBAR FILTERS ----
    with st.sidebar:
        st.header("🔍 Filters")

        markets = sorted(long_df["Market"].dropna().unique().tolist())
        market_sel = st.multiselect("Markets", markets, default=markets)

        trends = sorted(long_df["Trend"].dropna().unique().tolist())
        trend_sel = st.multiselect("Trends (optional)", trends, default=trends[:5])

        min_date, max_date = long_df["Week"].min(), long_df["Week"].max()
        date_range = st.date_input(
            "Date range", [min_date, max_date], min_value=min_date, max_value=max_date
        )

    # Apply filters
    view = long_df[
        (long_df["Market"].isin(market_sel))
        & (long_df["Week"].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))
    ]
    if trend_sel:
        view = view[view["Trend"].isin(trend_sel)]

    # ---- COMPARE TRENDS (Top N version) ----
    st.markdown("### Compare Trends")
    spikes = (
        view.groupby(["Trend", "Market"], as_index=False)["Volume_norm"]
        .mean()
        .sort_values("Volume_norm", ascending=False)
    )
    topN = st.slider("Top N", 3, 15, 5)
    top_pairs = spikes.head(topN)[["Trend", "Market"]].drop_duplicates().values.tolist()
    if not top_pairs and len(view):
        top_pairs = (
            list(
                view.groupby(["Trend", "Market"])["Volume_norm"]
                .mean()
                .sort_values(ascending=False)
                .head(topN)
                .index
            )
        )

    p = view.copy()
    mask = pd.Series(False, index=p.index)
    for tr, mk in top_pairs:
        mask = mask | ((p["Trend"] == tr) & (p["Market"] == mk))
    p = p[mask]

    if len(p):
        line = p.pivot_table(
            index="Week",
            columns=p["Trend"] + " • " + p["Market"],
            values="Volume_norm",
            aggfunc="mean",
        )
        st.line_chart(line)
    else:
        st.info("No data available for selected filters.")

    # ---- HEATMAP ----
    st.subheader("🔥 Weekly Heatmap")
    if len(view):
        heat_df = (
            view.pivot_table(
                index=["Trend", "Market"], columns="Week", values="Volume_norm", aggfunc="mean"
            )
            .fillna(0)
        )
        fig = px.imshow(
            heat_df.values,
            labels=dict(x="Week", y="Trend • Market", color="Volume (norm)"),
            x=[w.strftime("%Y-%m-%d") for w in heat_df.columns],
            y=[f"{i[0]} • {i[1]}" for i in heat_df.index],
            aspect="auto",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---- NEWS CONTEXT (always visible) ----
    st.subheader("🗞️ News / Event Context")
    with st.expander("View latest related headlines"):
        keyword = st.text_input("Enter a keyword or trend:", "")
        selected_market = st.selectbox("Optional market filter:", ["All"] + markets)

        if keyword:
            try:
                query = keyword if selected_market == "All" else f"{keyword} {selected_market}"
                feed_url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
                feed = feedparser.parse(feed_url)
                if feed.entries:
                    for entry in feed.entries[:10]:
                        st.markdown(f"**[{entry.title}]({entry.link})**")
                        if hasattr(entry, "published"):
                            st.caption(entry.published)
                        st.write("---")
                else:
                    st.info("No recent news found for that query.")
            except Exception as e:
                st.error(f"Error fetching news: {e}")

    # ---- EXPORT ----
    st.download_button(
        label="💾 Download filtered data as CSV",
        data=view.to_csv(index=False),
        file_name="pinterest_trends_filtered.csv",
        mime="text/csv",
    )

    # ---- FOOTER ----
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center; font-size:14px;">Created by: '
        '<a href="https://www.linkedin.com/in/limwuiliang/" target="_blank" '
        'style="color:#0072b1; text-decoration:none;">Wui-Liang Lim</a></div>',
        unsafe_allow_html=True,
    )

else:
    st.info(
        """
        👋 **Welcome to Pinterest Growing Trends: Insightboard**

        **To get started:**
        1. Visit [trends.pinterest.com](https://trends.pinterest.com)
        2. Under *Trends Type*, select **Growing Trends**
        3. Choose a *Region* (Pinterest only allows selection of one Region as of now) and export the CSV. 
        4. Remove **Rank** column
        5. Add a column called **Market** (e.g., US, JP, IN)
        6. If comparing multiple regions, repeat the export for each region, paste all into one sheet, and upload it here.
        """
    )
