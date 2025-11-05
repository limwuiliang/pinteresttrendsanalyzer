import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import feedparser
import requests
from io import StringIO

st.set_page_config(page_title="Pinterest Growing Trends: Insightboard", layout="wide")

# ---- HEADER ----
st.title("📈 Pinterest Growing Trends: Insightboard")
st.caption("Visualize and explore Pinterest's Growing Trends by market, compare patterns, and tie spikes to real-world events and news.")

# ---- FILE UPLOAD ----
uploaded_file = st.file_uploader("📤 Upload your Pinterest 'Growing Trends' CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    # Identify date columns (week columns F→R style)
    date_cols = [c for c in df.columns if any(x in c for x in ["/", "-", "202"])]
    meta_cols = [c for c in df.columns if c not in date_cols]

    # Melt to long format
    long_df = df.melt(id_vars=meta_cols, value_vars=date_cols,
                      var_name="Week", value_name="Volume_norm")

    # Convert week to datetime
    long_df["Week"] = pd.to_datetime(long_df["Week"], errors="coerce")

    # Convert market and trend columns
    if "Trend" not in long_df.columns:
        # Try to find similar columns (Pinterest export may call it "keyword" or "search term")
        trend_col = next((c for c in long_df.columns if "trend" in c.lower() or "keyword" in c.lower()), None)
        if trend_col:
            long_df.rename(columns={trend_col: "Trend"}, inplace=True)
        else:
            st.error("Could not find a column for 'Trend' or 'Keyword'. Please rename one of the columns.")
    if "Market" not in long_df.columns:
        st.error("Please include a 'Market' column in your CSV (e.g., JP, US, IN).")
        st.stop()

    # Drop missing dates or volumes
    long_df = long_df.dropna(subset=["Week", "Volume_norm"])

    # Ensure numeric
    long_df["Volume_norm"] = pd.to_numeric(long_df["Volume_norm"], errors="coerce")

    # Sidebar filters
    with st.sidebar:
        st.header("🔍 Filters")

        markets = sorted(long_df["Market"].dropna().unique().tolist())
        market_sel = st.multiselect("Markets", markets, default=markets)

        trends = sorted(long_df["Trend"].dropna().unique().tolist())
        trend_sel = st.multiselect("Trends", trends, default=trends[:5])

        # Default date range = min/max in data
        min_date, max_date = long_df["Week"].min(), long_df["Week"].max()
        date_range = st.date_input(
            "Date range", [min_date, max_date],
            min_value=min_date, max_value=max_date
        )

    # Filter by selection
    mask = (
        long_df["Market"].isin(market_sel)
        & long_df["Trend"].isin(trend_sel)
        & (long_df["Week"].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))
    )
    filtered = long_df[mask]

    # ---- COMPARE TRENDS ----
    st.subheader("📊 Compare Trends Over Time")

    if len(filtered):
        fig2 = px.line(
            filtered,
            x="Week",
            y="Volume_norm",
            color=filtered["Trend"] + " • " + filtered["Market"],
            markers=True,
            title="Normalized Trend Search Volume Over Time"
        )
        fig2.update_layout(legend_title_text="Trend • Market", height=500)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No data available for the selected filters.")

    # ---- HEATMAP ----
    st.subheader("🔥 Weekly Heatmap")

    if len(filtered):
        heat_df = (
            filtered.pivot_table(
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
    else:
        st.info("Upload your file to see the heatmap.")

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
                        if hasattr(entry, 'published'):
                            st.caption(entry.published)
                        st.write("---")
                else:
                    st.info("No recent news found for that query.")
            except Exception as e:
                st.error(f"Error fetching news: {e}")

    # ---- EXPORT ----
    st.download_button(
        label="💾 Download filtered data as CSV",
        data=filtered.to_csv(index=False),
        file_name="pinterest_trends_filtered.csv",
        mime="text/csv",
    )

    # ---- FOOTER ----
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center; font-size:14px;">Created by: '
        '<a href="https://www.linkedin.com/in/limwuiliang/" target="_blank" style="color:#0072b1; text-decoration:none;">Wui-Liang Lim</a>'
        '</div>',
        unsafe_allow_html=True
    )

else:
    st.info(
        """
        👋 **Welcome to Pinterest Growing Trends: Insightboard**

        **To get started:**
        1. Visit [trends.pinterest.com](https://trends.pinterest.com)
        2. Under *Trends Type*, select **Growing Trends**
        3. Choose a *Region* and export the CSV
        4. Add a column called **Market** in your sheet (e.g., US, JP, IN)
        5. If comparing multiple regions, repeat the export for each region, paste all into one sheet, and upload it here.
        """
    )
