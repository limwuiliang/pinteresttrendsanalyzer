import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
from dateutil import parser as dateparser
import re

# Optional Plotly (nice charts) ------------------------------------------------
try:
    import plotly.express as px
    USE_PLOTLY = True
except Exception:
    px = None
    USE_PLOTLY = False

# Optional RSS deps for News / Event Context ----------------------------------
try:
    import feedparser
    import urllib.parse
except Exception:
    feedparser = None

st.set_page_config(page_title="Pinterest Growing Trends: Insightboard", layout="wide")

# =============================================================================
# Data helpers
# =============================================================================
@st.cache_data
def load_data(uploaded_file=None):
    """Load CSV from upload; auto-detect week columns that look like dates."""
    if uploaded_file is None:
        return None, None, None

    df = pd.read_csv(uploaded_file)

    # Detect weekly/date columns vs metadata
    date_cols, meta_cols = [], []
    for c in df.columns:
        try:
            dateparser.parse(str(c), dayfirst=False, yearfirst=False, fuzzy=False)
            if pd.api.types.is_numeric_dtype(df[c]):
                date_cols.append(c)
            else:
                meta_cols.append(c)
        except Exception:
            meta_cols.append(c)

    if len(date_cols) == 0:
        # Fallback: last N numeric columns as weeks
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        date_cols = num_cols[-10:]
        meta_cols = [c for c in df.columns if c not in date_cols]

    try:
        date_cols = sorted(date_cols, key=lambda x: dateparser.parse(str(x)))
    except Exception:
        pass

    return df, meta_cols, date_cols


def tidy_data(df, meta_cols, date_cols):
    """Wide -> long, compute normalized volume."""
    long = df.melt(id_vars=meta_cols, value_vars=date_cols, var_name="Week", value_name="Volume")

    def safe_parse(x):
        try:
            return dateparser.parse(str(x)).date()
        except Exception:
            return None

    long["Week"] = long["Week"].apply(safe_parse)
    long = long.dropna(subset=["Week"])

    # Expect required columns
    if "Trend" not in long.columns or "Market" not in long.columns:
        raise ValueError("CSV must include 'Trend' and 'Market' columns.")

    long["Volume_raw"] = long["Volume"]
    long["Volume_norm"] = long.groupby(["Trend", "Market"])["Volume_raw"].transform(
        lambda s: (s - s.min()) / (s.max() - s.min()) * 100 if (s.max() > s.min()) else 0.0
    )

    return long


def compute_growth(long):
    """WoW deltas + rolling z-score on normalized volume; used to rank 'spikes'."""
    long = long.sort_values(["Trend", "Market", "Week"])
    long["WoW"] = long.groupby(["Trend", "Market"])["Volume_raw"].pct_change() * 100.0
    long["WoW_norm"] = long.groupby(["Trend", "Market"])["Volume_norm"].pct_change() * 100.0

    def rolling_z(x, win=4):
        r = x.rolling(win, min_periods=2)
        return (x - r.mean()) / r.std(ddof=0)

    long["Zscore"] = long.groupby(["Trend", "Market"])["Volume_norm"].transform(rolling_z)
    return long


# =============================================================================
# News / Events (always visible)
# =============================================================================
def google_news_rss(query: str, market: str = None, from_date=None, to_date=None, max_items=12):
    """Fetch headlines via Google News RSS. No API key required."""
    if feedparser is None or not query:
        return []
    q = query if (not market or market.lower() in query.lower()) else f"{query} {market}"
    encoded = urllib.parse.quote_plus(q)
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
    try:
        feed = feedparser.parse(url)
    except Exception:
        return []
    items = []
    for e in feed.entries[:max_items * 2]:
        pub = None
        try:
            if hasattr(e, "published_parsed") and e.published_parsed:
                pub = datetime(*e.published_parsed[:6]).date()
        except Exception:
            pub = None
        if from_date and pub and pub < from_date:
            continue
        if to_date and pub and pub > to_date:
            continue
        items.append({
            "title": getattr(e, "title", ""),
            "link": getattr(e, "link", ""),
            "published": str(pub) if pub else "",
            "source": getattr(getattr(e, "source", {}), "title", ""),
        })
    return items[:max_items]


# =============================================================================
# UI
# =============================================================================
def main():
    st.title("Pinterest Growing Trends: Insightboard")
    if not USE_PLOTLY:
        st.warning(
            "Plotly isn't installed — using fallback visuals. "
            "To enable Plotly charts on Streamlit Cloud, ensure `requirements.txt` at repo root includes `plotly` and redeploy.",
            icon="⚠️"
        )

    with st.expander("First-time instructions"):
        st.markdown(
            """
**How to get the CSV from Pinterest Trends (Growing Trends):**

1. Visit **[trends.pinterest.com](https://trends.pinterest.com/)**.  
2. Under **Trends Type**, select **Growing Trends**.  
3. Under **Region**, select **one** region you’re interested in (Pinterest currently exports one region at a time).  
4. Export/download the **CSV**.  
5. In your spreadsheet, remove the 'Rank" column. Create a **column named `Market`** and fill it with the **country code or region label** that matches the export (e.g., `US`, `JP`, `AU`, `SG`, etc.).  
6. If you want to compare **multiple regions**, repeat steps 2–5 for each region and **paste the rows into the same sheet** (keeping the same columns, and `Market` set accordingly).  

**Expected columns in the CSV you upload here:**
- `Trend`, `Market`, and weekly columns whose headers are date-like (e.g., `7/25/25`, `8/1/25`, ...).  
- Other columns (e.g., `Weekly change`, `Monthly change`) are fine — they’ll be treated as metadata.
"""
        )

    # --- Sidebar: file upload -------------------------------------------------
    with st.sidebar:
        st.header("Data")
        uploaded = st.file_uploader("Upload your Pinterest Growing Trends CSV", type=["csv"])
        if uploaded is None:
            st.info("Upload a CSV to begin.")
            st.stop()

    # --- Load & shape data ----------------------------------------------------
    try:
        df, meta_cols, date_cols = load_data(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    try:
        long = tidy_data(df, meta_cols, date_cols)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Compute growth metrics (WoW_norm, Zscore) we use for 'spikes' ranking
    long = compute_growth(long)

    # --- Filters: all markets shown and selected by default -------------------
    all_markets = sorted(long["Market"].dropna().unique().tolist())
    if not all_markets:
        st.error("No values found in 'Market' column. Please ensure you've added it.")
        st.stop()

    col_filters = st.container()
    with col_filters:
        c1, c2 = st.columns([2, 3])

        with c1:
            selected_markets = st.multiselect(
                "Markets",
                options=all_markets,
                default=all_markets  # select ALL by default
            )

        # Full date range default (from the uploaded sheet)
        min_week, max_week = long["Week"].min(), long["Week"].max()
        with c2:
            dr = st.slider(
                "Date range",
                min_value=min_week,
                max_value=max_week,
                value=(min_week, max_week)  # default to full range
            )

    # Optional trend text filter used below (for your requested logic)
    trend_query = st.text_input("Find keywords (regex ok)", "")
    trends_all = sorted(long.query("Market in @selected_markets and @dr[0] <= Week <= @dr[1]")["Trend"].dropna().unique().tolist())
    if trend_query:
        trends_filtered = [t for t in trends_all if pd.Series([t]).str.contains(trend_query, case=False, regex=True).item()]
    else:
        trends_filtered = trends_all

    # Current view (filtered base)
    view = long.query("Market in @selected_markets and @dr[0] <= Week <= @dr[1] and Trend in @trends_filtered")

    # Prepare 'spikes' silently (used to pick top pairs for Compare Trends)
    if view.empty:
        st.info("No rows in the current filter. Adjust markets, date range, or keyword filter.")
        st.stop()

    latest_week = view["Week"].max()
    latest = view[view["Week"] == latest_week].copy()
    # Build a Score as in your earlier spec: WoW_norm + 25*Zscore
    latest["Score"] = latest["WoW_norm"].fillna(0) + latest["Zscore"].fillna(0) * 25
    spikes = latest.sort_values("Score", ascending=False)

    # --- YOUR REQUESTED COMPARE TRENDS BLOCK ---------------------------------
    st.markdown("### Compare Trends")
    topN = st.slider("Select top N trends (by Score) to plot", 3, 15, 5)
    top_trends = spikes.head(topN)[["Trend", "Market"]].drop_duplicates().values.tolist()
    if len(top_trends) == 0 and len(trends_filtered) > 0:
        fallback = view.groupby(["Trend", "Market"])["Volume_norm"].mean().sort_values(ascending=False).head(topN).index.tolist()
        top_trends = list(fallback)

    plot_df = view.copy()
    mask = pd.Series(False, index=plot_df.index)
    for tr, mk in top_trends:
        mask = mask | ((plot_df["Trend"] == tr) & (plot_df["Market"] == mk))
    plot_df = plot_df[mask]

    if len(plot_df):
        if USE_PLOTLY:
            fig2 = px.line(
                plot_df, x="Week", y="Volume_norm",
                color=plot_df["Trend"] + " • " + plot_df["Market"],
                markers=True
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            piv = plot_df.pivot_table(
                index="Week",
                columns=plot_df["Trend"] + " • " + plot_df["Market"],
                values="Volume_norm",
                aggfunc="mean"
            )
            st.line_chart(piv)
    else:
        st.info("No data matches the current filters. Try different markets, date range, or keyword filter.")

    # --- Heatmap --------------------------------------------------------------
    st.markdown("### Weekly Heatmap (Normalized Volume)")
    heat_df = view.pivot_table(
        index=["Trend", "Market"], columns="Week",
        values="Volume_norm", aggfunc="mean"
    ).fillna(0)

    if heat_df.shape[0] > 200:
        heat_df = heat_df.head(200)
        st.info("Showing first 200 rows for performance. Use filters to narrow further.")

    if USE_PLOTLY:
        fig_hm = px.imshow(
            heat_df.values,
            labels=dict(x="Week", y="Trend • Market", color="Volume (norm)"),
            x=[w.strftime("%Y-%m-%d") for w in heat_df.columns],
            y=[f"{i[0]} • {i[1]}" for i in heat_df.index],
            aspect="auto"
        )
        st.plotly_chart(fig_hm, use_container_width=True)
    else:
        styled = heat_df.copy()
        styled.index = [f"{i[0]} • {i[1]}" for i in styled.index]
        styled.columns = [w.strftime("%Y-%m-%d") for w in styled.columns]
        st.dataframe(styled.style.background_gradient(axis=None))

    # --- News / Event Context (ALWAYS visible) --------------------------------
    st.markdown("### News / Event Context")
    n1, n2 = st.columns([2, 1])

    with n1:
        default_kw = top_trends[0][0] if len(top_trends) else (trends_filtered[0] if trends_filtered else "")
        kw = st.text_input("Keyword for news search", default_kw)
        mk = st.selectbox("Market for news bias (optional)", options=["(none)"] + all_markets)
        mk_val = None if mk == "(none)" else mk
        news_from, news_to = st.date_input("News date window", (dr[0], dr[1]))

        if st.button("Fetch Headlines"):
            items = google_news_rss(kw, mk_val, from_date=news_from, to_date=news_to, max_items=12)
            if not items:
                if feedparser is None:
                    st.info("RSS parser not available in this environment. The app will still work without headlines.")
                else:
                    st.info("No headlines found. Try a broader keyword or wider date window.")
            else:
                for it in items:
                    st.markdown(
                        f"- [{it['title']}]({it['link']})  \n"
                        f"  <small>{it.get('published','')} · {it.get('source','')}</small>",
                        unsafe_allow_html=True,
                    )

    with n2:
        st.caption(
            "Use a trend keyword and (optionally) a market to bias search results, "
            "constrained to your selected date range."
        )

    # --- Export ---------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Export Filtered Data")
    exp_cols = ["Trend", "Market", "Week", "Volume_raw", "Volume_norm", "WoW", "WoW_norm", "Zscore"]
    csv_bytes = view.sort_values(["Trend", "Market", "Week"])[exp_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered data (CSV)",
        data=csv_bytes,
        file_name="pinterest_trends_filtered.csv",
        mime="text/csv"
    )

    # --- FOOTER ---------------------------------------------------------------
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center; font-size:14px;">Created by: '
        '<a href="https://www.linkedin.com/in/limwuiliang/" target="_blank" '
        'style="color:#0072b1; text-decoration:none;">Wui-Liang Lim</a></div>',
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
