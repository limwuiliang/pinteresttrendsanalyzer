import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
from dateutil import parser as dateparser
import re

# ---- Plotly (guarded import with graceful fallback) -------------------------
try:
    import plotly.express as px
    USE_PLOTLY = True
except Exception:
    px = None
    USE_PLOTLY = False

# ---- Optional news deps (only used if toggled on) ---------------------------
try:
    import feedparser
    import urllib.parse
except Exception:
    feedparser = None

st.set_page_config(page_title="Pinterest Growing Trends — APAC", layout="wide")

# =============================================================================
# Data loading & transforms
# =============================================================================
@st.cache_data
def load_data(default_path: str = None, uploaded_file=None):
    """Load CSV from upload or default path; auto-detect weekly date columns."""
    df, src = None, None

    # Prefer uploaded file
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        src = "uploaded"

    # Otherwise, try default path if provided
    elif default_path:
        try:
            df = pd.read_csv(default_path)
            src = "default"
        except Exception:
            df, src = None, None  # fallback to uploader-only mode

    if df is None:
        return None, None, None, src

    # Identify week/date columns (wide format) vs metadata
    date_cols, meta_cols = [], []
    for c in df.columns:
        try:
            dateparser.parse(str(c), dayfirst=False, yearfirst=False, fuzzy=False)
            # consider it a week column only if the column is numeric
            if pd.api.types.is_numeric_dtype(df[c]):
                date_cols.append(c)
            else:
                meta_cols.append(c)
        except Exception:
            meta_cols.append(c)

    # Fallback: if no parsed dates, take last 10 numeric columns as weeks
    if len(date_cols) == 0:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        date_cols = num_cols[-10:]
        meta_cols = [c for c in df.columns if c not in date_cols]

    # Sort date columns chronologically if possible
    try:
        date_cols = sorted(date_cols, key=lambda x: dateparser.parse(str(x)))
    except Exception:
        pass

    # Normalize expected names (we require Trend + Market)
    df.rename(columns={"Trend": "Trend", "Market": "Market"}, inplace=True)

    return df, meta_cols, date_cols, src


def tidy_data(df, meta_cols, date_cols):
    """Wide-to-long; compute normalized volumes per Trend/Market."""
    long = df.melt(id_vars=meta_cols, value_vars=date_cols, var_name="Week", value_name="Volume")

    def safe_parse(x):
        try:
            return dateparser.parse(str(x)).date()
        except Exception:
            return None

    long["Week"] = long["Week"].apply(safe_parse)
    long = long.dropna(subset=["Week"])

    long["Volume_raw"] = long["Volume"]
    long["Volume_norm"] = long.groupby(["Trend", "Market"])["Volume_raw"].transform(
        lambda s: (s - s.min()) / (s.max() - s.min()) * 100 if (s.max() > s.min()) else 0.0
    )
    return long


def compute_growth(long):
    """WoW deltas + rolling z-score on normalized volume."""
    long = long.sort_values(["Trend", "Market", "Week"])
    long["WoW"] = long.groupby(["Trend", "Market"])["Volume_raw"].pct_change() * 100.0
    long["WoW_norm"] = long.groupby(["Trend", "Market"])["Volume_norm"].pct_change() * 100.0

    def rolling_z(x, win=4):
        r = x.rolling(win, min_periods=2)
        return (x - r.mean()) / r.std(ddof=0)

    long["Zscore"] = long.groupby(["Trend", "Market"])["Volume_norm"].transform(rolling_z)
    return long


# =============================================================================
# APAC events catalog (file-based heuristics)
# Only events for markets present in the sheet + international ("any")
# =============================================================================
EVENTS_CATALOG = [
    # International
    {"name": "Halloween", "markets": "any",
     "patterns": [r"halloween", r"pumpkin", r"spooky", r"costume"],
     "window": ("2025-10-01", "2025-10-31"), "notes": "Seasonal decor, costumes, recipes"},
    {"name": "Black Friday / Cyber Monday", "markets": "any",
     "patterns": [r"black\s*friday", r"cyber\s*monday", r"bfcm", r"sale"],
     "window": ("2025-11-20", "2025-12-05"), "notes": "Retail promotion period"},
    {"name": "Back to School", "markets": "any",
     "patterns": [r"back\s*to\s*school", r"school\s*supply", r"uniform"],
     "window": ("2025-07-15", "2025-09-15"), "notes": "Varies by market"},
    # East Asia
    {"name": "Mid-Autumn Festival / Mooncake", "markets": ["Singapore", "Malaysia", "Hong Kong", "Taiwan", "China"],
     "patterns": [r"mooncake", r"lantern", r"mid[-\s]?autumn"],
     "window": ("2025-09-01", "2025-10-10"), "notes": "Lanterns, mooncakes"},
    {"name": "Chuseok (Korea)", "markets": ["Korea"],
     "patterns": [r"chuseok", r"hanbok", r"songpyeon"],
     "window": ("2025-09-01", "2025-10-05"), "notes": "Korean thanksgiving"},
    {"name": "Obon / Summer Festivals", "markets": ["Japan"],
     "patterns": [r"yukata", r"hanabi", r"matsuri", r"obon"],
     "window": ("2025-07-01", "2025-08-31"), "notes": "Summer festivals JP"},
    # South / SEA
    {"name": "Diwali / Deepavali", "markets": ["India", "Singapore", "Malaysia"],
     "patterns": [r"diwali", r"deepavali", r"rangoli", r"diyas?"],
     "window": ("2025-10-10", "2025-11-20"), "notes": "Festival of lights"},
    {"name": "Merdeka Day (MY)", "markets": ["Malaysia"],
     "patterns": [r"merdeka", r"jalur\s+gemilang"],
     "window": ("2025-08-01", "2025-08-31"), "notes": "Malaysia Aug 31"},
    {"name": "National Day (SG)", "markets": ["Singapore"],
     "patterns": [r"national\s*day", r"ndp", r"sg\d{2}"],
     "window": ("2025-07-20", "2025-08-15"), "notes": "Singapore Aug 9"},
    {"name": "Thai Vegetarian Festival", "markets": ["Thailand", "Malaysia", "Singapore"],
     "patterns": [r"vegetarian\s*festival", r"nine\s*emperor", r"เจ"],
     "window": ("2025-09-25", "2025-10-15"), "notes": "Vegan festival"},
    # ANZ
    {"name": "Footy Finals (AFL/NRL)", "markets": ["Australia"],
     "patterns": [r"afl", r"nrl", r"footy\s*finals"],
     "window": ("2025-08-15", "2025-10-05"), "notes": "Sport finals AU"},
]

def prune_catalog(allowed_markets: set):
    """Keep only rules for markets present in data + 'any'."""
    pruned = []
    lower_allowed = {am.lower() for am in allowed_markets}
    for rule in EVENTS_CATALOG:
        m = rule.get("markets", "any")
        if m == "any":
            pruned.append(rule)
        else:
            if any(mm.lower() in lower_allowed for mm in m):
                pruned.append(rule)
    return pruned

def _in_window(week: date, window):
    if not window:
        return True
    try:
        start = dateparser.parse(window[0]).date() if isinstance(window[0], str) else window[0]
        end = dateparser.parse(window[1]).date() if isinstance(window[1], str) else window[1]
    except Exception:
        return True
    return (start is None or week >= start) and (end is None or week <= end)

def _market_match(market: str, rule_markets):
    if rule_markets == "any":
        return True
    return any(market.lower() == m.lower() for m in rule_markets)

def _keyword_match(keyword: str, patterns):
    k = str(keyword or "").lower()
    for pat in patterns:
        if re.search(pat, k, flags=re.IGNORECASE):
            return True
    return False

def auto_annotate(spikes_df: pd.DataFrame, catalog=None):
    rows = []
    catalog = catalog or EVENTS_CATALOG
    for _, r in spikes_df.iterrows():
        kw = str(r.get("Trend", ""))
        mk = str(r.get("Market", ""))
        wk = r.get("Week", None)
        for rule in catalog:
            if not _market_match(mk, rule["markets"]):
                continue
            if not _in_window(wk, rule.get("window")):
                continue
            if not _keyword_match(kw, rule["patterns"]):
                continue
            rows.append({
                "Trend": kw,
                "Market": mk,
                "Week": wk,
                "Matched Event": rule["name"],
                "Notes": rule.get("notes", ""),
                "WoW_norm": r.get("WoW_norm", np.nan),
                "Zscore": r.get("Zscore", np.nan),
                "Score": r.get("Score", np.nan),
            })
    return pd.DataFrame(rows)

# =============================================================================
# Optional: Google News RSS (off by default to keep file-based)
# =============================================================================
def google_news_rss(query: str, market: str = None, from_date=None, to_date=None, max_items=10):
    if feedparser is None:
        return []
    if not query:
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
    st.title("Pinterest Growing Trends — APAC Insightboard")
    st.caption("Interactive weekly trends with growth detection, auto-annotations (restricted to your markets + international), and optional news context.")

    if not USE_PLOTLY:
        st.warning(
            "Plotly isn't installed — using fallback visuals. "
            "To enable Plotly charts on Streamlit Cloud, ensure `requirements.txt` at repo root includes `plotly` and redeploy.",
            icon="⚠️"
        )

    # If you commit a sample CSV into the repo, you can point to it here:
    DEFAULT_PATH = ""  # leave blank for Cloud; users will upload via the sidebar

    with st.sidebar:
        st.header("Data")
        uploaded = st.file_uploader("Upload a Pinterest Trends CSV", type=["csv"])
        df, meta_cols, date_cols, src = load_data(DEFAULT_PATH if DEFAULT_PATH else None, uploaded)
        if df is None:
            st.info("Upload your CSV to get started. Expected columns include `Trend`, `Market`, and weekly columns (e.g., 7/25/25 ...).")
            st.stop()
        st.success(f"Loaded {len(df):,} rows • {len(date_cols)} weekly periods")

        st.write("Meta columns:", ", ".join(meta_cols))
        st.write("Week columns:", ", ".join(map(str, date_cols)))

        st.divider()
        st.subheader("Network Features")
        enable_news = st.toggle("Enable online news lookup (Google News RSS)", value=False,
                                help="Keeps the app file-based by default. Turn on to fetch headlines.")

    # Required columns
    if "Trend" not in df.columns or "Market" not in df.columns:
        st.error("Expected 'Trend' and 'Market' columns were not found in your CSV.")
        st.stop()

    # Prepare data
    long = tidy_data(df, meta_cols, date_cols)
    long = compute_growth(long)

    # Markets present in the sheet
    markets = sorted(long["Market"].dropna().unique().tolist())
    allowed_markets = set(markets)
    catalog_pruned = prune_catalog(allowed_markets)

    # Filters
    selected_markets = st.multiselect("Market(s)", markets, default=markets[:3] if len(markets) >= 3 else markets)
    trend_query = st.text_input("Find keywords (regex ok)", "")
    trends_all = long.query("Market in @selected_markets")["Trend"].unique().tolist()
    if trend_query:
        trends_filtered = [t for t in trends_all if pd.Series([t]).str.contains(trend_query, case=False, regex=True).item()]
    else:
        trends_filtered = trends_all

    min_week, max_week = long["Week"].min(), long["Week"].max()
    dr = st.slider("Date range", min_value=min_week, max_value=max_week, value=(min_week, max_week))

    # Current view
    view = long.query("Market in @selected_markets and @dr[0] <= Week <= @dr[1] and Trend in @trends_filtered")

    # Top spikes for the latest week in range
    st.markdown("### Top Weekly Spikes")
    threshold = st.slider("Spike threshold (WoW % on normalized volume)", min_value=10, max_value=300, value=50, step=5)

    if view.empty:
        st.info("No rows in the current filter. Adjust markets, date range, or keyword filter.")
        st.stop()

    latest_week = view["Week"].max()
    latest = view[view["Week"] == latest_week].copy()
    latest["Score"] = latest["WoW_norm"].fillna(0) + latest["Zscore"].fillna(0) * 25
    spikes = latest.sort_values("Score", ascending=False)
    spikes = spikes[spikes["WoW_norm"] >= threshold].head(25)
    st.dataframe(spikes[["Trend", "Market", "Week", "Volume_raw", "Volume_norm", "WoW_norm", "Zscore", "Score"]])

    col1, col2 = st.columns([2, 1], gap="large")

    # Heatmap + lines
    with col1:
        st.markdown("### Weekly Heatmap (Volume Norm)")
        heat_df = view.pivot_table(index=["Trend", "Market"], columns="Week", values="Volume_norm", aggfunc="mean").fillna(0)
        if heat_df.shape[0] > 100:
            heat_df = heat_df.head(100)
            st.info("Showing first 100 rows for performance. Use filters to narrow further.")
        if USE_PLOTLY:
            fig = px.imshow(
                heat_df.values,
                labels=dict(x="Week", y="Trend • Market", color="Volume (norm)"),
                x=[w.strftime("%Y-%m-%d") for w in heat_df.columns],
                y=[f"{i[0]} • {i[1]}" for i in heat_df.index],
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            styled = heat_df.copy()
            styled.index = [f"{i[0]} • {i[1]}" for i in styled.index]
            styled.columns = [w.strftime("%Y-%m-%d") for w in styled.columns]
            st.dataframe(styled.style.background_gradient(axis=None))

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

        st.markdown("### Auto-Annotate Spikes (Your markets + international)")
        st.caption("Heuristic annotations using an APAC events catalog (keywords + date windows + market filters).")
        ann = auto_annotate(spikes, catalog=catalog_pruned) if len(spikes) else pd.DataFrame()
        if ann.empty:
            st.info("No auto-annotations matched the current spikes. Try adjusting threshold, date range, or markets.")
        else:
            st.dataframe(ann)
            csv_bytes = ann.to_csv(index=False).encode("utf-8")
            st.download_button("Download annotations (CSV)", data=csv_bytes, file_name="trend_annotations.csv", mime="text/csv")

    # Manual tagging + optional news
    with col2:
        st.markdown("### Manual Tagging")
        st.caption("Attach your own event tag to a selected spike; exported with your session.")
        if len(spikes):
            options = [f"{r.Trend} • {r.Market} ({r.Week})" for _, r in spikes.iterrows()]
            sel = st.selectbox("Choose spike", options)
            tag = st.text_input("Event tag/notes", "")
            if st.button("Add tag"):
                if "tags" not in st.session_state:
                    st.session_state["tags"] = []
                st.session_state["tags"].append({"selection": sel, "tag": tag})
                st.success("Added.")
        if "tags" in st.session_state and st.session_state["tags"]:
            st.write("Your tags:")
            st.json(st.session_state["tags"])
            if st.button("Export tags CSV"):
                tdf = pd.DataFrame(st.session_state["tags"])
                st.download_button("Download tags.csv", data=tdf.to_csv(index=False), file_name="trend_tags.csv", mime="text/csv")

        st.markdown("---")
        st.markdown("### News / Event Context (Optional)")
        if enable_news and feedparser is not None:
            default_kw = spikes.iloc[0]["Trend"] if len(spikes) else (trends_filtered[0] if trends_filtered else "")
            kw = st.text_input("Keyword for news search", default_kw or "")
            mk = st.selectbox("Market for news bias", options=["(none)"] + selected_markets)
            mk_val = None if mk == "(none)" else mk
            news_from, news_to = st.date_input("News date window", (max(latest_week, dr[0]), max(latest_week, dr[1])))
            if st.button("Fetch Headlines"):
                items = google_news_rss(kw, mk_val, from_date=news_from, to_date=news_to, max_items=12)
                if not items:
                    st.info("No headlines found. Try a broader keyword or wider date range.")
                else:
                    for it in items:
                        st.markdown(f"- [{it['title']}]({it['link']})  \n  <small>{it['published']} · {it.get('source','')}</small>", unsafe_allow_html=True)
        else:
            st.caption("Online news lookup is disabled (file-based mode). Enable it from the sidebar if needed.")

    st.markdown("---")
    st.markdown("### Export Filtered Data")
    exp_cols = ["Trend", "Market", "Week", "Volume_raw", "Volume_norm", "WoW", "WoW_norm", "Zscore"]
    csv_bytes = view.sort_values(["WoW_norm", "Zscore"], ascending=[False, False])[exp_cols].to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered data (CSV)", data=csv_bytes, file_name="pinterest_trends_filtered.csv", mime="text/csv")

    st.caption("Tip: host on Streamlit Cloud as a public app. Upload CSVs via the sidebar to stay file-based.")

if __name__ == "__main__":
    main()
