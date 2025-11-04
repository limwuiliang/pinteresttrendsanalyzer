
# Pinterest Growing Trends — APAC Insightboard (Streamlit)

An interactive dashboard for visualizing Pinterest Growing Trends weekly data and tying spikes to real-world events.

## Features
- CSV uploader (or bundled sample)
- Filters by market, keyword search, date range
- Heatmap of weekly normalized volumes
- Spike detection (WoW% + rolling z-score)
- Multi-series trend comparison
- **News/Event context** via Google News RSS (no API key)
- Manual tagging of spikes + CSV export

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud
1. Push `app.py` and `requirements.txt` to a GitHub repo.
2. Add your CSV in the repo or upload at runtime.
3. Create a new app in Streamlit Cloud and point to `app.py`.
4. Optional: open network access is required for Google News RSS.

## Data expectations
- Columns: `Trend`, `Market`, weekly columns with date-like headers (e.g., `7/25/25`, `8/1/25`, ...).
- Non-date metadata columns like `Weekly change`, `Monthly change` can be present.

## Event linkage
The app uses Google News RSS to surface likely related headlines for a selected keyword/market and date window. For higher precision, you can integrate NewsAPI, GDELT, or custom curated event lists per market.
