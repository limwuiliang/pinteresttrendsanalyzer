Pinterest Growing Trends: Insightboard
======================================

A Streamlit dashboard for exploring and visualizing Pinterest “Growing Trends” CSV exports, comparing multiple markets, and tying rising search trends to current news and events — all in one interactive view.


Overview
--------
Pinterest Growing Trends: Insightboard lets you:
- Upload one or more Pinterest “Growing Trends” CSVs.
- Visualize normalized trend data over time using interactive charts and a weekly heatmap.
- Compare growth patterns across markets, regions, or categories.
- Instantly search related news headlines to understand what’s driving each spike.
- Export filtered and annotated data for use in reports or slides.


How to Use (First-Time Setup)
-----------------------------
1. Go to https://trends.pinterest.com/
2. Under “Trends Type”, choose “Growing Trends”.
3. Under “Region”, select the market or country you’re interested in. (Pinterest currently exports one region at a time.)
4. Click “Export CSV” to download the file.
5. Open your CSV in Excel or Google Sheets and add a new column called “Market”, filling it with the country code or region name (e.g., US, JP, AU, IN, BR, etc.).
6. To compare multiple regions:
   - Repeat steps 2–5 for each region.
   - Copy and paste all rows into a single sheet with the same column headers.
7. Save your combined file as a .csv and upload it to the app.


Features
--------
- Compare Trends: Select multiple trend keywords and compare their normalized growth curves over time.
- Heatmap View: Quickly see trend intensity by week and market, color-coded from low to high volume.
- Markets Filter: Automatically detects all “Market” values in your CSV and selects them all by default.
- Date Range Filter: Defaults to the full date span present in your data.
- News / Event Context: Always visible — enter a trend keyword and optional market to fetch related news headlines from Google News RSS.
- CSV Export: Download the currently filtered data as pinterest_trends_filtered.csv for external analysis.


Running the App
---------------
Local:
1. Install dependencies:
   pip install -r requirements.txt
2. Run Streamlit:
   streamlit run app.py
3. Open your browser at http://localhost:8501

Streamlit Cloud (Public):
1. Push these files to a public GitHub repository:
   - app.py
   - requirements.txt
   - README.txt (optional)
2. Go to https://share.streamlit.io → “Deploy
