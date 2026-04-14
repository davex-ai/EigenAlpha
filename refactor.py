import pandas as pd
import requests
from io import StringIO
import os
import time


def get_sp500_tickers():
    file_name = "sp500_tickers.csv"

    # Check if file exists AND if it's less than 30 days old
    if os.path.exists(file_name):
        file_age_days = (time.time() - os.path.getmtime(file_name)) / (60 * 60 * 24)

        if file_age_days < 30:
            print(f"--- Using local file (Updated {file_age_days:.1f} days ago) ---")
            return pd.read_csv(file_name)
        else:
            print("--- Local file is over 30 days old. Re-scraping... ---")
    else:
        print("--- Local file not found. Scraping Wikipedia... ---")

    # Scrape if file is missing or old
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        df = pd.read_html(StringIO(response.text), match="Symbol")[0]
        df.to_csv(file_name, index=False)
        print(f"--- Successfully updated and saved to {file_name} ---")
        return df
    else:
        print(f"Error reaching Wikipedia: {response.status_code}")
        return pd.read_csv(file_name) if os.path.exists(file_name) else None

def load_local_tickers(file_name="sp500_tickers.csv"):
    """Loads the S&P 500 ticker list from a local CSV file."""
    if os.path.exists(file_name):
        print(f"--- Loading data from {file_name} ---")
        df = pd.read_csv(file_name)
        return df['Symbol'].tolist()
    else:
        print(f"Error: {file_name} not found locally.")
        return []