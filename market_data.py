"""
market_data.py
──────────────
Fetches real S&P 500 constituents from Wikipedia and stock prices
via yfinance, then stores them in the local SQLite database.

Two date concepts are tracked for sp500_stocks rows
────────────────────────────────────────────────────
• price_date  – the actual market date of the OHLCV quote returned by yfinance
                (e.g. last Friday if today is Monday / a holiday)
• fetch_date  – the calendar date we ran the fetch (today's date)
"""

import io

import pandas as pd
import requests
import yfinance as yf
from datetime import date

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def get_sp500_constituents() -> pd.DataFrame:
    """
    Scrapes the S&P 500 list from Wikipedia (with browser-like headers to avoid 403).
    Returns DataFrame with columns: symbol, name, sector, industry.
    """
    resp = requests.get(SP500_WIKI_URL, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text), header=0)
    tbl = tables[0][["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].copy()
    tbl.columns = ["symbol", "name", "sector", "industry"]
    # yfinance uses '-' instead of '.' (e.g. BRK.B → BRK-B)
    tbl["symbol"] = tbl["symbol"].str.replace(".", "-", regex=False)
    return tbl.reset_index(drop=True)


def fetch_today_prices(tickers: list[str]) -> pd.DataFrame:
    """
    Downloads the latest available daily OHLCV data for a list of tickers.
    Uses a 5-day window so we always get data even when the market is closed.

    Returns DataFrame with columns:
        symbol, open, high, low, close, volume, price_date

    price_date is the actual market trading date of the returned data —
    which may be earlier than today if today is a weekend / holiday.
    """
    if not tickers:
        return pd.DataFrame()

    raw = yf.download(
        tickers,
        period="5d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if raw.empty:
        return pd.DataFrame()

    rows = []

    if len(tickers) == 1:
        # Single ticker: flat column structure (Open, High, Low, Close, Volume)
        sym = tickers[0]
        clean = raw.dropna(how="all")
        if not clean.empty:
            r           = clean.iloc[-1]
            price_date  = clean.index[-1].strftime("%Y-%m-%d")
            rows.append({
                "symbol":     sym,
                "price_date": price_date,
                "open":       _safe_float(r.get("Open")),
                "high":       _safe_float(r.get("High")),
                "low":        _safe_float(r.get("Low")),
                "close":      _safe_float(r.get("Close")),
                "volume":     _safe_int(r.get("Volume")),
            })
    else:
        # Multiple tickers: MultiIndex columns (price_type, ticker)
        if not isinstance(raw.columns, pd.MultiIndex):
            # Fallback for unexpected single-level structure
            return pd.DataFrame()

        try:
            close_df  = raw["Close"]
            open_df   = raw["Open"]
            high_df   = raw["High"]
            low_df    = raw["Low"]
            volume_df = raw["Volume"]
        except KeyError:
            return pd.DataFrame()

        for sym in tickers:
            if sym not in close_df.columns:
                continue
            series = close_df[sym].dropna()
            if series.empty:
                continue
            last_idx   = series.index[-1]
            price_date = last_idx.strftime("%Y-%m-%d")
            rows.append({
                "symbol":     sym,
                "price_date": price_date,
                "open":       _safe_float(open_df[sym].get(last_idx)),
                "high":       _safe_float(high_df[sym].get(last_idx)),
                "low":        _safe_float(low_df[sym].get(last_idx)),
                "close":      _safe_float(close_df[sym].get(last_idx)),
                "volume":     _safe_int(volume_df[sym].get(last_idx)),
            })

    return pd.DataFrame(rows)


def fetch_sp500_prices_csv(csv_path: str, years: int = 2) -> pd.DataFrame:
    """
    Fetches the current S&P 500 constituent list from Wikipedia, then downloads
    `years` of daily adjusted close prices for every stock via yfinance.

    Tickers with more than 20 % missing rows are dropped (e.g. recent IPOs).
    The result is written to csv_path and returned as a DataFrame.

    Raises ValueError if yfinance returns no data at all.
    """
    from pathlib import Path

    # Step 1: get the current constituent list
    constituents = get_sp500_constituents()
    tickers = constituents["symbol"].tolist()

    # Step 2: bulk-download historical closes
    raw = yf.download(
        tickers,
        period=f"{years}y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if raw.empty:
        raise ValueError("yfinance returned no price data for S&P 500 tickers.")

    # Extract close prices — bulk download always returns MultiIndex
    if isinstance(raw.columns, pd.MultiIndex):
        close_df = raw["Close"].copy()
    else:
        close_df = raw.copy()

    # Drop rows that are entirely NaN, then drop tickers with >20 % missing
    close_df = close_df.dropna(how="all")
    min_rows  = int(len(close_df) * 0.80)
    close_df  = close_df.dropna(axis=1, thresh=min_rows)
    close_df  = close_df.dropna(how="all")
    close_df.index.name = "Date"

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    close_df.to_csv(csv_path)

    return close_df


def fetch_and_store_sp500(db) -> pd.DataFrame:
    """
    Main entry point: fetches S&P 500 constituents + latest prices,
    merges them, stores in DB, and returns the merged DataFrame.

    The 'price_date' column records the actual market date of the OHLCV data
    (may differ from fetch_date on weekends / market holidays).
    """
    constituents = get_sp500_constituents()
    tickers = constituents["symbol"].tolist()
    prices = fetch_today_prices(tickers)

    today = date.today().isoformat()

    if prices.empty:
        merged = constituents.copy()
        for col in ("open", "high", "low", "close", "volume", "price_date"):
            merged[col] = None
    else:
        merged = constituents.merge(prices, on="symbol", how="left")

    merged["fetch_date"] = today
    db.upsert_sp500_stocks(merged)
    return merged


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_float(val) -> float | None:
    try:
        f = float(val)
        return round(f, 4) if not pd.isna(f) else None
    except (TypeError, ValueError):
        return None


def _safe_int(val) -> int | None:
    try:
        f = float(val)
        return int(f) if not pd.isna(f) else None
    except (TypeError, ValueError):
        return None
