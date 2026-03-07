"""
market_data.py
──────────────
Fetches real S&P 500 constituents from Wikipedia and stock prices
via yfinance, then stores them in the local SQLite database.

Price history (close prices) is stored in the `prices` table of
whichever PortfolioDatabase instance is passed in:
  • simulation DB  – full history from 2019 + incremental live updates
  • backtest DB    – window snapshot (start-90 cal days → window end)

Two date concepts are tracked for sp500_stocks rows
────────────────────────────────────────────────────
• price_date  – the actual market date of the OHLCV quote returned by yfinance
                (e.g. last Friday if today is Monday / a holiday)
• fetch_date  – the calendar date we ran the fetch (today's date)
"""

import io
import logging

import pandas as pd
import requests
import yfinance as yf
from datetime import date

logger = logging.getLogger(__name__)

# yfinance logs "Failed download: ['XYZ']: TypeError(...)" at ERROR level for
# tickers that return None internally — a known yfinance bug that is harmless.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

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
        sym = tickers[0]
        clean = raw.dropna(how="all")
        if not clean.empty:
            r          = clean.iloc[-1]
            price_date = clean.index[-1].strftime("%Y-%m-%d")
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
        if not isinstance(raw.columns, pd.MultiIndex):
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


def fetch_and_store_prices(
    db,
    start: str | None = None,
    period: str = "60d",
    batch_size: int = 50,
) -> pd.DataFrame:
    """
    Fetch S&P 500 close prices from Yahoo Finance and upsert into db.prices.

    Downloads in batches of `batch_size` tickers to keep peak memory low —
    each batch is upserted to SQLite and discarded before the next is fetched.

    `start` (e.g. "2024-01-01") fetches history from that date.
    `period` (e.g. "90d") is used when no start date is given.

    Existing rows are upserted (INSERT OR REPLACE), so incremental calls
    are safe.  Returns the full wide-format close-price DataFrame from DB.

    Raises ValueError if yfinance returns no data for any batch.
    """
    constituents = get_sp500_constituents()
    tickers = constituents["symbol"].tolist()

    total_stored = 0
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i: i + batch_size]
        try:
            if start is not None:
                raw = yf.download(
                    batch, start=start, interval="1d",
                    auto_adjust=True, progress=False, threads=False,
                )
            else:
                raw = yf.download(
                    batch, period=period, interval="1d",
                    auto_adjust=True, progress=False, threads=False,
                )
        except Exception as exc:
            logger.warning("fetch_and_store_prices: batch %d–%d failed (%s), skipping.",
                           i, i + batch_size, exc)
            continue

        if raw.empty:
            logger.warning("fetch_and_store_prices: empty response for batch %d–%d.", i, i + batch_size)
            continue

        close_df = raw["Close"].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
        close_df = close_df.dropna(how="all")
        if close_df.empty:
            continue

        min_rows = int(len(close_df) * 0.80)
        close_df = close_df.dropna(axis=1, thresh=min_rows)
        close_df = close_df.dropna(how="all")
        close_df.index.name = "Date"

        db.upsert_prices(close_df)
        total_stored += len(close_df.columns)
        del raw, close_df   # free memory before next batch

    if total_stored == 0:
        raise ValueError("yfinance returned no price data for any S&P 500 tickers.")

    logger.info("fetch_and_store_prices: stored prices for %d tickers.", total_stored)
    return db.load_prices()


def refresh_latest_prices(db) -> pd.DataFrame:
    """
    Append any trading days newer than the last date already in db.prices.
    Falls back to a 90-day fetch if the table is empty.

    Call this before each simulation step so prices are always current.
    Returns the full updated wide-format DataFrame.
    """
    _, max_dt = db.get_prices_date_range()

    if max_dt is None:
        return fetch_and_store_prices(db, period="90d")

    tickers = db.get_prices_symbols()
    if not tickers:
        return fetch_and_store_prices(db, period="90d")

    raw = yf.download(
        tickers, period="5d", interval="1d",
        auto_adjust=True, progress=False, threads=True,
    )
    if raw.empty:
        logger.warning("refresh_latest_prices: yfinance returned no data.")
        return db.load_prices()

    close_df  = raw["Close"].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
    overlap   = [c for c in tickers if c in close_df.columns]
    close_df  = close_df[overlap]
    new_dates = close_df.index[close_df.index > max_dt]

    if new_dates.empty:
        return db.load_prices()

    db.upsert_prices(close_df.loc[new_dates])
    return db.load_prices()


def fetch_and_store_sp500(db) -> pd.DataFrame:
    """
    Fetches S&P 500 constituents + latest OHLCV prices, merges them,
    stores in db.sp500_stocks, and returns the merged DataFrame.
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
