"""
database.py
───────────
SQLite-backed portfolio store.

Schema
──────
simulation      – single-row: game state
positions       – one row per held ETF ticker
transactions    – append-only trade log
daily_snapshots – daily snapshot of total equity for plotting
"""

import sqlite3
from contextlib import contextmanager
from typing import Any
import pandas as pd

from config import DB_PATH, INITIAL_CAPITAL


class PortfolioDatabase:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_schema()

    # ── Connection helper ────────────────────────────────────────────────
    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── Schema creation ──────────────────────────────────────────────────
    def _init_schema(self):
        with self._conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS simulation (
                    id                    INTEGER PRIMARY KEY CHECK (id = 1),
                    current_date          TEXT    NOT NULL DEFAULT '',
                    cash_balance          REAL    NOT NULL,
                    total_portfolio_value REAL    NOT NULL DEFAULT 0,
                    last_advance_date     TEXT    NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS positions (
                    symbol       TEXT PRIMARY KEY,
                    shares       INTEGER NOT NULL,
                    average_cost REAL    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS transactions (
                    id     INTEGER PRIMARY KEY AUTOINCREMENT,
                    date   TEXT    NOT NULL,
                    action TEXT    NOT NULL,
                    symbol TEXT    NOT NULL,
                    shares INTEGER NOT NULL,
                    price  REAL    NOT NULL,
                    reason TEXT    NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS daily_snapshots (
                    date         TEXT PRIMARY KEY,
                    total_equity REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sp500_stocks (
                    symbol     TEXT NOT NULL,
                    name       TEXT,
                    sector     TEXT,
                    industry   TEXT,
                    open       REAL,
                    high       REAL,
                    low        REAL,
                    close      REAL,
                    volume     INTEGER,
                    fetch_date TEXT NOT NULL,
                    PRIMARY KEY (symbol, fetch_date)
                );
            """)

        # ── Migration: add columns that didn't exist in older DB files ────
        with self._conn() as c:
            try:
                c.execute(
                    "ALTER TABLE simulation ADD COLUMN last_advance_date TEXT NOT NULL DEFAULT ''"
                )
            except Exception:
                pass  # column already exists — nothing to do

    # ── Reset ────────────────────────────────────────────────────────────
    def reset(self, initial_capital: float = INITIAL_CAPITAL):
        """Resets portfolio state only (keeps sp500_stocks cache intact)."""
        with self._conn() as c:
            c.execute("DELETE FROM simulation")
            c.execute("DELETE FROM positions")
            c.execute("DELETE FROM transactions")
            c.execute("DELETE FROM daily_snapshots")
            c.execute(
                "INSERT INTO simulation (id, cash_balance, total_portfolio_value, last_advance_date)"
                " VALUES (1, ?, ?, '')",
                (initial_capital, initial_capital),
            )

    def reset_all(self, initial_capital: float = INITIAL_CAPITAL):
        """Full wipe — clears every table including sp500_stocks, then seeds simulation row."""
        with self._conn() as c:
            c.execute("DELETE FROM simulation")
            c.execute("DELETE FROM positions")
            c.execute("DELETE FROM transactions")
            c.execute("DELETE FROM daily_snapshots")
            c.execute("DELETE FROM sp500_stocks")
            c.execute(
                "INSERT INTO simulation (id, cash_balance, total_portfolio_value, last_advance_date)"
                " VALUES (1, ?, ?, '')",
                (initial_capital, initial_capital),
            )

    # ── Reads ────────────────────────────────────────────────────────────
    def get_cash(self) -> float:
        with self._conn() as c:
            row = c.execute("SELECT cash_balance FROM simulation WHERE id = 1").fetchone()
            return float(row["cash_balance"]) if row else INITIAL_CAPITAL

    def get_raw_positions(self) -> dict[str, dict]:
        """Return {symbol: {shares, average_cost}}."""
        with self._conn() as c:
            rows = c.execute("SELECT symbol, shares, average_cost FROM positions").fetchall()
            return {
                r["symbol"]: {"shares": r["shares"], "average_cost": r["average_cost"]}
                for r in rows
            }

    def get_portfolio_state(self, prices: dict[str, float]) -> dict[str, Any]:
        cash = self.get_cash()
        raw = self.get_raw_positions()

        positions: dict[str, Any] = {}
        positions_value = 0.0

        for symbol, pos in raw.items():
            price = prices.get(symbol)
            if price is None:
                continue
            value = pos["shares"] * price
            pnl_pct = (price / pos["average_cost"] - 1.0) * 100.0 if pos["average_cost"] > 0 else 0.0
            positions[symbol] = {
                "shares":        pos["shares"],
                "average_cost":  pos["average_cost"],
                "current_price": price,
                "value":         value,
                "pnl_pct":       pnl_pct,
            }
            positions_value += value

        return {
            "cash":             cash,
            "positions":        positions,
            "positions_value":  positions_value,
            "total_value":      cash + positions_value,
        }

    # ── Writes ───────────────────────────────────────────────────────────
    def execute_trade(self, trade: dict, date: str):
        """Apply a validated trade dict to simulation + positions tables and log it."""
        symbol = trade.get("ticker", trade.get("symbol", ""))
        action = trade["action"]
        shares = int(trade.get("quantity", trade.get("shares", 0)))
        price  = float(trade["price"])
        amount = shares * price
        reason = str(trade.get("reasoning", trade.get("reason", "")))

        with self._conn() as c:
            if action == "BUY":
                c.execute(
                    "UPDATE simulation SET cash_balance = cash_balance - ?, current_date = ? WHERE id = 1",
                    (amount, date),
                )
                existing = c.execute(
                    "SELECT shares, average_cost FROM positions WHERE symbol = ?", (symbol,)
                ).fetchone()
                if existing:
                    new_shares = existing["shares"] + shares
                    new_cost   = (existing["average_cost"] * existing["shares"] + price * shares) / new_shares
                    c.execute(
                        "UPDATE positions SET shares = ?, average_cost = ? WHERE symbol = ?",
                        (new_shares, new_cost, symbol),
                    )
                else:
                    c.execute(
                        "INSERT INTO positions (symbol, shares, average_cost) VALUES (?, ?, ?)",
                        (symbol, shares, price),
                    )

            elif action == "SELL":
                c.execute(
                    "UPDATE simulation SET cash_balance = cash_balance + ?, current_date = ? WHERE id = 1",
                    (amount, date),
                )
                existing = c.execute(
                    "SELECT shares FROM positions WHERE symbol = ?", (symbol,)
                ).fetchone()
                if existing:
                    new_shares = existing["shares"] - shares
                    if new_shares <= 0:
                        c.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
                    else:
                        c.execute(
                            "UPDATE positions SET shares = ? WHERE symbol = ?",
                            (new_shares, symbol),
                        )

            c.execute(
                "INSERT INTO transactions (date, action, symbol, shares, price, reason)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (date, action, symbol, shares, price, reason),
            )

    def record_portfolio_value(self, date: str, total_equity: float):
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO daily_snapshots (date, total_equity) VALUES (?, ?)",
                (date, total_equity),
            )
            c.execute(
                "UPDATE simulation SET total_portfolio_value = ? WHERE id = 1",
                (total_equity,),
            )

    # ── Bulk reads (for display) ─────────────────────────────────────────
    def get_portfolio_history(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql("SELECT * FROM daily_snapshots ORDER BY date", c)

    def get_trades(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql("SELECT * FROM transactions ORDER BY date, id", c)

    # ── Advance-date gate (prevents multiple advances on the same real day) ──
    def get_last_advance_date(self) -> str:
        """Returns the real calendar date when 'Deal Next Hand' was last pressed, or ''."""
        with self._conn() as c:
            row = c.execute(
                "SELECT last_advance_date FROM simulation WHERE id = 1"
            ).fetchone()
            return row["last_advance_date"] if row else ""

    def set_last_advance_date(self, date_str: str):
        """Persist the real calendar date of the last advance."""
        with self._conn() as c:
            c.execute(
                "UPDATE simulation SET last_advance_date = ? WHERE id = 1",
                (date_str,),
            )

    # ── S&P 500 stock data ───────────────────────────────────────────────────
    def upsert_sp500_stocks(self, df: pd.DataFrame):
        """Insert or replace rows in sp500_stocks from a DataFrame."""
        rows = [
            (
                str(r.get("symbol", "")),
                r.get("name"),
                r.get("sector"),
                r.get("industry"),
                r.get("open"),
                r.get("high"),
                r.get("low"),
                r.get("close"),
                r.get("volume"),
                str(r.get("fetch_date", "")),
            )
            for r in df.to_dict(orient="records")
        ]
        with self._conn() as c:
            c.executemany(
                """INSERT OR REPLACE INTO sp500_stocks
                   (symbol, name, sector, industry, open, high, low, close, volume, fetch_date)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )

    def get_sp500_stocks(self, fetch_date: str | None = None) -> pd.DataFrame:
        """
        Returns the sp500_stocks table for a given date (defaults to most recent).
        """
        with self._conn() as c:
            if fetch_date:
                return pd.read_sql(
                    "SELECT * FROM sp500_stocks WHERE fetch_date = ? ORDER BY symbol",
                    c,
                    params=(fetch_date,),
                )
            # Most recent fetch_date available
            row = c.execute(
                "SELECT MAX(fetch_date) AS latest FROM sp500_stocks"
            ).fetchone()
            if not row or not row["latest"]:
                return pd.DataFrame()
            return pd.read_sql(
                "SELECT * FROM sp500_stocks WHERE fetch_date = ? ORDER BY symbol",
                c,
                params=(row["latest"],),
            )

    def get_sp500_fetch_dates(self) -> list[str]:
        """Returns all distinct fetch dates in sp500_stocks, newest first."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT DISTINCT fetch_date FROM sp500_stocks ORDER BY fetch_date DESC"
            ).fetchall()
            return [r["fetch_date"] for r in rows]
