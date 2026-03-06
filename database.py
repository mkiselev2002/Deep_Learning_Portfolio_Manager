"""
database.py
───────────
SQLite-backed portfolio store.

Schema
──────
simulation       – single-row: game state
positions        – one row per currently-held ticker (live view)
position_history – full lifecycle record for every position (open + closed)
transactions     – append-only trade log (includes trade amount)
daily_snapshots  – daily snapshot of total equity for plotting
sp500_stocks     – S&P 500 stock cache with OHLCV + price_date / fetch_date
"""

import json
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

                CREATE TABLE IF NOT EXISTS position_history (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol           TEXT    NOT NULL,
                    opened_date      TEXT    NOT NULL,
                    closed_date      TEXT,
                    shares           REAL    NOT NULL DEFAULT 0,
                    avg_cost         REAL    NOT NULL DEFAULT 0,
                    close_price      REAL,
                    realized_pnl     REAL,
                    realized_pnl_pct REAL
                );

                CREATE TABLE IF NOT EXISTS transactions (
                    id     INTEGER PRIMARY KEY AUTOINCREMENT,
                    date   TEXT    NOT NULL,
                    action TEXT    NOT NULL,
                    symbol TEXT    NOT NULL,
                    shares INTEGER NOT NULL,
                    price  REAL    NOT NULL,
                    amount REAL    NOT NULL DEFAULT 0,
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
                    price_date TEXT,
                    PRIMARY KEY (symbol, fetch_date)
                );

                CREATE TABLE IF NOT EXISTS day_results (
                    day_num     INTEGER PRIMARY KEY,
                    date        TEXT    NOT NULL,
                    result_json TEXT    NOT NULL
                );
            """)

        # ── Migrations: safely add columns that didn't exist in older DBs ─
        _migrations = [
            "ALTER TABLE simulation    ADD COLUMN last_advance_date TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE transactions  ADD COLUMN amount REAL NOT NULL DEFAULT 0",
            "ALTER TABLE sp500_stocks  ADD COLUMN price_date TEXT",
        ]
        with self._conn() as c:
            for sql in _migrations:
                try:
                    c.execute(sql)
                except Exception:
                    pass  # column already exists — nothing to do

    # ── Reset ────────────────────────────────────────────────────────────
    def reset(self, initial_capital: float = INITIAL_CAPITAL):
        """Resets portfolio state only (keeps sp500_stocks cache intact)."""
        with self._conn() as c:
            c.execute("DELETE FROM simulation")
            c.execute("DELETE FROM positions")
            c.execute("DELETE FROM position_history")
            c.execute("DELETE FROM transactions")
            c.execute("DELETE FROM daily_snapshots")
            c.execute("DELETE FROM day_results")
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
            c.execute("DELETE FROM position_history")
            c.execute("DELETE FROM transactions")
            c.execute("DELETE FROM daily_snapshots")
            c.execute("DELETE FROM day_results")
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
        """
        Apply a validated trade dict to simulation + positions tables, log it,
        and maintain position_history (open / close dates, realized P&L).
        """
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
                    # ── Adding to an existing position ───────────────────
                    new_shares = existing["shares"] + shares
                    new_cost   = (existing["average_cost"] * existing["shares"] + price * shares) / new_shares
                    c.execute(
                        "UPDATE positions SET shares = ?, average_cost = ? WHERE symbol = ?",
                        (new_shares, new_cost, symbol),
                    )
                    # Update open position_history entry (blend cost/shares)
                    hist = c.execute(
                        "SELECT id, shares, avg_cost FROM position_history"
                        " WHERE symbol = ? AND closed_date IS NULL",
                        (symbol,),
                    ).fetchone()
                    if hist:
                        blended_shares = hist["shares"] + shares
                        blended_cost   = (hist["avg_cost"] * hist["shares"] + price * shares) / blended_shares
                        c.execute(
                            "UPDATE position_history SET shares = ?, avg_cost = ? WHERE id = ?",
                            (blended_shares, blended_cost, hist["id"]),
                        )
                    else:
                        # No open history row — create one (data integrity fallback)
                        c.execute(
                            "INSERT INTO position_history (symbol, opened_date, shares, avg_cost)"
                            " VALUES (?, ?, ?, ?)",
                            (symbol, date, new_shares, new_cost),
                        )
                else:
                    # ── Opening a brand-new position ─────────────────────
                    c.execute(
                        "INSERT INTO positions (symbol, shares, average_cost) VALUES (?, ?, ?)",
                        (symbol, shares, price),
                    )
                    c.execute(
                        "INSERT INTO position_history (symbol, opened_date, shares, avg_cost)"
                        " VALUES (?, ?, ?, ?)",
                        (symbol, date, shares, price),
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
                        # ── Full exit — close position ────────────────────
                        c.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
                        hist = c.execute(
                            "SELECT id, avg_cost, shares FROM position_history"
                            " WHERE symbol = ? AND closed_date IS NULL",
                            (symbol,),
                        ).fetchone()
                        if hist:
                            realized_pnl     = (price - hist["avg_cost"]) * hist["shares"]
                            realized_pnl_pct = (
                                (price / hist["avg_cost"] - 1.0) * 100.0
                                if hist["avg_cost"] > 0 else 0.0
                            )
                            c.execute(
                                "UPDATE position_history"
                                " SET closed_date = ?, close_price = ?,"
                                "     realized_pnl = ?, realized_pnl_pct = ?"
                                " WHERE id = ?",
                                (date, price,
                                 round(realized_pnl, 2), round(realized_pnl_pct, 2),
                                 hist["id"]),
                            )
                    else:
                        # ── Partial sell — reduce shares ──────────────────
                        c.execute(
                            "UPDATE positions SET shares = ? WHERE symbol = ?",
                            (new_shares, symbol),
                        )
                        hist = c.execute(
                            "SELECT id FROM position_history"
                            " WHERE symbol = ? AND closed_date IS NULL",
                            (symbol,),
                        ).fetchone()
                        if hist:
                            c.execute(
                                "UPDATE position_history SET shares = ? WHERE id = ?",
                                (new_shares, hist["id"]),
                            )

            # ── Log the transaction (with total amount) ───────────────────
            c.execute(
                "INSERT INTO transactions (date, action, symbol, shares, price, amount, reason)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (date, action, symbol, shares, price, round(amount, 2), reason),
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
            return pd.read_sql("SELECT * FROM transactions ORDER BY date DESC, id DESC", c)

    def get_position_history(self) -> pd.DataFrame:
        """
        Returns all position lifecycle records, newest-opened first.
        Open positions have closed_date = NULL.
        Closed positions have realized_pnl and realized_pnl_pct populated.
        """
        with self._conn() as c:
            return pd.read_sql(
                "SELECT * FROM position_history ORDER BY opened_date DESC, id DESC",
                c,
            )

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
        """Insert or replace rows in sp500_stocks from a DataFrame.

        Expects columns: symbol, name, sector, industry,
                         open, high, low, close, volume,
                         fetch_date  (when we pulled the data),
                         price_date  (the actual market date of the OHLCV quote).
        """
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
                str(r["price_date"]) if r.get("price_date") not in (None, "", "None", "nan") else None,
            )
            for r in df.to_dict(orient="records")
        ]
        with self._conn() as c:
            c.executemany(
                """INSERT OR REPLACE INTO sp500_stocks
                   (symbol, name, sector, industry, open, high, low, close, volume,
                    fetch_date, price_date)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )

    def get_sp500_stocks(self, fetch_date: str | None = None) -> pd.DataFrame:
        """
        Returns the sp500_stocks table for a given fetch_date (defaults to most recent).
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

    # ── Day-result persistence (for cross-session state restore) ─────────────

    def has_simulation(self) -> bool:
        """Returns True if a simulation row (id=1) exists in the DB."""
        with self._conn() as c:
            row = c.execute("SELECT id FROM simulation WHERE id = 1").fetchone()
            return row is not None

    def save_day_result(self, result: dict) -> None:
        """Persist a completed simulation day as a JSON blob."""
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO day_results (day_num, date, result_json)"
                " VALUES (?, ?, ?)",
                (result["day_num"], result["date"], json.dumps(result, default=str)),
            )

    def load_all_day_results(self) -> list[dict]:
        """Load all persisted daily results ordered by day number."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT result_json FROM day_results ORDER BY day_num"
            ).fetchall()
        return [json.loads(r["result_json"]) for r in rows]
