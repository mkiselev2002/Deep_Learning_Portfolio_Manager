"""
database.py
───────────
SQLite-backed portfolio store.

Schema
──────
portfolio        – single-row: cash balance
positions        – one row per held ETF ticker
trades           – append-only trade log
portfolio_history– daily snapshot of total value
agent_log        – append-only agent reasoning log
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
                CREATE TABLE IF NOT EXISTS portfolio (
                    id           INTEGER PRIMARY KEY CHECK (id = 1),
                    cash         REAL    NOT NULL,
                    last_updated TEXT    NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS positions (
                    ticker   TEXT PRIMARY KEY,
                    quantity REAL NOT NULL,
                    avg_cost REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trades (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    date      TEXT    NOT NULL,
                    ticker    TEXT    NOT NULL,
                    action    TEXT    NOT NULL,
                    quantity  REAL    NOT NULL,
                    price     REAL    NOT NULL,
                    amount    REAL    NOT NULL,
                    reasoning TEXT    NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS portfolio_history (
                    date             TEXT PRIMARY KEY,
                    total_value      REAL NOT NULL,
                    cash             REAL NOT NULL,
                    positions_value  REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS agent_log (
                    id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    date    TEXT    NOT NULL,
                    agent   TEXT    NOT NULL,
                    content TEXT    NOT NULL
                );
            """)

    # ── Reset ────────────────────────────────────────────────────────────
    def reset(self, initial_capital: float = INITIAL_CAPITAL):
        with self._conn() as c:
            c.execute("DELETE FROM portfolio")
            c.execute("DELETE FROM positions")
            c.execute("DELETE FROM trades")
            c.execute("DELETE FROM portfolio_history")
            c.execute("DELETE FROM agent_log")
            c.execute(
                "INSERT INTO portfolio (id, cash) VALUES (1, ?)",
                (initial_capital,),
            )

    # ── Reads ────────────────────────────────────────────────────────────
    def get_cash(self) -> float:
        with self._conn() as c:
            row = c.execute("SELECT cash FROM portfolio WHERE id = 1").fetchone()
            return float(row["cash"]) if row else INITIAL_CAPITAL

    def get_raw_positions(self) -> dict[str, dict]:
        """Return {ticker: {quantity, avg_cost}}."""
        with self._conn() as c:
            rows = c.execute("SELECT ticker, quantity, avg_cost FROM positions").fetchall()
            return {r["ticker"]: {"quantity": r["quantity"], "avg_cost": r["avg_cost"]} for r in rows}

    def get_portfolio_state(self, prices: dict[str, float]) -> dict[str, Any]:
        """
        Compute full portfolio snapshot enriched with current prices.

        Returns
        -------
        {
            "cash": float,
            "positions": {
                ticker: {quantity, avg_cost, current_price, value, pnl_pct}
            },
            "positions_value": float,
            "total_value": float,
        }
        """
        cash = self.get_cash()
        raw = self.get_raw_positions()

        positions: dict[str, Any] = {}
        positions_value = 0.0

        for ticker, pos in raw.items():
            price = prices.get(ticker)
            if price is None:
                continue
            value = pos["quantity"] * price
            pnl_pct = (price / pos["avg_cost"] - 1.0) * 100.0 if pos["avg_cost"] > 0 else 0.0
            positions[ticker] = {
                "quantity":      pos["quantity"],
                "avg_cost":      pos["avg_cost"],
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
        """Apply a validated trade dict to cash + positions tables and log it."""
        ticker    = trade["ticker"]
        action    = trade["action"]
        qty       = float(trade["quantity"])
        price     = float(trade["price"])
        amount    = float(trade["amount"])
        reasoning = str(trade.get("reasoning", ""))

        with self._conn() as c:
            if action == "BUY":
                c.execute(
                    "UPDATE portfolio SET cash = cash - ?, last_updated = ? WHERE id = 1",
                    (amount, date),
                )
                existing = c.execute(
                    "SELECT quantity, avg_cost FROM positions WHERE ticker = ?", (ticker,)
                ).fetchone()
                if existing:
                    new_qty  = existing["quantity"] + qty
                    new_cost = (existing["avg_cost"] * existing["quantity"] + price * qty) / new_qty
                    c.execute(
                        "UPDATE positions SET quantity = ?, avg_cost = ? WHERE ticker = ?",
                        (new_qty, new_cost, ticker),
                    )
                else:
                    c.execute(
                        "INSERT INTO positions (ticker, quantity, avg_cost) VALUES (?, ?, ?)",
                        (ticker, qty, price),
                    )

            elif action == "SELL":
                c.execute(
                    "UPDATE portfolio SET cash = cash + ?, last_updated = ? WHERE id = 1",
                    (amount, date),
                )
                existing = c.execute(
                    "SELECT quantity FROM positions WHERE ticker = ?", (ticker,)
                ).fetchone()
                if existing:
                    new_qty = existing["quantity"] - qty
                    if new_qty < 1e-6:
                        c.execute("DELETE FROM positions WHERE ticker = ?", (ticker,))
                    else:
                        c.execute(
                            "UPDATE positions SET quantity = ? WHERE ticker = ?",
                            (new_qty, ticker),
                        )

            c.execute(
                "INSERT INTO trades (date, ticker, action, quantity, price, amount, reasoning)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (date, ticker, action, qty, price, amount, reasoning),
            )

    def record_portfolio_value(
        self,
        date: str,
        total_value: float,
        cash: float,
        positions_value: float,
    ):
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO portfolio_history"
                " (date, total_value, cash, positions_value)"
                " VALUES (?, ?, ?, ?)",
                (date, total_value, cash, positions_value),
            )

    def log_agent(self, date: str, agent: str, content: str):
        with self._conn() as c:
            c.execute(
                "INSERT INTO agent_log (date, agent, content) VALUES (?, ?, ?)",
                (date, agent, content),
            )

    # ── Bulk reads (for display) ─────────────────────────────────────────
    def get_portfolio_history(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql("SELECT * FROM portfolio_history ORDER BY date", c)

    def get_trades(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql("SELECT * FROM trades ORDER BY date, id", c)

    def get_agent_log(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql("SELECT * FROM agent_log ORDER BY id", c)
