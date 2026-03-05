"""
agents/execution_agent.py
──────────────────────────
ExecutionAgent — places approved trades and persists all results to the DB.

Responsibilities
────────────────
  1. Receives the list of approved trades from RiskAgent.
  2. Executes each trade via PortfolioDatabase (deducts/adds cash, updates
     positions, appends to the transactions log).
  3. Queries the post-execution portfolio state and writes a daily equity
     snapshot to daily_snapshots.
  4. Returns a structured execution report for the UI layer.

Output (execute() return value)
────────────────────────────────
  {
      "date":            str,          # "YYYY-MM-DD"
      "executed_trades": list[dict],   # trades that were actually committed
      "portfolio":       dict,         # post-execution portfolio state
      "equity":          float,        # total portfolio value after trades
  }
"""

import logging

from database import PortfolioDatabase

logger = logging.getLogger(__name__)


class ExecutionAgent:
    """
    Stateless execution layer.  One instance can be reused across many days.

    Parameters
    ----------
    db : PortfolioDatabase
        The shared database instance for the simulation.
    """

    def __init__(self, db: PortfolioDatabase):
        self.db = db

    # ── Public interface ──────────────────────────────────────────────────────

    def execute(
        self,
        approved_trades: list[dict],
        prices:          dict[str, float],
        date_str:        str,
    ) -> dict:
        """
        Execute all approved trades for a single simulation day.

        Parameters
        ----------
        approved_trades : output of RiskAgent.validate()
        prices          : {ticker: price} for the current simulation date
        date_str        : ISO date string "YYYY-MM-DD"

        Returns
        -------
        Execution report dict (see module docstring).
        """
        executed: list[dict] = []

        for trade in approved_trades:
            try:
                self.db.execute_trade(trade, date_str)
                executed.append(trade)
                logger.info(
                    "Executed %s %s x%.4f @ $%.4f on %s",
                    trade["action"],
                    trade["ticker"],
                    trade["quantity"],
                    trade["price"],
                    date_str,
                )
            except Exception as exc:
                logger.error(
                    "Failed to execute %s %s: %s",
                    trade.get("action"),
                    trade.get("ticker"),
                    exc,
                )

        # Snapshot post-execution portfolio
        portfolio = self.db.get_portfolio_state(prices)
        equity    = portfolio["total_value"]
        self.db.record_portfolio_value(date_str, equity)

        return {
            "date":            date_str,
            "executed_trades": executed,
            "portfolio":       portfolio,
            "equity":          equity,
        }
