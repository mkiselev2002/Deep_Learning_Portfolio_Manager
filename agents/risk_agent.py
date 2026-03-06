"""
agents/risk_agent.py
─────────────────────
RiskAgent — validates proposed trades against portfolio-level risk constraints.

Responsibilities
────────────────
  1. Receives the list of proposed trades from StrategyAgent.
  2. Checks each trade against all active risk constraints.
  3. Trims oversized positions to the allowed limit where possible.
  4. Rejects trades that cannot be made compliant.
  5. Returns the approved (and potentially trimmed) trade list, plus a log of
     every decision made.

─────────────────────────────────────────────────────────────────────────────
  RISK CONSTRAINTS
  ── Awaiting configuration from user ──
  Add the actual constraint logic inside `_check_constraints()` once the
  rules are provided.  Each constraint should append a message to
  `self.violations` and return False if the trade must be rejected.
─────────────────────────────────────────────────────────────────────────────

Output
──────
  approved_trades : list[dict]  – trades cleared for execution
  self.violations : list[str]   – human-readable log of all decisions
"""

import logging

from config import MAX_POSITION_PCT, MAX_TRADES_PER_DAY

logger = logging.getLogger(__name__)


class RiskAgent:
    """
    Stateless validator.  A fresh instance is created each simulation day
    so `self.violations` starts empty every day.
    """

    def __init__(self):
        self.violations: list[str] = []

    # ── Public interface ──────────────────────────────────────────────────────

    def validate(
        self,
        proposed:  list[dict],
        portfolio: dict,
        prices:    dict[str, float],
    ) -> list[dict]:
        """
        Parameters
        ----------
        proposed  : trade proposals from StrategyAgent
        portfolio : current portfolio state from PortfolioDatabase
        prices    : {ticker: current_price} dict

        Returns
        -------
        List of approved (and potentially trimmed) trades.
        """
        self.violations = []
        approved: list[dict] = []

        total      = portfolio["total_value"]
        cash_avail = portfolio["cash"]

        # Shadow-track intra-day position changes so sequential trades
        # are validated correctly against each other.
        pending: dict[str, dict] = {
            t: dict(v) for t, v in portfolio.get("positions", {}).items()
        }

        for trade in proposed:
            # ── Built-in hard limit: max trades per day ────────────────────
            if len(approved) >= MAX_TRADES_PER_DAY:
                self.violations.append(
                    f"REJECTED {trade.get('ticker')} {trade.get('action')}: "
                    f"max {MAX_TRADES_PER_DAY} trades/day reached."
                )
                continue

            ticker = trade.get("ticker", "")
            action = trade.get("action", "")
            pct    = min(float(trade.get("pct_of_portfolio", 20)), MAX_POSITION_PCT * 100)
            price  = prices.get(ticker)

            # ── Basic sanity checks ───────────────────────────────────────
            if price is None:
                self.violations.append(f"REJECTED {ticker}: price not available.")
                continue
            if action not in ("BUY", "SELL"):
                self.violations.append(f"REJECTED {ticker}: unknown action '{action}'.")
                continue

            trade_value = total * (pct / 100.0)

            # ─────────────────────────────────────────────────────────────
            # ADDITIONAL RISK CONSTRAINTS
            # ── Insert user-defined constraint checks here.
            # ── Each check should call self.violations.append(msg) and
            #    `continue` (or set a flag) to reject the trade.
            # ─────────────────────────────────────────────────────────────

            # ── BUY checks ────────────────────────────────────────────────
            if action == "BUY":
                if trade_value > cash_avail * 0.99:
                    trade_value = cash_avail * 0.95
                    pct         = trade_value / total * 100.0
                    if pct < 1.0:
                        self.violations.append(
                            f"REJECTED {ticker} BUY: insufficient cash "
                            f"(${cash_avail:,.0f} available)."
                        )
                        continue

                current_val = pending.get(ticker, {}).get("value", 0.0)
                if current_val + trade_value > total * MAX_POSITION_PCT:
                    allowed = total * MAX_POSITION_PCT - current_val
                    if allowed < total * 0.01:
                        self.violations.append(
                            f"REJECTED {ticker} BUY: already at "
                            f"{MAX_POSITION_PCT*100:.0f}% concentration limit."
                        )
                        continue
                    self.violations.append(
                        f"TRIMMED {ticker} BUY from {pct:.1f}% → "
                        f"{allowed/total*100:.1f}% to respect concentration limit."
                    )
                    trade_value = allowed
                    pct         = trade_value / total * 100.0

                qty = trade_value / price

                # Update shadow state
                if ticker in pending:
                    prev     = pending[ticker]
                    new_qty  = prev["shares"] + qty
                    new_cost = (prev["average_cost"] * prev["shares"] + price * qty) / new_qty
                    pending[ticker] = {
                        "shares":        new_qty,
                        "average_cost":  new_cost,
                        "current_price": price,
                        "value":         new_qty * price,
                        "pnl_pct":       (price / new_cost - 1) * 100,
                    }
                else:
                    pending[ticker] = {
                        "shares":        qty,
                        "average_cost":  price,
                        "current_price": price,
                        "value":         trade_value,
                        "pnl_pct":       0.0,
                    }
                cash_avail -= trade_value

            # ── SELL checks ───────────────────────────────────────────────
            elif action == "SELL":
                if ticker not in pending:
                    self.violations.append(
                        f"REJECTED {ticker} SELL: no position held."
                    )
                    continue

                pos         = pending[ticker]
                sell_value  = min(trade_value, pos["value"])
                qty         = min(sell_value / price, pos["shares"])
                trade_value = qty * price

                new_qty = pos["shares"] - qty
                if new_qty < 1e-6:
                    del pending[ticker]
                else:
                    pending[ticker]["shares"] = new_qty
                    pending[ticker]["value"]  = new_qty * price
                cash_avail += trade_value

            # ── Approved — attach execution details ───────────────────────
            approved.append({
                "ticker":           ticker,
                "action":           action,
                "pct_of_portfolio": round(pct, 2),
                "quantity":         round(qty, 6),
                "price":            round(price, 4),
                "amount":           round(trade_value, 2),
                "reasoning":        trade.get("reasoning", ""),
            })

        return approved
