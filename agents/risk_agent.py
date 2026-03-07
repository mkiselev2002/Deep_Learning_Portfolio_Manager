"""
agents/risk_agent.py
─────────────────────
RiskAgent — validates proposed trades against portfolio-level risk constraints.

Responsibilities
────────────────
  1. Receives the list of proposed trades from StrategyAgent.
  2. Processes ALL SELL trades first, liquidating full positions regardless
     of the proposed pct_of_portfolio (ensures clean daily rebalance).
  3. For BUY trades: enforces max-5-stock limit and computes a perfectly
     equal dollar allocation across picks, using integer (whole-share) quantities.
  4. Returns the approved trade list plus a log of every decision made.

Key rules enforced here (not delegated to the LLM)
───────────────────────────────────────────────────
  • Full liquidation — every SELL always closes the entire position.
  • Max 5 stocks — BUY proposals beyond 5 are rejected.
  • Equal dollar split — all BUY allocations are recalculated here;
    the LLM's pct_of_portfolio is used only for stock selection ranking.
  • Whole shares only — quantities are math.floor(allocation / price).
  • Single concentration cap — no single position > MAX_POSITION_PCT of portfolio.

Output
──────
  approved_trades : list[dict]  – trades cleared for execution
  self.violations : list[str]   – human-readable log of all decisions
"""

import logging
import math

from config import MAX_POSITION_PCT, MAX_TRADES_PER_DAY

logger = logging.getLogger(__name__)

MAX_BUY_STOCKS = 5   # hard cap: never hold more than 5 positions


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

        # Shadow-track positions so sequential trades validate correctly.
        pending: dict[str, dict] = {
            t: dict(v) for t, v in portfolio.get("positions", {}).items()
        }

        # ── Pass 1: process ALL SELLs — always full position liquidation ──────
        sell_proposals = [t for t in proposed if t.get("action") == "SELL"]
        for trade in sell_proposals:
            if len(approved) >= MAX_TRADES_PER_DAY:
                self.violations.append(
                    f"REJECTED {trade.get('ticker')} SELL: "
                    f"max {MAX_TRADES_PER_DAY} trades/day reached."
                )
                continue

            ticker = trade.get("ticker", "")
            price  = prices.get(ticker)

            if price is None:
                self.violations.append(f"REJECTED {ticker}: price not available.")
                continue
            if ticker not in pending:
                self.violations.append(
                    f"REJECTED {ticker} SELL: no position held."
                )
                continue

            # Always sell the FULL position — ignore proposed pct entirely.
            pos         = pending[ticker]
            qty         = pos["shares"]          # sell every share
            trade_value = qty * price

            del pending[ticker]
            cash_avail += trade_value

            approved.append({
                "ticker":           ticker,
                "action":           "SELL",
                "pct_of_portfolio": round(trade_value / total * 100, 2),
                "quantity":         round(qty, 6),
                "price":            round(price, 4),
                "amount":           round(trade_value, 2),
                "reasoning":        trade.get("reasoning",
                                              "Full liquidation before daily rebalance."),
            })

        # ── Pass 2: collect BUY proposals, cap at 5, enforce equal split ─────
        buy_proposals = [t for t in proposed if t.get("action") == "BUY"]

        # De-duplicate tickers; skip tickers with no price.
        seen_tickers: set[str] = set()
        valid_buys: list[dict] = []
        for trade in buy_proposals:
            ticker = trade.get("ticker", "")
            if ticker in seen_tickers:
                continue
            if prices.get(ticker) is None:
                self.violations.append(
                    f"REJECTED {ticker} BUY: price not available."
                )
                continue
            seen_tickers.add(ticker)
            valid_buys.append(trade)

        # Hard cap: at most MAX_BUY_STOCKS picks.
        if len(valid_buys) > MAX_BUY_STOCKS:
            for t in valid_buys[MAX_BUY_STOCKS:]:
                self.violations.append(
                    f"REJECTED {t['ticker']} BUY: exceeds {MAX_BUY_STOCKS}-stock maximum."
                )
            valid_buys = valid_buys[:MAX_BUY_STOCKS]

        n_buys = len(valid_buys)
        if n_buys > 0 and cash_avail > 1.0:
            # Equal dollar allocation — leave a tiny 0.5 % cash buffer to avoid
            # rounding overspend after floor() on share counts.
            per_stock_cash = (cash_avail * 0.995) / n_buys

            for trade in valid_buys:
                if len(approved) >= MAX_TRADES_PER_DAY:
                    self.violations.append(
                        f"REJECTED {trade['ticker']} BUY: "
                        f"max {MAX_TRADES_PER_DAY} trades/day reached."
                    )
                    continue

                ticker = trade.get("ticker", "")
                price  = prices[ticker]

                # Whole shares only — floor to avoid going over cash.
                qty = math.floor(per_stock_cash / price)

                if qty < 1:
                    self.violations.append(
                        f"REJECTED {ticker} BUY: insufficient cash for "
                        f"1 share at ${price:.2f}."
                    )
                    continue

                trade_value = qty * price
                pct         = trade_value / total * 100.0

                # Concentration guard — trim if needed.
                current_val = pending.get(ticker, {}).get("value", 0.0)
                if current_val + trade_value > total * MAX_POSITION_PCT:
                    allowed_val = total * MAX_POSITION_PCT - current_val
                    allowed_qty = math.floor(allowed_val / price)
                    if allowed_qty < 1:
                        self.violations.append(
                            f"REJECTED {ticker} BUY: already at "
                            f"{MAX_POSITION_PCT*100:.0f}% concentration limit."
                        )
                        continue
                    self.violations.append(
                        f"TRIMMED {ticker} BUY: capped to "
                        f"{MAX_POSITION_PCT*100:.0f}% concentration limit."
                    )
                    qty         = allowed_qty
                    trade_value = qty * price
                    pct         = trade_value / total * 100.0

                # Update shadow state.
                pending[ticker] = {
                    "shares":        float(qty),
                    "average_cost":  price,
                    "current_price": price,
                    "value":         trade_value,
                    "pnl_pct":       0.0,
                }
                cash_avail -= trade_value

                approved.append({
                    "ticker":           ticker,
                    "action":           "BUY",
                    "pct_of_portfolio": round(pct, 2),
                    "quantity":         float(qty),
                    "price":            round(price, 4),
                    "amount":           round(trade_value, 2),
                    "reasoning":        trade.get("reasoning", ""),
                })

        return approved
