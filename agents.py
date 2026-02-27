"""
agents.py
─────────
Three-agent pipeline that mirrors a CrewAI-style orchestration but uses the
Anthropic SDK directly for maximum transparency and control.

Agent 1 – MarketAnalysisAgent  (pure Python / pandas)
    Computes technical indicators for every ETF up to the current sim date.

Agent 2 – TradeProposalAgent   (Claude LLM + rule-based fallback)
    Reads the indicators + portfolio state and proposes trades as structured JSON.

Agent 3 – RiskAgent            (pure Python)
    Enforces the hard risk rules before any trade is sent for execution:
      • Max 40 % concentration per position
      • Max 2 trades per day
"""

import json
import logging
import re

import anthropic
import numpy as np
import pandas as pd

from config import (
    API_KEY,
    MODEL,
    MAX_POSITION_PCT,
    MAX_TRADES_PER_DAY,
    MOMENTUM_WINDOW,
    MEAN_REV_WINDOW,
    RSI_WINDOW,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Agent 1 – Market Analysis
# ═══════════════════════════════════════════════════════════════════════════════

class MarketAnalysisAgent:
    """
    Pure-Python agent that computes a standard set of technical indicators
    for each ETF in the universe, using only price data up to *current_date*.
    """

    def analyze(
        self,
        prices_df: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> dict[str, dict]:
        """
        Parameters
        ----------
        prices_df  : DataFrame indexed by date, columns = ETF tickers
        current_date: the 'today' of the simulation (inclusive)

        Returns
        -------
        {
            ticker: {
                "price":         float,
                "momentum_20d":  float,   # % return over 20 trading days
                "momentum_5d":   float,   # % return over 5 trading days
                "zscore":        float,   # (price - mean_20) / std_20
                "rsi":           float,   # 14-day RSI
                "above_sma20":   bool,    # price > 20-day SMA
            }
        }
        """
        history = prices_df[prices_df.index <= current_date]

        if len(history) < max(MOMENTUM_WINDOW, RSI_WINDOW) + 2:
            raise ValueError(
                f"Not enough history before {current_date.date()} "
                f"(need {max(MOMENTUM_WINDOW, RSI_WINDOW) + 2} rows)."
            )

        result: dict[str, dict] = {}
        for ticker in prices_df.columns:
            series = history[ticker].dropna()
            if len(series) < MOMENTUM_WINDOW + 1:
                continue

            price = float(series.iloc[-1])

            # ── Momentum ──────────────────────────────────────────────────
            mom_20d = (price / float(series.iloc[-MOMENTUM_WINDOW]) - 1.0) * 100.0
            mom_5d  = (price / float(series.iloc[-5]) - 1.0) * 100.0 if len(series) >= 5 else 0.0

            # ── Mean-reversion z-score ────────────────────────────────────
            window = series.iloc[-MEAN_REV_WINDOW:]
            mu, std = window.mean(), window.std()
            zscore = float((price - mu) / (std + 1e-9))

            # ── RSI (Wilder smoothing) ────────────────────────────────────
            deltas = series.iloc[-(RSI_WINDOW + 1):].diff().dropna()
            gains  = deltas.clip(lower=0)
            losses = (-deltas.clip(upper=0))
            avg_gain = gains.mean()
            avg_loss = losses.mean()
            rs  = avg_gain / (avg_loss + 1e-9)
            rsi = float(100.0 - 100.0 / (1.0 + rs))

            # ── SMA filter ───────────────────────────────────────────────
            sma20 = float(series.iloc[-MOMENTUM_WINDOW:].mean())

            result[ticker] = {
                "price":        round(price, 2),
                "momentum_20d": round(mom_20d, 2),
                "momentum_5d":  round(mom_5d, 2),
                "zscore":       round(zscore, 3),
                "rsi":          round(rsi, 1),
                "above_sma20":  price > sma20,
            }

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Agent 2 – Trade Proposal (LLM + fallback)
# ═══════════════════════════════════════════════════════════════════════════════

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

class TradeProposalAgent:
    """
    Uses Claude to propose a short-list of trades based on market analysis
    and the current portfolio state.  Falls back to a deterministic rule-based
    strategy when the API is unavailable or returns malformed output.
    """

    def __init__(self, strategy: str):
        self.strategy  = strategy          # "Momentum" or "Mean Reversion"
        self.use_llm   = bool(API_KEY)
        self.reasoning = ""                # populated after each call

        if self.use_llm:
            self.client = anthropic.Anthropic(api_key=API_KEY)

    # ── Public interface ─────────────────────────────────────────────────────
    def propose_trades(
        self,
        analysis:     dict[str, dict],
        portfolio:    dict,
        current_date: str,
        day_number:   int,
    ) -> list[dict]:
        """
        Returns a list of raw trade proposals (before risk validation):
            [{"ticker": str, "action": "BUY"|"SELL",
              "pct_of_portfolio": float, "reasoning": str}]
        """
        if self.use_llm:
            return self._llm_propose(analysis, portfolio, current_date, day_number)
        return self._rule_propose(analysis, portfolio)

    # ── LLM branch ──────────────────────────────────────────────────────────
    def _llm_propose(
        self,
        analysis:     dict[str, dict],
        portfolio:    dict,
        current_date: str,
        day_number:   int,
    ) -> list[dict]:
        prompt = self._build_prompt(analysis, portfolio, current_date, day_number)
        try:
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            self.reasoning = raw
            return self._parse_json(raw)

        except Exception as exc:
            logger.warning("LLM call failed (%s) – using rule-based fallback.", exc)
            self.reasoning = f"[LLM error – fallback] {exc}"
            return self._rule_propose(analysis, portfolio)

    def _build_prompt(
        self,
        analysis:     dict[str, dict],
        portfolio:    dict,
        current_date: str,
        day_number:   int,
    ) -> str:
        total  = portfolio["total_value"]
        cash   = portfolio["cash"]
        cash_p = cash / total * 100.0

        # Current positions table
        pos_lines = []
        for t, p in portfolio.get("positions", {}).items():
            w   = p["value"] / total * 100.0
            pnl = p["pnl_pct"]
            pos_lines.append(f"  {t:4s}: {w:5.1f}%  (P&L {pnl:+.1f}%)")
        pos_block = "\n".join(pos_lines) if pos_lines else "  (no positions – fully in cash)"

        # Analysis table
        rows = []
        for t, m in analysis.items():
            rows.append(
                f"  {t:4s}: ${m['price']:8.2f}  "
                f"mom20={m['momentum_20d']:+6.2f}%  "
                f"mom5={m['momentum_5d']:+5.2f}%  "
                f"z={m['zscore']:+5.2f}  "
                f"rsi={m['rsi']:5.1f}  "
                f"{'above' if m['above_sma20'] else 'below'} SMA20"
            )
        analysis_block = "\n".join(rows)

        guidance = {
            "Momentum": (
                "BUY the 1-2 ETFs with the strongest positive 20-day momentum AND rsi < 70. "
                "SELL any held ETFs whose 20-day momentum turned negative or rsi > 75."
            ),
            "Mean Reversion": (
                "BUY 1-2 ETFs with z-score < -1.0 (oversold) AND rsi < 45. "
                "SELL held ETFs with z-score > +1.0 (overbought) OR rsi > 65."
            ),
        }.get(self.strategy, "")

        return f"""You are a quantitative portfolio manager running a paper-trading simulation.

DATE: {current_date}  (Day {day_number} of 5)
STRATEGY: {self.strategy}
TOTAL PORTFOLIO VALUE: ${total:,.2f}
CASH AVAILABLE: ${cash:,.2f}  ({cash_p:.1f}%)

CURRENT POSITIONS:
{pos_block}

MARKET ANALYSIS (data up to today):
{analysis_block}

STRATEGY GUIDANCE:
{guidance}

HARD CONSTRAINTS (enforced by the Risk Agent downstream):
• Max position size: 40% of total portfolio
• Max 3 proposals (only 2 will be executed)
• Only BUY if cash is available; only SELL positions we actually hold

TASK:
Propose 0–3 trades. Return ONLY a valid JSON array — no prose, no markdown, no comments.
Each element must have exactly these keys:
  "ticker"            – one of: SPY QQQ IWM EFA EEM GLD TLT VNQ XLE XLF
  "action"            – "BUY" or "SELL"
  "pct_of_portfolio"  – integer 1–40 (% of total portfolio value to trade)
  "reasoning"         – one concise sentence

Return an empty array [] if no trade is warranted today.

Example:
[
  {{"ticker": "QQQ", "action": "BUY",  "pct_of_portfolio": 25, "reasoning": "Strongest momentum, RSI not overbought."}},
  {{"ticker": "TLT", "action": "SELL", "pct_of_portfolio": 12, "reasoning": "Negative momentum, reduce duration."}}
]"""

    @staticmethod
    def _parse_json(text: str) -> list[dict]:
        """Extract and parse the first JSON array from LLM output."""
        # Strip markdown code fences if present
        m = _JSON_BLOCK.search(text)
        if m:
            text = m.group(1).strip()

        # Try direct parse
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        # Fallback: find first [...] block
        start = text.find("[")
        end   = text.rfind("]") + 1
        if start != -1 and end > start:
            try:
                data = json.loads(text[start:end])
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass

        return []

    # ── Rule-based fallback ──────────────────────────────────────────────────
    def _rule_propose(self, analysis: dict[str, dict], portfolio: dict) -> list[dict]:
        proposals: list[dict] = []
        total     = portfolio["total_value"]
        cash      = portfolio["cash"]
        positions = portfolio.get("positions", {})

        if self.strategy == "Momentum":
            ranked = sorted(
                analysis.items(),
                key=lambda kv: kv[1]["momentum_20d"],
                reverse=True,
            )
            # BUY top 2 if not already held and cash is available
            for ticker, m in ranked[:3]:
                if len(proposals) >= 3:
                    break
                if ticker not in positions and cash > total * 0.15 and m["momentum_20d"] > 1.0:
                    proposals.append({
                        "ticker":           ticker,
                        "action":           "BUY",
                        "pct_of_portfolio": 20,
                        "reasoning":        f"Rule-based Momentum: {m['momentum_20d']:+.1f}% 20d return.",
                    })
            # SELL laggards
            for ticker, m in ranked[-3:]:
                if len(proposals) >= 3:
                    break
                if ticker in positions and m["momentum_20d"] < -2.0:
                    proposals.append({
                        "ticker":           ticker,
                        "action":           "SELL",
                        "pct_of_portfolio": round(positions[ticker]["value"] / total * 100),
                        "reasoning":        f"Rule-based Momentum: {m['momentum_20d']:+.1f}% 20d return.",
                    })

        else:  # Mean Reversion
            for ticker, m in sorted(analysis.items(), key=lambda kv: kv[1]["zscore"]):
                if len(proposals) >= 3:
                    break
                if m["zscore"] < -1.1 and ticker not in positions and cash > total * 0.15:
                    proposals.append({
                        "ticker":           ticker,
                        "action":           "BUY",
                        "pct_of_portfolio": 20,
                        "reasoning":        f"Rule-based MeanRev: z={m['zscore']:+.2f} (oversold).",
                    })
            for ticker, m in sorted(analysis.items(), key=lambda kv: kv[1]["zscore"], reverse=True):
                if len(proposals) >= 3:
                    break
                if m["zscore"] > 1.1 and ticker in positions:
                    proposals.append({
                        "ticker":           ticker,
                        "action":           "SELL",
                        "pct_of_portfolio": round(positions[ticker]["value"] / total * 100),
                        "reasoning":        f"Rule-based MeanRev: z={m['zscore']:+.2f} (overbought).",
                    })

        self.reasoning = (
            f"[Rule-based {self.strategy}] Proposed {len(proposals)} trade(s)."
        )
        return proposals


# ═══════════════════════════════════════════════════════════════════════════════
# Agent 3 – Risk Validation
# ═══════════════════════════════════════════════════════════════════════════════

class RiskAgent:
    """
    Stateless validator that enforces portfolio-level risk rules.

    Rules
    -----
    1. Maximum 2 trades executed per day  (MAX_TRADES_PER_DAY)
    2. Maximum 40 % weight per position   (MAX_POSITION_PCT)

    Trades that would breach a rule are either trimmed (to the allowed limit)
    or rejected outright.  The `violations` attribute lists every decision.
    """

    def __init__(self):
        self.violations: list[str] = []

    def validate(
        self,
        proposed:  list[dict],
        portfolio: dict,
        prices:    dict[str, float],
    ) -> list[dict]:
        """
        Returns approved (and potentially trimmed) trade list.
        Populates self.violations with human-readable messages.
        """
        self.violations = []
        approved:  list[dict] = []
        total      = portfolio["total_value"]
        cash_avail = portfolio["cash"]

        # Shadow-track intra-day changes so we can validate sequential trades
        pending_positions: dict[str, dict] = {
            t: dict(v) for t, v in portfolio.get("positions", {}).items()
        }

        for trade in proposed:
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

            if price is None:
                self.violations.append(f"REJECTED {ticker}: not in price data.")
                continue
            if action not in ("BUY", "SELL"):
                self.violations.append(f"REJECTED {ticker}: unknown action '{action}'.")
                continue

            trade_value = total * (pct / 100.0)

            # ── BUY logic ─────────────────────────────────────────────────
            if action == "BUY":
                # Cash check
                if trade_value > cash_avail * 0.99:
                    trade_value = cash_avail * 0.95   # use up to 95 % of remaining cash
                    pct         = trade_value / total * 100.0
                    if pct < 1.0:
                        self.violations.append(
                            f"REJECTED {ticker} BUY: insufficient cash "
                            f"(${cash_avail:,.0f} available)."
                        )
                        continue

                # Concentration check
                current_value = pending_positions.get(ticker, {}).get("value", 0.0)
                if current_value + trade_value > total * MAX_POSITION_PCT:
                    allowed_value = total * MAX_POSITION_PCT - current_value
                    if allowed_value < total * 0.01:
                        self.violations.append(
                            f"REJECTED {ticker} BUY: already at {MAX_POSITION_PCT*100:.0f}% limit."
                        )
                        continue
                    self.violations.append(
                        f"TRIMMED {ticker} BUY from {pct:.1f}% → "
                        f"{allowed_value/total*100:.1f}% to respect concentration limit."
                    )
                    trade_value = allowed_value
                    pct         = trade_value / total * 100.0

                qty = trade_value / price

                # Update shadow state
                if ticker in pending_positions:
                    prev      = pending_positions[ticker]
                    new_qty   = prev["quantity"] + qty
                    new_cost  = (prev["avg_cost"] * prev["quantity"] + price * qty) / new_qty
                    pending_positions[ticker] = {
                        "quantity":      new_qty,
                        "avg_cost":      new_cost,
                        "current_price": price,
                        "value":         new_qty * price,
                        "pnl_pct":       (price / new_cost - 1) * 100,
                    }
                else:
                    pending_positions[ticker] = {
                        "quantity":      qty,
                        "avg_cost":      price,
                        "current_price": price,
                        "value":         trade_value,
                        "pnl_pct":       0.0,
                    }
                cash_avail -= trade_value

            # ── SELL logic ────────────────────────────────────────────────
            elif action == "SELL":
                if ticker not in pending_positions:
                    self.violations.append(
                        f"REJECTED {ticker} SELL: no position held."
                    )
                    continue

                pos        = pending_positions[ticker]
                sell_value = min(trade_value, pos["value"])
                qty        = min(sell_value / price, pos["quantity"])
                trade_value = qty * price

                # Update shadow state
                new_qty = pos["quantity"] - qty
                if new_qty < 1e-6:
                    del pending_positions[ticker]
                else:
                    pending_positions[ticker]["quantity"] = new_qty
                    pending_positions[ticker]["value"]    = new_qty * price
                cash_avail += trade_value

            # ── Append approved trade with execution details ──────────────
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
