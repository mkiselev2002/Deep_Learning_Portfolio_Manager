"""
agents/strategy_agent.py
─────────────────────────
StrategyAgent — selects trades based on configurable strategy rules.

Responsibilities
────────────────
  1. Receives the market analysis from DataAgent and the current portfolio state.
  2. Applies the active strategy's entry/exit rules to identify trade candidates.
  3. Returns a list of proposed trades for RiskAgent to validate.

─────────────────────────────────────────────────────────────────────────────
  STRATEGY RULES
  ── Awaiting configuration from user ──
  The `_rules_momentum` and `_rules_mean_reversion` methods below contain
  placeholder logic.  Replace the body of each method with the actual rules
  once they are provided.
─────────────────────────────────────────────────────────────────────────────

Output format (each proposed trade)
────────────────────────────────────
  {
      "ticker":           str,    # symbol
      "action":           "BUY" | "SELL",
      "pct_of_portfolio": float,  # % of total portfolio value to trade (1–40)
      "reasoning":        str,    # one-line explanation
  }
"""

import json
import logging
import re

import anthropic

from config import API_KEY, MODEL, MAX_POSITION_PCT, MAX_TRADES_PER_DAY

logger = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class StrategyAgent:
    """
    Selects trades via an LLM (Claude) with a rule-based fallback.

    Parameters
    ----------
    strategy : "Momentum" | "Mean Reversion"
    """

    def __init__(self, strategy: str):
        self.strategy  = strategy
        self.use_llm   = bool(API_KEY)
        self.reasoning = ""           # populated after each call

        if self.use_llm:
            self.client = anthropic.Anthropic(api_key=API_KEY)

    # ── Public interface ──────────────────────────────────────────────────────

    def propose_trades(
        self,
        analysis:     dict[str, dict],
        portfolio:    dict,
        current_date: str,
        day_number:   int,
    ) -> list[dict]:
        """
        Returns a list of raw trade proposals (before risk validation).

        Parameters
        ----------
        analysis     : output of DataAgent.analyze()
        portfolio    : output of PortfolioDatabase.get_portfolio_state()
        current_date : ISO date string "YYYY-MM-DD"
        day_number   : 1-based index within the simulation week
        """
        if self.use_llm:
            return self._llm_propose(analysis, portfolio, current_date, day_number)
        return self._rule_propose(analysis, portfolio)

    # ── LLM branch ───────────────────────────────────────────────────────────

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
            logger.warning("StrategyAgent LLM call failed (%s) — using rule-based fallback.", exc)
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

        pos_lines = []
        for t, p in portfolio.get("positions", {}).items():
            w   = p["value"] / total * 100.0
            pnl = p["pnl_pct"]
            pos_lines.append(f"  {t:6s}: {w:5.1f}%  (P&L {pnl:+.1f}%)")
        pos_block = "\n".join(pos_lines) if pos_lines else "  (no positions – fully in cash)"

        rows = []
        for t, m in analysis.items():
            rows.append(
                f"  {t:6s}: ${m['price']:8.2f}  "
                f"mom20={m['momentum_20d']:+6.2f}%  "
                f"mom5={m['momentum_5d']:+5.2f}%  "
                f"z={m['zscore']:+5.2f}  "
                f"rsi={m['rsi']:5.1f}  "
                f"{'above' if m['above_sma20'] else 'below'} SMA20"
            )
        analysis_block = "\n".join(rows)

        # ─────────────────────────────────────────────────────────────────────
        # STRATEGY GUIDANCE
        # ── Replace these strings with the actual rule descriptions once
        #    provided by the user.
        # ─────────────────────────────────────────────────────────────────────
        guidance = {
            "Momentum": (
                "BUY the 1-2 tickers with the strongest positive 20-day momentum AND rsi < 70. "
                "SELL any held tickers whose 20-day momentum turned negative or rsi > 75."
            ),
            "Mean Reversion": (
                "BUY 1-2 tickers with z-score < -1.0 (oversold) AND rsi < 45. "
                "SELL held tickers with z-score > +1.0 (overbought) OR rsi > 65."
            ),
        }.get(self.strategy, "")

        tickers_str = " ".join(analysis.keys())

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
• Max position size: {MAX_POSITION_PCT*100:.0f}% of total portfolio
• Max {MAX_TRADES_PER_DAY} proposals executed per day
• Only BUY if cash is available; only SELL positions we actually hold

TASK:
Propose 0–3 trades. Return ONLY a valid JSON array — no prose, no markdown, no comments.
Each element must have exactly these keys:
  "ticker"            – one of: {tickers_str}
  "action"            – "BUY" or "SELL"
  "pct_of_portfolio"  – integer 1–{int(MAX_POSITION_PCT*100)} (% of total portfolio value to trade)
  "reasoning"         – one concise sentence

Return an empty array [] if no trade is warranted today."""

    @staticmethod
    def _parse_json(text: str) -> list[dict]:
        m = _JSON_BLOCK.search(text)
        if m:
            text = m.group(1).strip()
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
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

    # ── Rule-based fallback ───────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # RULE IMPLEMENTATIONS
    # ── These methods will be replaced with the actual strategy rules
    #    once provided by the user.  Current logic is a temporary placeholder.
    # ─────────────────────────────────────────────────────────────────────────

    def _rule_propose(self, analysis: dict[str, dict], portfolio: dict) -> list[dict]:
        if self.strategy == "Momentum":
            proposals = self._rules_momentum(analysis, portfolio)
        else:
            proposals = self._rules_mean_reversion(analysis, portfolio)

        self.reasoning = (
            f"[Rule-based {self.strategy}] Proposed {len(proposals)} trade(s)."
        )
        return proposals

    def _rules_momentum(self, analysis: dict[str, dict], portfolio: dict) -> list[dict]:
        """
        MOMENTUM STRATEGY RULES
        ── Placeholder — replace with actual rules.
        """
        proposals: list[dict] = []
        total     = portfolio["total_value"]
        cash      = portfolio["cash"]
        positions = portfolio.get("positions", {})

        ranked = sorted(analysis.items(), key=lambda kv: kv[1]["momentum_20d"], reverse=True)

        for ticker, m in ranked[:3]:
            if len(proposals) >= 3:
                break
            if ticker not in positions and cash > total * 0.15 and m["momentum_20d"] > 1.0:
                proposals.append({
                    "ticker":           ticker,
                    "action":           "BUY",
                    "pct_of_portfolio": 20,
                    "reasoning":        f"Placeholder momentum rule: {m['momentum_20d']:+.1f}% 20d return.",
                })

        for ticker, m in ranked[-3:]:
            if len(proposals) >= 3:
                break
            if ticker in positions and m["momentum_20d"] < -2.0:
                proposals.append({
                    "ticker":           ticker,
                    "action":           "SELL",
                    "pct_of_portfolio": round(positions[ticker]["value"] / total * 100),
                    "reasoning":        f"Placeholder momentum rule: {m['momentum_20d']:+.1f}% 20d return.",
                })

        return proposals

    def _rules_mean_reversion(self, analysis: dict[str, dict], portfolio: dict) -> list[dict]:
        """
        MEAN REVERSION STRATEGY RULES
        ── Placeholder — replace with actual rules.
        """
        proposals: list[dict] = []
        total     = portfolio["total_value"]
        cash      = portfolio["cash"]
        positions = portfolio.get("positions", {})

        for ticker, m in sorted(analysis.items(), key=lambda kv: kv[1]["zscore"]):
            if len(proposals) >= 3:
                break
            if m["zscore"] < -1.1 and ticker not in positions and cash > total * 0.15:
                proposals.append({
                    "ticker":           ticker,
                    "action":           "BUY",
                    "pct_of_portfolio": 20,
                    "reasoning":        f"Placeholder mean-rev rule: z={m['zscore']:+.2f} (oversold).",
                })

        for ticker, m in sorted(analysis.items(), key=lambda kv: kv[1]["zscore"], reverse=True):
            if len(proposals) >= 3:
                break
            if m["zscore"] > 1.1 and ticker in positions:
                proposals.append({
                    "ticker":           ticker,
                    "action":           "SELL",
                    "pct_of_portfolio": round(positions[ticker]["value"] / total * 100),
                    "reasoning":        f"Placeholder mean-rev rule: z={m['zscore']:+.2f} (overbought).",
                })

        return proposals
