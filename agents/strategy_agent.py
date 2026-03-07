"""
agents/strategy_agent.py
─────────────────────────
StrategyAgent — daily rebalance: liquidate → high-vol losers.

Strategy (executed every time "Next Day" is clicked)
─────────────────────────────────────────────────────
  1. SELL all currently held positions (full liquidation to cash).
  2. From the S&P 500 universe, rank every ticker by 5-day realised
     volatility (annualised std-dev of daily log returns).
  3. Keep the top-50 highest-volatility tickers.
  4. From that shortlist, pick the 3–5 with the largest percentage LOSS
     on the previous trading day. Claude uses RSI and 20-day momentum
     to decide exactly how many to buy (3 = high conviction, 5 = spread).
  5. Buy equal amounts across the chosen picks
     (e.g. 33 % each for 3, 25 % for 4, 20 % for 5).

Primary execution path
──────────────────────
  Claude API (tool use) — receives a formatted table of the top-50
  volatile stocks, reasons about the picks, then calls the
  `submit_trade_proposals` tool with the final trade list.

Fallback execution path
───────────────────────
  If the API key is absent or the LLM call raises any exception,
  StrategyAgent falls back to the deterministic Python implementation
  of the same rules and sets `self.api_failed = True`.

Output format (each proposed trade)
────────────────────────────────────
  {
      "ticker":           str,   # symbol
      "action":           "BUY" | "SELL",
      "pct_of_portfolio": float, # % of total portfolio value to trade
      "reasoning":        str,   # one-line explanation
  }
"""

import logging

import anthropic

from config import API_KEY, MODEL

logger = logging.getLogger(__name__)


class StrategyAgent:
    """
    Volatility-reversion strategy agent.

    Attempts to use Claude (tool-use) to select and justify trades.
    Falls back to the deterministic implementation if the API is
    unavailable or raises an exception.

    Attributes
    ----------
    api_failed : bool   – True when the LLM call failed this turn
    api_error  : str    – human-readable error message (empty on success)
    reasoning  : str    – one-line summary of what the agent decided
    """

    def __init__(self, strategy: str):
        self.strategy   = strategy
        self.reasoning  = ""
        self.api_failed = False
        self.api_error  = ""

    # ── Public interface ──────────────────────────────────────────────────────

    def propose_trades(
        self,
        analysis:     dict[str, dict],
        portfolio:    dict,
        current_date: str,
        day_number:   int,
        feedback:     list[str] | None = None,
    ) -> list[dict]:
        """
        Returns a list of raw trade proposals (before risk validation).

        Stock selection is always deterministic (top-50 vol → worst prev-day
        losers, skipping already-held tickers) so backtests are reproducible.
        The LLM is no longer used for stock selection.

        feedback : list of violation strings from a prior RiskAgent run.
        """
        self.api_failed = False
        self.api_error  = ""
        _retry = bool(feedback)

        proposals = self._build_proposals(analysis, portfolio, feedback=feedback)
        n_sell = sum(1 for p in proposals if p["action"] == "SELL")
        n_buy  = sum(1 for p in proposals if p["action"] == "BUY")
        tag = " [REVISED]" if _retry else ""
        self.reasoning = (
            f"[Algo{tag}] Liquidating {n_sell} position(s); "
            f"buying {n_buy} highest-vol S&P 500 losers (RSI-gated 3–5 picks, "
            f"skipping already-held tickers)."
        )
        return proposals

    # ── LLM path (Claude tool use) ────────────────────────────────────────────

    def _llm_proposals(self, analysis: dict, portfolio: dict, api_key: str, model: str,
                       feedback: list[str] | None = None) -> list[dict]:
        """
        Ask Claude to propose trades via the `submit_trade_proposals` tool.

        Claude is given:
          • Current portfolio state (positions, cash, total value)
          • Top-50 S&P 500 stocks ranked by 5-day realised volatility,
            with all key indicators, sorted by prev-day loss (biggest first)

        It must call `submit_trade_proposals` exactly once with the full
        trade list (SELLs first, then BUYs).
        """
        client    = anthropic.Anthropic(api_key=api_key)
        total     = portfolio["total_value"]
        cash      = portfolio["cash"]
        positions = portfolio.get("positions", {})

        # ── Build top-50 universe sorted by prev-day loss ─────────────────
        vol_sorted = sorted(
            analysis.items(),
            key=lambda kv: kv[1].get("realized_vol_5d", 0.0),
            reverse=True,
        )[:50]

        candidates = sorted(
            vol_sorted,
            key=lambda kv: kv[1].get("prev_day_return", 0.0),   # most negative first
        )

        # ── Format context strings ────────────────────────────────────────
        pos_lines = "\n".join(
            f"  {sym:6s}  {pos['shares']:.2f} sh @ ${pos['average_cost']:.2f} avg  "
            f"mkt=${pos['value']:,.0f}  unrealised={pos['pnl_pct']:+.1f}%"
            for sym, pos in positions.items()
        ) or "  (none — fully in cash)"

        universe_lines = "\n".join(
            f"  {sym:6s}  5d_vol={m['realized_vol_5d']:.1%}  "
            f"prev_day={m['prev_day_return']:+.2f}%  "
            f"rsi={m.get('rsi', 0):.0f}  "
            f"mom_20d={m.get('momentum_20d', 0):+.1f}%  "
            f"price=${m['price']:.2f}"
            for sym, m in candidates
        )

        system_prompt = (
            "You are an algorithmic trading agent executing a high-volatility "
            "mean-reversion strategy on S&P 500 stocks.\n\n"
            "STRATEGY RULES (follow exactly)\n"
            "──────────────────────────────\n"
            "1. SELL every currently held position in full (liquidation before rebalance).\n"
            "2. You are given the top-50 S&P 500 stocks ranked by 5-day realised "
            "volatility — the most violently moving stocks in the market right now.\n"
            "3. From that high-vol universe, identify the 3 to 5 stocks with the "
            "largest previous-day percentage LOSS. These are your mean-reversion "
            "buy candidates — yesterday's biggest losers in the most volatile names.\n"
            "4. Decide exactly how many to buy (3, 4, or 5) based on conviction:\n"
            "   • 3 picks → high conviction (e.g. extreme RSI oversold + deep loss + "
            "strong prior momentum). Allocate ~33 % each.\n"
            "   • 4 picks → moderate conviction. Allocate ~25 % each.\n"
            "   • 5 picks → broad spread when multiple candidates look similar. "
            "Allocate 20 % each.\n"
            "5. BUY equal allocations across your chosen picks (percentages must "
            "sum to 100 % of portfolio value).\n\n"
            "Think step-by-step:\n"
            "  a) List the SELLs needed to liquidate current positions.\n"
            "  b) Scan the universe table for the worst prev_day losers.\n"
            "  c) Check RSI and 20d momentum — do they confirm an oversold bounce "
            "thesis? Use that signal to decide 3, 4, or 5 picks.\n"
            "  d) State your pick count and per-stock allocation percentage.\n"
            "  e) Call `submit_trade_proposals` with your complete trade list."
        )

        feedback_block = ""
        if feedback:
            feedback_block = (
                "\n\nRISK AGENT FEEDBACK — YOUR PREVIOUS PROPOSALS WERE REJECTED\n"
                "──────────────────────────────────────────────────────────────\n"
                + "\n".join(f"  • {v}" for v in feedback)
                + "\n\nYou MUST avoid the tickers/actions flagged above and propose "
                "alternative picks that satisfy all constraints (cash ≥ 0, "
                "max position ≤ 40 %, max 2 trades/day)."
            )

        user_msg = (
            f"TODAY'S REBALANCE\n"
            f"─────────────────\n"
            f"Portfolio value: ${total:,.0f}   Cash available: ${cash:,.0f}\n\n"
            f"CURRENT POSITIONS (all to be liquidated)\n"
            f"─────────────────────────────────────────\n"
            f"{pos_lines}\n\n"
            f"TOP-50 S&P 500 STOCKS BY 5-DAY REALISED VOLATILITY\n"
            f"(sorted by previous-day return — most negative first)\n"
            f"────────────────────────────────────────────────────\n"
            f"{universe_lines}"
            f"{feedback_block}\n\n"
            "Step through the rules: identify the 3–5 worst losers from this "
            "high-vol universe, decide how many to buy based on signal conviction, "
            "then call `submit_trade_proposals` with your full trade list."
        )

        submit_tool = {
            "name": "submit_trade_proposals",
            "description": (
                "Submit the final, complete list of trade proposals "
                "(SELLs and BUYs) to the risk-management and execution agents. "
                "Call this exactly once."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "trades": {
                        "type": "array",
                        "description": "All trade proposals for today's rebalance",
                        "items": {
                            "type": "object",
                            "properties": {
                                "ticker": {
                                    "type":        "string",
                                    "description": "Stock ticker symbol (e.g. AAPL)",
                                },
                                "action": {
                                    "type":        "string",
                                    "enum":        ["BUY", "SELL"],
                                    "description": "Trade direction",
                                },
                                "pct_of_portfolio": {
                                    "type":        "number",
                                    "description": (
                                        "Percentage of total portfolio value. "
                                        "For SELLs: use current position weight. "
                                        "For BUYs: equal split across picks — "
                                        "~33 for 3 picks, 25 for 4, 20 for 5. "
                                        "All BUY pct_of_portfolio values must sum to 100."
                                    ),
                                },
                                "reasoning": {
                                    "type":        "string",
                                    "description": (
                                        "One-sentence rationale referencing "
                                        "vol, prev-day loss, RSI, or momentum"
                                    ),
                                },
                            },
                            "required": [
                                "ticker", "action", "pct_of_portfolio", "reasoning"
                            ],
                        },
                    }
                },
                "required": ["trades"],
            },
        }

        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=system_prompt,
            tools=[submit_tool],
            tool_choice={"type": "tool", "name": "submit_trade_proposals"},
            messages=[{"role": "user", "content": user_msg}],
        )

        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_trade_proposals":
                return block.input["trades"]

        raise ValueError(
            "Claude did not call submit_trade_proposals — "
            f"stop_reason={response.stop_reason}"
        )

    # ── Deterministic fallback ────────────────────────────────────────────────

    def _build_proposals(
        self,
        analysis:  dict[str, dict],
        portfolio: dict,
        feedback:  list[str] | None = None,
    ) -> list[dict]:
        proposals: list[dict] = []
        total     = portfolio["total_value"]
        positions = portfolio.get("positions", {})

        # Step 1: liquidate every current position
        for ticker, pos in positions.items():
            pct = max(1, round(pos["value"] / total * 100))
            proposals.append({
                "ticker":           ticker,
                "action":           "SELL",
                "pct_of_portfolio": pct,
                "reasoning":        "Full liquidation before daily rebalance.",
            })

        # Extract tickers previously rejected by the Risk Agent so we skip them
        rejected_tickers: set[str] = set()
        if feedback:
            import re as _re
            for msg in feedback:
                m = _re.match(r"REJECTED (\w+)", msg)
                if m:
                    rejected_tickers.add(m.group(1))

        # Tickers currently held — do not re-enter a position just liquidated.
        held_tickers: set[str] = set(positions.keys())

        # Step 2: rank S&P 500 universe by 5-day realised volatility
        vol_ranked = sorted(
            (
                (k, v) for k, v in analysis.items()
                if k not in rejected_tickers and k not in held_tickers
            ),
            key=lambda kv: kv[1].get("realized_vol_5d", 0.0),
            reverse=True,
        )
        top_50 = vol_ranked[:50]

        # Step 3: from top-50, pick the 3–5 biggest previous-day losers.
        # Decide count by RSI: deeply oversold (RSI < 30) → fewer, higher-conviction
        # picks; more moderate → broader spread.
        loss_ranked = sorted(
            top_50,
            key=lambda kv: kv[1].get("prev_day_return", 0.0),
        )
        candidates = loss_ranked[:5]   # always consider at most 5

        if not candidates:
            logger.warning("StrategyAgent: no picks found — check analysis data.")
            return proposals

        # Count picks: if ≥2 of the top-5 have RSI < 30 → use 3 picks (high
        # conviction on the most oversold); otherwise spread across 4 or 5.
        oversold_count = sum(
            1 for _, m in candidates if m.get("rsi", 50) < 30
        )
        if oversold_count >= 2:
            n_picks = 3
        elif oversold_count == 1:
            n_picks = 4
        else:
            n_picks = 5

        picks = candidates[:n_picks]
        alloc_pct = round(100 / n_picks)   # equal weight

        # Step 4: equal-weight BUY
        for ticker, m in picks:
            proposals.append({
                "ticker":           ticker,
                "action":           "BUY",
                "pct_of_portfolio": alloc_pct,
                "reasoning": (
                    f"5d vol {m['realized_vol_5d']:.1%} (top-50 universe); "
                    f"prev-day {m['prev_day_return']:+.2f}% loss; "
                    f"RSI {m.get('rsi', 0):.0f}. "
                    f"Buying {n_picks} picks at {alloc_pct}% each."
                ),
            })

        return proposals
