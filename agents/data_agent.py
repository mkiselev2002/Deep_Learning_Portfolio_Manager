"""
agents/data_agent.py
────────────────────
DataAgent — the "Analyst" in the pipeline.

Two responsibilities
────────────────────
  1. analyze()                 — technical indicators for simulation days
  2. generate_portfolio_report() — daily auto-briefing (runs on app launch)

Daily Portfolio Report
──────────────────────
  • Current positions with P&L
  • 5-day and all-time return vs initial capital
  • Identifies best / worst performers
  • Scrapes recent Yahoo Finance news headlines for held tickers
  • Uses Claude (or rule-based fallback) to produce a written narrative summary
"""

import json
import logging
from datetime import date, timedelta
from typing import TYPE_CHECKING

import anthropic
import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    API_KEY,
    MODEL,
    INITIAL_CAPITAL,
    MOMENTUM_WINDOW,
    MEAN_REV_WINDOW,
    RSI_WINDOW,
)

if TYPE_CHECKING:
    from database import PortfolioDatabase

logger = logging.getLogger(__name__)


class DataAgent:
    """
    Pure-Python analyst agent.

    Usage — technical indicators
    ----------------------------
    agent    = DataAgent()
    analysis = agent.analyze(prices_df, current_date)

    Usage — portfolio report (auto-run on app launch)
    -------------------------------------------------
    report = agent.generate_portfolio_report(db, prices_df)
    """

    # ═══════════════════════════════════════════════════════════════════════
    # 1.  Technical Indicators  (used by StrategyAgent during simulation)
    # ═══════════════════════════════════════════════════════════════════════

    def analyze(
        self,
        prices_df: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> dict[str, dict]:
        """
        Computes technical indicators for every ticker up to current_date.

        Returns
        -------
        {ticker: {price, momentum_20d, momentum_5d, zscore, rsi, above_sma20}}
        """
        history = prices_df[prices_df.index <= current_date]

        min_rows = max(MOMENTUM_WINDOW, RSI_WINDOW) + 2
        if len(history) < min_rows:
            raise ValueError(
                f"DataAgent: not enough price history before {current_date.date()} "
                f"(need {min_rows} rows, got {len(history)})."
            )

        result: dict[str, dict] = {}

        for ticker in prices_df.columns:
            series = history[ticker].dropna()
            if len(series) < MOMENTUM_WINDOW + 1:
                continue

            price = float(series.iloc[-1])

            # Momentum
            mom_20d = (price / float(series.iloc[-MOMENTUM_WINDOW]) - 1.0) * 100.0
            mom_5d  = (
                (price / float(series.iloc[-5]) - 1.0) * 100.0
                if len(series) >= 5 else 0.0
            )

            # Mean-reversion z-score
            window = series.iloc[-MEAN_REV_WINDOW:]
            mu, std = window.mean(), window.std()
            zscore = float((price - mu) / (std + 1e-9))

            # RSI (Wilder smoothing)
            deltas   = series.iloc[-(RSI_WINDOW + 1):].diff().dropna()
            gains    = deltas.clip(lower=0)
            losses   = (-deltas.clip(upper=0))
            avg_gain = gains.mean()
            avg_loss = losses.mean()
            rs  = avg_gain / (avg_loss + 1e-9)
            rsi = float(100.0 - 100.0 / (1.0 + rs))

            # SMA filter
            sma20 = float(series.iloc[-MOMENTUM_WINDOW:].mean())

            # 5-day realized volatility (annualized std dev of daily log returns)
            if len(series) >= 7:
                recent_6 = series.iloc[-6:]
                log_rets = np.log(recent_6 / recent_6.shift(1)).dropna()
                realized_vol_5d = float(log_rets.std() * np.sqrt(252))
            else:
                realized_vol_5d = 0.0

            # Previous trading day return
            prev_day_return = (
                (price / float(series.iloc[-2]) - 1.0) * 100.0
                if len(series) >= 2 else 0.0
            )

            result[ticker] = {
                "price":            round(price,           2),
                "momentum_20d":     round(mom_20d,         2),
                "momentum_5d":      round(mom_5d,          2),
                "zscore":           round(zscore,           3),
                "rsi":              round(rsi,              1),
                "above_sma20":      price > sma20,
                "realized_vol_5d":  round(realized_vol_5d, 4),
                "prev_day_return":  round(prev_day_return,  4),
            }

        return result

    # ═══════════════════════════════════════════════════════════════════════
    # 2.  Daily Portfolio Report  (auto-run on app launch)
    # ═══════════════════════════════════════════════════════════════════════

    def generate_portfolio_report(
        self,
        db: "PortfolioDatabase",
        prices_df: pd.DataFrame,
    ) -> dict:
        """
        Builds a comprehensive daily portfolio briefing.

        Returns a dict with:
          generated_date, portfolio_value, cash, positions,
          alltime_pnl, alltime_pnl_pct, fiveday_pnl, fiveday_pnl_pct,
          top_performers, underperformers, news, summary
        """
        today = date.today().isoformat()

        # Latest prices available in price history
        latest_prices = prices_df.iloc[-1].to_dict() if not prices_df.empty else {}
        portfolio     = db.get_portfolio_state(latest_prices)
        history_df    = db.get_portfolio_history()

        # ── Performance metrics ──────────────────────────────────────────
        metrics = self._compute_performance(portfolio, history_df)

        # ── Position detail ──────────────────────────────────────────────
        positions_detail = []
        for ticker, pos in portfolio["positions"].items():
            # Compute yesterday's price change from the price history CSV
            yesterday_chg = None
            if ticker in prices_df.columns:
                series = prices_df[ticker].dropna()
                if len(series) >= 2:
                    yesterday_chg = round(
                        (float(series.iloc[-1]) / float(series.iloc[-2]) - 1.0) * 100.0, 2
                    )
            positions_detail.append({
                "ticker":        ticker,
                "shares":        round(pos["shares"], 4),
                "avg_cost":      round(pos["average_cost"], 2),
                "current_price": round(pos["current_price"], 2),
                "value":         round(pos["value"], 2),
                "pnl_pct":       round(pos["pnl_pct"], 2),
                "pnl_dollar":    round(pos["value"] - pos["shares"] * pos["average_cost"], 2),
                "yesterday_chg": yesterday_chg,   # % move in latest session
            })

        positions_detail.sort(key=lambda x: x["pnl_pct"], reverse=True)
        top_performers  = [p for p in positions_detail if p["pnl_pct"] > 0][:3]
        underperformers = [p for p in positions_detail if p["pnl_pct"] < 0][:3]

        # ── News headlines — held tickers first, then broad market ───────
        held_tickers = [p["ticker"] for p in positions_detail]
        news = self._fetch_news(held_tickers) if held_tickers else {}

        # ── Build context dict ───────────────────────────────────────────
        context = {
            "generated_date":  today,
            "portfolio_value": round(portfolio["total_value"], 2),
            "cash":            round(portfolio["cash"], 2),
            "initial_capital": INITIAL_CAPITAL,
            "positions":       positions_detail,
            "top_performers":  top_performers,
            "underperformers": underperformers,
            "news_headlines":  news,
            **metrics,
        }

        # ── AI narrative summary ─────────────────────────────────────────
        context["summary"] = self._generate_summary(context)

        return context

    # ── Performance calculations ─────────────────────────────────────────────

    def _compute_performance(
        self,
        portfolio: dict,
        history_df: pd.DataFrame,
    ) -> dict:
        total_value = portfolio["total_value"]

        # All-time vs initial capital
        alltime_pnl     = total_value - INITIAL_CAPITAL
        alltime_pnl_pct = (alltime_pnl / INITIAL_CAPITAL) * 100.0 if INITIAL_CAPITAL > 0 else 0.0

        # 5-day: compare today vs value 5 snapshots ago
        fiveday_pnl     = 0.0
        fiveday_pnl_pct = 0.0
        if not history_df.empty and len(history_df) >= 2:
            lookback_idx   = max(0, len(history_df) - 6)   # up to 5 rows back
            base_value     = float(history_df["total_equity"].iloc[lookback_idx])
            fiveday_pnl     = total_value - base_value
            fiveday_pnl_pct = (fiveday_pnl / base_value) * 100.0 if base_value > 0 else 0.0

        # Day-over-day change
        daily_pnl     = 0.0
        daily_pnl_pct = 0.0
        if not history_df.empty and len(history_df) >= 2:
            prev_value    = float(history_df["total_equity"].iloc[-2])
            daily_pnl     = total_value - prev_value
            daily_pnl_pct = (daily_pnl / prev_value) * 100.0 if prev_value > 0 else 0.0

        # Win rate (% of days portfolio grew)
        win_rate = 0.0
        if not history_df.empty and len(history_df) > 1:
            changes  = history_df["total_equity"].diff().dropna()
            win_rate = (changes > 0).sum() / len(changes) * 100.0

        return {
            "alltime_pnl":      round(alltime_pnl,     2),
            "alltime_pnl_pct":  round(alltime_pnl_pct, 2),
            "fiveday_pnl":      round(fiveday_pnl,     2),
            "fiveday_pnl_pct":  round(fiveday_pnl_pct, 2),
            "daily_pnl":        round(daily_pnl,        2),
            "daily_pnl_pct":    round(daily_pnl_pct,    2),
            "win_rate":         round(win_rate,          1),
            "total_days":       len(history_df),
        }

    # ── News scraping ─────────────────────────────────────────────────────────

    def _fetch_news(self, tickers: list[str], max_per_ticker: int = 3) -> dict[str, list[str]]:
        """
        Fetches recent Yahoo Finance news headlines for up to 5 held tickers.
        Returns {ticker: [headline, ...]}
        """
        news: dict[str, list[str]] = {}
        for ticker in tickers[:5]:
            try:
                yf_ticker = yf.Ticker(ticker)
                articles  = yf_ticker.news or []
                headlines = []
                for a in articles[:max_per_ticker]:
                    # yfinance news items have different structures across versions
                    title = (
                        a.get("title")
                        or a.get("headline")
                        or (a.get("content", {}) or {}).get("title", "")
                    )
                    if title:
                        headlines.append(title)
                if headlines:
                    news[ticker] = headlines
            except Exception as exc:
                logger.debug("News fetch failed for %s: %s", ticker, exc)
        return news

    # ── AI narrative ──────────────────────────────────────────────────────────

    def _generate_summary(self, context: dict) -> str:
        import config as _cfg   # re-read at call time so a late-loaded key is picked up
        if _cfg.API_KEY:
            return self._llm_summary(context, _cfg.API_KEY, _cfg.MODEL)
        return self._rule_based_summary(context)

    def _llm_summary(self, context: dict, api_key: str, model: str) -> str:
        """Ask Claude to write the portfolio narrative."""

        # ── Per-position block with yesterday's move ──────────────────────
        pos_lines = []
        for p in context["positions"]:
            yday = (
                f"  ↕ yesterday {p['yesterday_chg']:+.1f}%"
                if p.get("yesterday_chg") is not None
                else ""
            )
            pos_lines.append(
                f"  {p['ticker']:6s}  value=${p['value']:>10,.0f}"
                f"  unrealised={p['pnl_pct']:+.1f}%"
                f"  (entry ${p['avg_cost']:.2f} → now ${p['current_price']:.2f})"
                f"{yday}"
            )
        positions_text = "\n".join(pos_lines) or "  (no open positions — fully in cash)"

        # ── News block: held tickers first, labelled ──────────────────────
        news_text = ""
        held = set(p["ticker"] for p in context["positions"])
        # Held-ticker news first
        for ticker, headlines in context.get("news_headlines", {}).items():
            if ticker not in held:
                continue
            news_text += f"\n  ★ {ticker} (HELD):\n"
            for h in headlines:
                news_text += f"      • {h}\n"
        # Then any other tickers (shouldn't exist given current fetch logic, but future-proof)
        for ticker, headlines in context.get("news_headlines", {}).items():
            if ticker in held:
                continue
            news_text += f"\n  {ticker}:\n"
            for h in headlines:
                news_text += f"      • {h}\n"
        if not news_text:
            news_text = "\n  (no headlines available for held positions)"

        # ── Strategy reminder ─────────────────────────────────────────────
        strategy_note = (
            "NOTE: This portfolio runs a high-volatility mean-reversion strategy. "
            "Every day it liquidates all positions and buys the 5 S&P 500 stocks "
            "with the highest 5-day realised volatility that fell the most the "
            "previous day. Gains come from overnight mean-reversion bounces in "
            "beaten-down, volatile names."
        )

        # Inject a Day-1 warning so Claude doesn't fabricate prior-day comparisons
        is_day_one = context.get("total_days", 0) <= 1
        day1_note  = (
            "\n⚠️ DAY 1 ALERT: This is the very first trading session — positions were "
            "just opened today. There is NO prior portfolio value to compare against. "
            "All P/L figures are zero or trivial noise; do NOT describe the portfolio "
            "as 'up' or 'down'. Do NOT reference 'yesterday's performance' of the book. "
            "Instead describe the mean-reversion thesis for the names just entered and "
            "what signals to watch for the first overnight hold.\n"
            if is_day_one else ""
        )

        prompt = f"""You are a sharp portfolio analyst writing a daily briefing for a paper trading desk.

{strategy_note}{day1_note}

PORTFOLIO SNAPSHOT  ({context['generated_date']})  —  Day {context['total_days']} of simulation
──────────────────────────────────────────────────────────────────────────────
Total Value:   ${context['portfolio_value']:,.2f}
Cash:          ${context['cash']:,.2f}
All-time P/L:  ${context['alltime_pnl']:+,.2f}  ({context['alltime_pnl_pct']:+.2f}%)
5-day P/L:     ${context['fiveday_pnl']:+,.2f}  ({context['fiveday_pnl_pct']:+.2f}%)
Yesterday P/L: ${context['daily_pnl']:+,.2f}  ({context['daily_pnl_pct']:+.2f}%)
Win Rate:      {context['win_rate']:.0f}%  ({context['total_days']} trading days)

CURRENT POSITIONS (with yesterday's price move)
────────────────────────────────────────────────
{positions_text}

NEWS HEADLINES FOR HELD POSITIONS (★ = currently held)
───────────────────────────────────────────────────────{news_text}

Write a concise 3-paragraph briefing. Strict formatting rules:
- NEVER use markdown headers (no #, ##, ###)
- NO bullet points or dashes
- NO backtick code spans (never wrap numbers or words in backticks)
- NO dollar signs ($) — write amounts as e.g. "USD 248,749" or "248k" to avoid rendering conflicts
- Each paragraph: 2–3 sentences maximum
- Start each paragraph with an inline bold label exactly as written below

**PERFORMANCE:** Are we up or down overall and yesterday? Name the biggest movers and their dollar impact on the book. (If Day 1, skip P/L comparison — describe the positions just opened.)

**DRIVERS:** What news or price action explains the moves in our held names? Connect specific events to specific P/L.

**STRATEGY:** Is mean-reversion working or failing today? One concrete thing to watch on the next trading day.

Analytical, high-stakes tone. Every number must come from the data above."""

        try:
            client   = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                max_tokens=450,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as exc:
            logger.warning("LLM summary failed (%s) — using fallback.", exc)
            context["_summary_error"] = str(exc)
            return self._rule_based_summary(context)

    def _rule_based_summary(self, context: dict) -> str:
        """Deterministic fallback summary when the LLM is unavailable."""
        pv    = context["portfolio_value"]
        pnl   = context["alltime_pnl"]
        ppct  = context["alltime_pnl_pct"]
        fpct  = context["fiveday_pnl_pct"]
        dpnl  = context["daily_pnl"]
        dpct  = context["daily_pnl_pct"]
        sign  = "up" if pnl >= 0 else "down"
        dsign = "gained" if dpnl >= 0 else "lost"

        winners = ", ".join(
            f"{p['ticker']} ({p['pnl_pct']:+.1f}%)"
            for p in context["top_performers"]
        ) or "none"
        losers = ", ".join(
            f"{p['ticker']} ({p['pnl_pct']:+.1f}%)"
            for p in context["underperformers"]
        ) or "none"

        # Per-position yesterday move summary
        moves = []
        for p in context["positions"]:
            if p.get("yesterday_chg") is not None:
                moves.append(f"{p['ticker']} {p['yesterday_chg']:+.1f}%")
        moves_text = (
            "Yesterday's moves for held names: " + ", ".join(moves) + "."
            if moves else ""
        )

        _err = context.get("_summary_error")
        if _err:
            api_note = f"⚠️ Claude API error: {_err}"
        elif API_KEY:
            api_note = "⚠️ AI narrative unavailable — rule-based fallback in use."
        else:
            api_note = "ℹ️ Connect a Claude API key for AI-generated market analysis."

        return (
            f"Portfolio valued at ${pv:,.2f}, {sign} "
            f"${abs(pnl):,.2f} ({ppct:+.2f}%) from the initial "
            f"${context['initial_capital']:,.0f} stake. "
            f"Yesterday the book {dsign} ${abs(dpnl):,.2f} ({dpct:+.2f}%), "
            f"and over the last 5 trading days it moved {fpct:+.2f}%. "
            f"Win rate: {context['win_rate']:.0f}% across "
            f"{context['total_days']} recorded day(s).\n\n"
            f"Top performers: {winners}. Underperformers: {losers}. "
            f"{moves_text}\n\n"
            f"{api_note}"
        )
