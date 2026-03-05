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

            result[ticker] = {
                "price":        round(price,   2),
                "momentum_20d": round(mom_20d, 2),
                "momentum_5d":  round(mom_5d,  2),
                "zscore":       round(zscore,  3),
                "rsi":          round(rsi,     1),
                "above_sma20":  price > sma20,
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
            positions_detail.append({
                "ticker":        ticker,
                "shares":        round(pos["shares"], 4),
                "avg_cost":      round(pos["average_cost"], 2),
                "current_price": round(pos["current_price"], 2),
                "value":         round(pos["value"], 2),
                "pnl_pct":       round(pos["pnl_pct"], 2),
                "pnl_dollar":    round(pos["value"] - pos["shares"] * pos["average_cost"], 2),
            })

        positions_detail.sort(key=lambda x: x["pnl_pct"], reverse=True)
        top_performers   = [p for p in positions_detail if p["pnl_pct"] > 0][:3]
        underperformers  = [p for p in positions_detail if p["pnl_pct"] < 0][:3]

        # ── News headlines ───────────────────────────────────────────────
        held_tickers = [p["ticker"] for p in positions_detail]
        news = self._fetch_news(held_tickers) if held_tickers else {}

        # ── Build context dict ───────────────────────────────────────────
        context = {
            "generated_date":   today,
            "portfolio_value":  round(portfolio["total_value"], 2),
            "cash":             round(portfolio["cash"], 2),
            "initial_capital":  INITIAL_CAPITAL,
            "positions":        positions_detail,
            "top_performers":   top_performers,
            "underperformers":  underperformers,
            "news_headlines":   news,
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
        if API_KEY:
            return self._llm_summary(context)
        return self._rule_based_summary(context)

    def _llm_summary(self, context: dict) -> str:
        """Ask Claude to write the portfolio narrative."""
        # Build a compact representation for the prompt
        positions_text = "\n".join(
            f"  {p['ticker']}: ${p['value']:,.0f}  P/L {p['pnl_pct']:+.1f}%"
            for p in context["positions"]
        ) or "  (no open positions)"

        news_text = ""
        for ticker, headlines in context.get("news_headlines", {}).items():
            news_text += f"\n  {ticker}:\n"
            for h in headlines:
                news_text += f"    • {h}\n"
        if not news_text:
            news_text = "  (no headlines available)"

        prompt = f"""You are a sharp portfolio analyst writing a daily briefing for a paper trading desk.

PORTFOLIO SNAPSHOT  ({context['generated_date']})
──────────────────────────────────────────────────
Total Value:   ${context['portfolio_value']:,.2f}
Cash:          ${context['cash']:,.2f}
All-time P/L:  ${context['alltime_pnl']:+,.2f}  ({context['alltime_pnl_pct']:+.2f}%)
5-day P/L:     ${context['fiveday_pnl']:+,.2f}  ({context['fiveday_pnl_pct']:+.2f}%)
Yesterday P/L: ${context['daily_pnl']:+,.2f}  ({context['daily_pnl_pct']:+.2f}%)
Win Rate:      {context['win_rate']:.0f}%  ({context['total_days']} trading days)

POSITIONS
─────────
{positions_text}

RECENT NEWS HEADLINES
─────────────────────{news_text}

Write a punchy 3-paragraph briefing:
1. OVERALL: Portfolio value, trend, and all-time / 5-day performance in plain English.
2. WINNERS & LOSERS: What's working and what isn't — be specific with tickers and numbers.
3. MARKET CONTEXT: Using the news headlines, what macro or sector forces likely explain the moves? If no news, make an educated guess based on the tickers.

Keep it concise, analytical, and slightly high-stakes. No bullet points — flowing prose only."""

        try:
            client   = anthropic.Anthropic(api_key=API_KEY)
            response = client.messages.create(
                model=MODEL,
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as exc:
            logger.warning("LLM summary failed (%s) — using fallback.", exc)
            return self._rule_based_summary(context)

    def _rule_based_summary(self, context: dict) -> str:
        """Deterministic fallback summary when the LLM is unavailable."""
        pv   = context["portfolio_value"]
        pnl  = context["alltime_pnl"]
        ppct = context["alltime_pnl_pct"]
        fpct = context["fiveday_pnl_pct"]
        sign = "up" if pnl >= 0 else "down"

        winners = ", ".join(
            f"{p['ticker']} ({p['pnl_pct']:+.1f}%)"
            for p in context["top_performers"]
        ) or "none"
        losers = ", ".join(
            f"{p['ticker']} ({p['pnl_pct']:+.1f}%)"
            for p in context["underperformers"]
        ) or "none"

        api_note = (
            "(AI summary temporarily unavailable — rule-based fallback in use.)"
            if API_KEY
            else "(Connect a Claude API key for a full AI-generated market context analysis.)"
        )

        return (
            f"Portfolio is currently valued at ${pv:,.2f}, {sign} "
            f"${abs(pnl):,.2f} ({ppct:+.2f}%) from the initial ${context['initial_capital']:,.0f} stake. "
            f"Over the last 5 trading days the book moved {fpct:+.2f}%, "
            f"with a win rate of {context['win_rate']:.0f}% across {context['total_days']} recorded days.\n\n"
            f"Top performers: {winners}. "
            f"Underperformers: {losers}.\n\n"
            f"{api_note}"
        )
