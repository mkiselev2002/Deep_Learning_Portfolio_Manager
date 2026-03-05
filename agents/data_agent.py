"""
agents/data_agent.py
────────────────────
DataAgent — the "Analyst" in the pipeline.

Responsibilities
────────────────
  1. Receives the full price history DataFrame and the current simulation date.
  2. Computes a standard suite of technical indicators for every ticker
     up to (and including) the current date.
  3. Returns a structured analysis dict consumed by StrategyAgent.

Indicators computed per ticker
──────────────────────────────
  price         – latest close
  momentum_20d  – 20-day price return (%)
  momentum_5d   – 5-day price return (%)
  zscore        – (price − SMA_20) / σ_20  (mean-reversion signal)
  rsi           – 14-day RSI (Wilder smoothing)
  above_sma20   – bool: price > 20-day simple moving average
"""

import numpy as np
import pandas as pd

from config import MOMENTUM_WINDOW, MEAN_REV_WINDOW, RSI_WINDOW


class DataAgent:
    """
    Pure-Python analyst agent.  No external API calls — only pandas math.

    Usage
    -----
    agent    = DataAgent()
    analysis = agent.analyze(prices_df, current_date)
    """

    def analyze(
        self,
        prices_df: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> dict[str, dict]:
        """
        Parameters
        ----------
        prices_df    : DataFrame indexed by date, one column per ticker.
        current_date : The 'today' of the simulation (inclusive upper bound).

        Returns
        -------
        {
            ticker: {
                "price":        float,
                "momentum_20d": float,   # % return over 20 trading days
                "momentum_5d":  float,   # % return over 5 trading days
                "zscore":       float,   # (price - mean_20) / std_20
                "rsi":          float,   # 14-day RSI
                "above_sma20":  bool,    # price > 20-day SMA
            },
            ...
        }
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

            # ── Momentum ──────────────────────────────────────────────────────
            mom_20d = (price / float(series.iloc[-MOMENTUM_WINDOW]) - 1.0) * 100.0
            mom_5d  = (
                (price / float(series.iloc[-5]) - 1.0) * 100.0
                if len(series) >= 5
                else 0.0
            )

            # ── Mean-reversion z-score ─────────────────────────────────────────
            window = series.iloc[-MEAN_REV_WINDOW:]
            mu, std = window.mean(), window.std()
            zscore = float((price - mu) / (std + 1e-9))

            # ── RSI (Wilder smoothing) ─────────────────────────────────────────
            deltas   = series.iloc[-(RSI_WINDOW + 1):].diff().dropna()
            gains    = deltas.clip(lower=0)
            losses   = (-deltas.clip(upper=0))
            avg_gain = gains.mean()
            avg_loss = losses.mean()
            rs  = avg_gain / (avg_loss + 1e-9)
            rsi = float(100.0 - 100.0 / (1.0 + rs))

            # ── SMA filter ────────────────────────────────────────────────────
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
