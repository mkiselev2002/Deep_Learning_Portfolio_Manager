"""
agents/
───────
Four-agent pipeline for the AI Paper Trading Simulator.

  DataAgent       – fetches & computes all technical indicators
  StrategyAgent   – selects trades based on configurable strategy rules
  RiskAgent       – validates trades against portfolio-level risk constraints
  ExecutionAgent  – executes approved trades and persists results to the DB
"""

from agents.data_agent      import DataAgent
from agents.strategy_agent  import StrategyAgent
from agents.risk_agent      import RiskAgent
from agents.execution_agent import ExecutionAgent

__all__ = ["DataAgent", "StrategyAgent", "RiskAgent", "ExecutionAgent"]
