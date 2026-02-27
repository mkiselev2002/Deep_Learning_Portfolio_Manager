"""
Configuration and constants for the AI Paper Trading Simulator.
Loads the API key robustly from env.env regardless of the variable name encoding.
"""
import os
from pathlib import Path

# ─── Locate and load env file ──────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent      # wherever config.py lives

_ENV_CANDIDATES = [
    _PROJECT_ROOT / ".env",
    _PROJECT_ROOT / "env.env",
]

def _extract_api_key() -> str:
    """Read the first line that looks like an Anthropic API key from any env file."""
    # 1. Check standard environment variable first
    for var in ("ANTHROPIC_API_KEY", "PROVIDER_API_KEY"):
        val = os.environ.get(var, "")
        if val.startswith("sk-ant-"):
            return val

    # 2. Scan candidate env files for a key-value pair whose value starts with sk-ant-
    for path in _ENV_CANDIDATES:
        if not path.exists():
            continue
        try:
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if "=" not in line or line.startswith("#"):
                    continue
                _, raw_val = line.split("=", 1)
                val = raw_val.strip().strip('"').strip("'").strip()
                if val.startswith("sk-ant-"):
                    return val
        except Exception:
            continue

    return ""


API_KEY: str = _extract_api_key()

# ─── Model ─────────────────────────────────────────────────────────────────
MODEL = "claude-3-5-haiku-20241022"          # fast & cheap for an agentic loop

# ─── Simulation parameters ─────────────────────────────────────────────────
INITIAL_CAPITAL: float = 100_000.0
SIMULATION_DAYS: int = 5                     # 1 trading week

# ─── Risk constraints ──────────────────────────────────────────────────────
MAX_POSITION_PCT: float = 0.40               # 40 % max single-position weight
MAX_TRADES_PER_DAY: int = 2

# ─── Technical-analysis look-back windows ──────────────────────────────────
MOMENTUM_WINDOW: int = 20                    # days for momentum calculation
MEAN_REV_WINDOW: int = 20                    # days for z-score calculation
RSI_WINDOW: int = 14
LOOKBACK_DAYS: int = 30                      # minimum history needed

# ─── Universe ──────────────────────────────────────────────────────────────
ETF_TICKERS: list[str] = [
    "SPY", "QQQ", "IWM",   # US equity
    "EFA", "EEM",           # International equity
    "GLD", "TLT",           # Defensive / bonds
    "VNQ", "XLE", "XLF",   # Sector ETFs
]

# ─── Paths ─────────────────────────────────────────────────────────────────
DB_PATH  = str(_PROJECT_ROOT / "portfolio.db")
CSV_PATH = str(_PROJECT_ROOT / "data" / "etf_prices.csv")
