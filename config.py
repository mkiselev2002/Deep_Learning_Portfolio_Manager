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
MODEL = "claude-haiku-4-5-20251001"          # fast & cheap for an agentic loop

# ─── Simulation parameters ─────────────────────────────────────────────────
INITIAL_CAPITAL: float = 1_000_000.0
SIMULATION_DAYS: int = 10                    # ~14 calendar days (2 trading weeks)

# ─── Risk constraints ──────────────────────────────────────────────────────
MAX_POSITION_PCT: float = 0.40               # 40 % max single-position weight
MAX_TRADES_PER_DAY: int = 20   # liquidate N positions + buy 5 = up to ~10/day

# ─── Technical-analysis look-back windows ──────────────────────────────────
MOMENTUM_WINDOW: int = 20                    # days for momentum calculation
MEAN_REV_WINDOW: int = 20                    # days for z-score calculation
RSI_WINDOW: int = 14
LOOKBACK_DAYS: int = 30                      # minimum history needed

# ─── Paths ─────────────────────────────────────────────────────────────────
# On Railway, DATA_DIR env var points to the persistent Volume (e.g. /data).
# portfolio.db goes there so simulation progress survives redeploys.
# backtest.db stays ephemeral (always reset per backtest run).
_DATA_DIR        = Path(os.environ.get("DATA_DIR", str(_PROJECT_ROOT)))
DB_PATH          = str(_DATA_DIR / "portfolio.db")
BACKTEST_DB_PATH = str(_PROJECT_ROOT / "backtest.db")

# ─── Security ───────────────────────────────────────────────────────────────
# Set the RESET_PASSWORD environment variable in Railway (or .env) to protect
# the "Start New Game" action.  Falls back to a safe default for local dev.
RESET_PASSWORD: str = os.environ.get("RESET_PASSWORD", "reset123")
