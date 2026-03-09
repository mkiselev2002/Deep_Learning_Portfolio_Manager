# CLAUDE.md — Deep Learning Portfolio Manager

AI-powered paper trading simulator themed as a casino. An agentic Claude pipeline makes daily S&P 500 trading decisions. Supports live simulation and historical backtesting.

---

## Commands

```bash
# Run locally
streamlit run app.py

# Install dependencies
pip install -r requirements.txt
```

**Local env file** — create `.env` or `env.env` in the project root:
```
ANTHROPIC_API_KEY=sk-ant-...
RESET_PASSWORD=your_password
```

**Railway deployment** — configured via `railway.json` (Nixpacks builder):
```
startCommand: streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

---

## Environment Variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | — | Claude API (also reads `PROVIDER_API_KEY`) |
| `RESET_PASSWORD` | No | `reset123` | Gate for "Start New Game" button |
| `DATA_DIR` | No | project root | Persistent volume path (Railway: `/data`) |
| `PORT` | No | `8501` | Set by Railway automatically |

---

## Architecture

### Two Modes

- **Simulation** — live paper trading, one day at a time. Uses `portfolio.db` (persistent volume).
- **Backtest** — replay a historical date window. Uses `backtest_<uuid10>.db` (ephemeral per session, auto-cleaned after 4 h).

### Four-Agent Pipeline

Each "hand" runs the agents in sequence:

```
DataAgent → StrategyAgent → RiskAgent → ExecutionAgent
```

**1. DataAgent** (`agents/data_agent.py`)
- Pure Python, no LLM.
- Computes per-ticker signals from `prices_df` filtered to `<= current_date` (no look-ahead).
- Signals: 20d momentum, 5d momentum, RSI (Wilder), z-score vs 20d SMA, 5d realized vol, previous-day return.
- Optionally uses Claude to write a narrative portfolio report; falls back to rule-based.

**2. StrategyAgent** (`agents/strategy_agent.py`)
- LLM-first with deterministic fallback.
- Algorithm: top-50 by 5d realized vol → sort by prev-day return (biggest losers) → Claude picks 3–5 via tool-use → propose BUY/SELL list.
- Fallback (no API key or exception): deterministic Python using RSI thresholds.
- Tracks `api_failed` and `api_error` for UI display.
- Full liquidation of all positions every day before buying.

**3. RiskAgent** (`agents/risk_agent.py`)
- Pure Python, enforces hard constraints — **overrides LLM sizing entirely**.
- SELLs always liquidate the full position.
- BUYs: equal-dollar split = `(cash × 0.995) / n_buys`, whole shares only (floor).
- Constraints: max 5 positions, max 40% per position, max 20 trades/day.
- Returns approved trades + violation log. If any REJECTED (not trimmed), StrategyAgent retries up to 3×.

**4. ExecutionAgent** (`agents/execution_agent.py`)
- Thin wrapper around `db.execute_trade()`.
- Writes trades to `transactions`, updates `positions`, closes `position_history` rows, posts `daily_snapshots`.

### Liquidation Safety Net

After `strategy_agent.propose_trades()` returns, `_deal_next_hand()` scans held positions and **injects missing SELL proposals** for any position the LLM omitted. Ensures full daily liquidation regardless of LLM output.

---

## Database Schema

Two SQLite files, same schema.

| Table | Purpose |
|---|---|
| `simulation` | Single row: current date, cash, total value |
| `positions` | Current open holdings (symbol, shares, avg_cost) |
| `position_history` | Full lifecycle: open/close dates, shares, avg_cost, realized P&L |
| `transactions` | Append-only trade log (symbol, action, shares, price, amount, reason) |
| `daily_snapshots` | Daily equity for charting |
| `sp500_stocks` | S&P 500 constituents + latest OHLCV (`price_date` vs `fetch_date`) |
| `prices` | Long-format close prices (date, symbol, close); indexed on date |
| `day_results` | JSON blobs of completed simulation day results |
| `agent_logs` | Pipeline event log (agent, event, trades, violations, reasoning) |

**`prices` table is wide-format in Python, long-format in SQLite.** `upsert_prices()` converts wide→long before writing; `load_prices()` pivots long→wide on read.

---

## Key Files

```
app.py              Main Streamlit app (~3 300 lines)
config.py           Constants, env loading, paths
database.py         PortfolioDatabase class, all SQL
market_data.py      yfinance + Wikipedia S&P 500 fetching
agents/
  data_agent.py     Technical indicators + portfolio report
  strategy_agent.py LLM trade proposals + fallback
  risk_agent.py     Hard constraint validator
  execution_agent.py DB writer
.streamlit/
  config.toml       Forces dark mode (base = "dark")
railway.json        Railway build/start config
Procfile            Heroku-compatible start command
```

---

## app.py Key Functions

| Function | Purpose |
|---|---|
| `main()` | Entry point; routes to mode/stage screens |
| `_get_session_backtest_db_path()` | UUID-based per-session backtest DB path |
| `_cleanup_old_backtest_dbs()` | Prune backtest DBs older than 4 h (runs once per session) |
| `_sync_sim_prices()` | Smart price init: sim DB → backtest DB → full 2019 fetch |
| `_ensure_price_history()` | Called on server start; wraps `_sync_sim_prices` |
| `run_single_day()` | Orchestrates all 4 agents for one simulation day |
| `_deal_next_hand()` | Full proposal pipeline with retry loop + SELL safety net |
| `show_proposals_dialog()` | Full-page trade confirmation (not `@st.dialog` — unreliable for DB writes) |
| `_start_backtest_with_loading()` | Probe window for existing data, fetch if needed, seed backtest DB |
| `_restore_session_from_db()` | Restore sim state from DB on page refresh |
| `_log_agent()` | Write agent event to session state + DB |
| `load_price_data()` | `@st.cache_data` wide-format price DF |

**Tab renderers:** `render_daily_summary`, `render_portfolio`, `render_performance`, `render_market_data`, `render_agent_logs`, `render_roulette`

---

## Session State Keys

```python
# Core simulation
sim_started          # bool — simulation is active
sim_date_idx         # int — index into sim_dates list
sim_day_num          # int — display day number (1-based)
sim_dates            # list[Timestamp] — ordered trading dates
sim_capital          # float — initial capital constant
sim_strategy         # str — "Momentum", "Mean Reversion", etc.
daily_results        # list[dict] — completed day results
pending_day_data     # dict | None — trade proposal payload (triggers confirmation page)
agent_logs           # list[dict] — pipeline event log

# Mode & routing
app_mode             # "simulation" | "backtest"
new_game_stage       # "mode_select" | "bt_setup" | "bt_resetting" | "bt_ready" | "fetching" | ...

# Backtest-specific
backtest_db_path     # str — per-session ephemeral DB path
bt_start_date        # str — ISO date chosen in bt_setup
_bt_pick_year/month/day  # int — date picker widget state

# Mini-game (session only, no DB)
roulette_wins/losses/streak/result
```

---

## Price Data Design

- **All prices are adjusted close** (`auto_adjust=True` in yfinance). Historical prices account for dividends/splits. This prevents artificial signals on ex-dividend dates.
- **`prices` table** stores long-format `(date TEXT, symbol TEXT, close REAL)`. `upsert_prices` / `load_prices` convert to/from wide-format DataFrame.
- **`portfolio.db` can have date gaps** — e.g., Oct 2019–Feb 2020 from one backtest + Dec 2025–present from live simulation. `_start_backtest_with_loading` probes the actual backtest window directly (not just min/max) before deciding whether to fetch.
- **S&P 500 symbols** — dots replaced with dashes (`BRK.B → BRK-B`) for yfinance compatibility.

---

## Backtest Mode Details

1. User picks a start date in `_render_bt_setup()`.
2. `bt_resetting` stage always passes the **session backtest DB explicitly** to `_start_backtest_with_loading` — `app_mode` is still `"simulation"` at this point, so the `db` variable would resolve to `portfolio.db` if used directly.
3. `_start_backtest_with_loading` probes the backtest window in `sim_db` (portfolio.db); fetches Oct 2019-window-end data only if the window has < `SIMULATION_DAYS` trading days already stored.
4. Copies the price window (buffer_start → end_ts) into the per-session `backtest_<id>.db`.
5. `bt_dates` list is stored in session state. Each day advances through it.
6. Backtest is **backwards-looking**: DataAgent always filters `prices_df` to `<= current_date`. The backtest DB contains future prices for lookback purposes but agents never see them.

---

## Performance Metrics (render_performance tab)

Computed from `daily_snapshots` vs SPY (fetched and cached 1 h):

| Metric | Formula |
|---|---|
| Total Return | `(final_value / initial - 1) × 100` |
| Ann. Return | Total return annualised to 252 trading days |
| Sharpe | `(mean daily return - rf/252) / std × √252` |
| Sortino | Same but only downside std dev |
| Beta | `cov(portfolio, SPY) / var(SPY)` |
| Alpha (Jensen) | `port_ann - rf - beta × (spy_ann - rf)` |
| Max Drawdown | `(peak - trough) / peak × 100` (most negative) |
| Ann. Volatility | `std(daily returns) × √252` |

Metric card arrows (▲/▼) are coloured **independently** from the value sign: green ▲ = beating SPY, red ▼ = losing to SPY.

---

## Known Quirks & Gotchas

- **`@st.dialog` removed** — `show_proposals_dialog` is rendered inline with `return` in `main()`. The dialog decorator silently failed to commit SQLite writes in Railway's deployment context.
- **Risk agent overrides LLM sizing** — pct_of_portfolio from StrategyAgent is ignored; RiskAgent recalculates equal-dollar allocations.
- **No fractional shares** — `math.floor()` applied, leaving small cash residual. 0.5% buffer reserved per trade.
- **SPY Sharpe in backtest** — may show N/A if SPY data window doesn't align.
- **Concurrent simulation** — `portfolio.db` is shared across all users (collaborative play). Backtest DBs are fully isolated per session.
- **`st.cache_data` on `load_price_data`** — cache is explicitly cleared in `_deal_next_hand` before each trade cycle to ensure fresh prices.
- **`_restore_session_from_db` guard** — runs only once per session (`if "sim_started" in st.session_state: return`). Must come before the defaults loop.
- **yfinance batch size** — 100 tickers per call to keep memory usage bounded; `del raw, close_batch` after each batch.
- **Position cost blending** — adding to an existing position recalculates `avg_cost` as a weighted average. Partial sells reduce shares but don't close; full exits record `closed_date` and `realized_pnl`.
