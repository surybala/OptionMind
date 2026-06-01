# Legacy OptionWheel README

This document is preserved for historical context only. It describes the inherited deterministic scanner and old OptionWheel behavior. The active OptionMind direction is ML-first and starts in the root [README.md](../README.md) and [North Star Spec](NORTH_STAR_SPEC.md).

---

# OptionMind

An algorithmic options-selling agent that scans index ETF and index-member universes for high-probability premium-selling opportunities, analyses market sentiment when enabled, monitors open positions for profit-take / stop-loss / gamma-risk exits, gates new trades at the portfolio level, and executes orders via the **Alpaca Trading API**.

OptionMind is now being extended into an ML-driven option trading research and execution platform. New sessions should read [docs/NORTH_STAR_SPEC.md](docs/NORTH_STAR_SPEC.md) first; the strategy/scanner sections below describe the inherited deterministic baseline, not the future intelligence layer. See also [docs/PROJECT_CHARTER.md](docs/PROJECT_CHARTER.md), [docs/ML_ROADMAP.md](docs/ML_ROADMAP.md), [docs/DATA_SOURCE_FINDINGS.md](docs/DATA_SOURCE_FINDINGS.md), and [docs/PROVIDER_INTERFACES.md](docs/PROVIDER_INTERFACES.md).

---

## Table of Contents

1. [Strategies](#strategies)
2. [Quick Start](#quick-start)
3. [Alpaca Credentials](#alpaca-credentials)
4. [Alpaca Account Setup](#alpaca-account-setup)
5. [Configuration](#configuration)
6. [Running the Agent](#running-the-agent)
7. [Headless / Daemon Mode](#headless--daemon-mode)
8. [Manual Position Management](#manual-position-management)
9. [Sentiment Analysis](#sentiment-analysis)
10. [Risk Management](#risk-management)
    - [Portfolio-Level Pre-Trade Risk Gates](#portfolio-level-pre-trade-risk-gates)
    - [Stop-Loss Monitor](#stop-loss-monitor)
    - [Gamma / Theta Risk Trigger](#gamma--theta-risk-trigger)
11. [HFT Mode](#hft-mode)
12. [Dashboard](#dashboard)
13. [How It Works](#how-it-works)
14. [Project Structure](#project-structure)
15. [Running Tests](#running-tests)

---

## Strategies

All strategies sell premium and profit when the underlying stays within a defined range. Defined-risk strategies (spreads, condors, butterflies) are preferred — they cap maximum loss and require far less capital than stock-ownership strategies.

| Strategy | Type | Description | Capital Required |
|---|---|---|---|
| **PCS** – Put Credit Spread | Defined risk | Sell a put + buy a lower put; bullish/neutral | `spread_width × 100` |
| **CCS** – Call Credit Spread | Defined risk | Sell a call + buy a higher call; bearish/neutral | `spread_width × 100` |
| **IC** – Iron Condor | Defined risk | PCS + CCS simultaneously; profits in a range | `max(wing) × 100` |
| **IFLY** – Iron Butterfly | Defined risk | Sell ATM put + ATM call + buy OTM wings; higher premium than IC, profits near entry | `max(wing) × 100` |
| **CSP** – Cash-Secured Put | Cash collateral | Sell an OTM put; worst case is owning the stock | `strike × 100` |
| **STRANGLE** – Short Strangle | Naked (disabled) | Sell OTM put + OTM call; maximum premium, high margin requirement | N/A |
| **CC** – Covered Call | Stock ownership (disabled) | Sell OTM call against a long stock position | `stock_price × 100` |

> **Disabled by default:** `IC`, `IFLY`, `CSP`, `CC`, and `STRANGLE` are disabled in `config.json`. PCS/CCS are the default active strategies. Enable optional strategies by setting `"enabled": true` in their config block.

Picks are ranked by a **yield-normalised score**: `score = (premium / spread_width) × prob_win²`. Dividing by width removes the incentive to pick near-the-money spreads that collect more absolute dollars — a 10% credit-yield spread at 10% OTM scores identically to a 10%-yield spread at 5% OTM, so `prob_win²` becomes the sole safety differentiator. For non-spread strategies (CSP, CC, STRANGLE) the original `premium × prob_win²` formula applies.

---

## Quick Start

### Windows
```bat
setup.bat
```

### macOS / Linux / Git Bash
```bash
chmod +x setup.sh
./setup.sh
```

Both scripts will:
1. Locate Python 3.10+
2. Create a `.venv` virtual environment
3. Install all dependencies (`alpaca-py`, `yfinance`, `scipy`, `pandas`, `requests`)
4. Set up your Alpaca credentials (see below)
5. Run the full test suite
6. Launch `agent.py`

---

## Alpaca Credentials

Credentials are resolved in this priority order — **env vars win**:

| Priority | Source | How to set |
|---|---|---|
| **1 (highest)** | Environment variables | `ALPACA_API_KEY`, `ALPACA_API_SECRET`, `ALPACA_PAPER` |
| **2** | `config.json` | `alpaca.api_key`, `alpaca.api_secret`, `alpaca.paper` |

### Recommended: use a `.env` file (never committed to git)

```bash
# Copy the template
cp .env.example .env

# Edit .env and fill in your real keys
ALPACA_API_KEY=your_api_key_here
ALPACA_API_SECRET=your_api_secret_here
ALPACA_PAPER=true      # true=paper, false=live
```

Then load it before running (or use `direnv` / your IDE's env support):

```bash
# Bash / Git Bash / macOS / Linux
export $(grep -v '^#' .env | xargs)

# PowerShell (Windows)
Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]+)=(.+)$') {
        [System.Environment]::SetEnvironmentVariable($Matches[1].Trim(), $Matches[2].Trim())
    }
}
```

> **`.env` is in `.gitignore`** — it will never be accidentally committed. `.env.example` is the safe checked-in template with no real values.

### Alternative: fill in `config.json` directly (not recommended for shared repos)

```json
"alpaca": {
  "api_key":    "YOUR_ALPACA_API_KEY",
  "api_secret": "YOUR_ALPACA_SECRET_KEY",
  "paper":      true
}
```

> **Credentials missing?** The agent still runs in scan-only mode — it scans, scores, and saves picks to `data/pending_picks.json` without placing any orders.

---

## Alpaca Account Setup

1. Sign up at [alpaca.markets](https://alpaca.markets)
2. Enable **Options Trading** in your account settings
3. Copy your **API Key** and **Secret Key** from the dashboard
4. Add them to `.env` (recommended) or `config.json`
5. Start with `ALPACA_PAPER=true` to test without real money

---

## Configuration

Edit **`config.json`** to tune strategy settings (leave `api_key`/`api_secret` blank — use env vars instead):

```json
{
  "market_cap_min": 1000000000,
  "expiry_days_max": 14,
  "top_n_per_strategy": 5,
  "max_picks_per_ticker": 1,
  "max_capital_per_period": 50000,
  "max_contracts_per_pick": 50,
  "universe": "etf",

  "alpaca": {
    "api_key":           "",
    "api_secret":        "",
    "paper":             true,
    "approve_mode":      true,
    "auto_execute_prob": 0.80
  },

  "risk_parameters": {
    "max_delta":                 0.1,
    "min_probability_of_expiry": 0.9,
    "auto_execute_prob":         0.80,
    "stop_loss_multiplier":      2.0,
    "max_loss_multiple": {
      "enabled": true,
      "default": 30.0,
      "by_strategy": {
        "PCS": 30.0,
        "CCS": 30.0,
        "IC":  30.0,
        "IFLY": 30.0
      }
    },
    "directional_exposure_caps": {
      "enabled": true,
      "put":  0.50,
      "call": 0.50,
      "min_side_cap_dollars": 1500
    },
    "portfolio_gamma_risk": {
      "enabled": true,
      "fail_closed": true,
      "max_stress_loss_pct": 0.10,
      "min_stress_loss_dollars": 500,
      "shock_moves_pct": [1, 2, 3, 5],
      "iv_shock_points": 10,
      "max_gamma_loss_to_daily_theta": 2.0,
      "gamma_loss_to_theta_warning_only": true,
      "near_expiry_dte": 2,
      "max_near_expiry_stress_pct": 0.0025,
      "min_near_expiry_stress_dollars": 250,
      "symbol_stress_cap_enabled": true,
      "max_symbol_stress_pct": 0.05,
      "min_symbol_stress_dollars": 250,
      "complex_spread_quantity_cap": {
        "enabled": true,
        "strategies": ["IC", "IFLY"],
        "symbol_stress_threshold_pct": 0.50,
        "max_quantity": 1
      },
      "expiry_bucket_cap_enabled": false,
      "max_expiry_bucket_pct": 0.0,
      "min_expiry_bucket_dollars": 500,
      "default_iv": 0.25
    },
    "regime_filter": {
      "enabled": true,
      "fail_closed": false,
      "yellow_quantity_multiplier": 0.65,
      "orange_quantity_multiplier": 0.30,
      "yellow_top_n_multiplier": 0.70,
      "orange_top_n_multiplier": 0.35,
      "vix": {
        "green_below": 18,
        "yellow_below": 25,
        "orange_below": 32
      },
      "vix_spike": {
        "one_day_pct": 0.15,
        "red_one_day_pct": 0.25,
        "three_day_pct": 0.25
      },
      "trend": {
        "symbol": "SPY",
        "sma_fast": 20,
        "sma_slow": 50
      },
      "realized_vol": {
        "fast_days": 5,
        "slow_days": 20,
        "expansion_multiple": 1.50
      }
    },
    "gamma_risk": {
      "enabled":                    true,
      "gamma_theta_ratio_threshold": 1.5,
      "min_delta_to_trigger":        0.15,
      "min_profit_captured_pct":     0.25,
      "urgent_delta_threshold":      0.30
    }
  },

  "sentiment": {
    "enabled":            false,
    "lookback_days":      20,
    "rsi_period":         14,
    "bull_threshold":     0.20,
    "bear_threshold":     0.20,
    "skew_factor":        0.30,
    "max_skew":           0.50,
    "top_n_skew_factor":  0.50,
    "weight_rsi":         0.35,
    "weight_sma":         0.35,
    "weight_momentum":    0.30
  },

  "dynamic_width": {
    "enabled": true,
    "tiers": [
      { "max_price": 50,   "width": 5  },
      { "max_price": 150,  "width": 10 },
      { "max_price": 300,  "width": 15 },
      { "max_price": 500,  "width": 15 },
      { "max_price": 9999, "width": 15 }
    ]
  },

  "strategies": {
    "put_credit_spread":  { "enabled": true,  "min_net_credit": 0.20, "max_delta_short_leg": 0.30, "strike_width": 5,  "min_prob_profit": 0.75 },
    "call_credit_spread": { "enabled": true,  "min_net_credit": 0.20, "max_delta_short_leg": 0.30, "strike_width": 5,  "min_prob_profit": 0.75 },
    "iron_condor":        { "enabled": false, "min_net_credit": 0.40, "max_delta_short_leg": 0.25, "put_strike_width": 5, "call_strike_width": 5, "min_prob_profit": 0.75 },
    "iron_butterfly":     { "enabled": false, "min_net_credit": 1.50, "put_wing_width": 10, "call_wing_width": 10, "min_prob_profit": 0.60, "atm_pct_tolerance": 0.025 },
    "covered_put":        { "enabled": false },
    "short_strangle":     { "enabled": false },
    "covered_call":       { "enabled": false }
  }
}
```

### Key Reference

| Key | Description |
|---|---|
| `paper` | `true` = paper trading, `false` = live account |
| `approve_mode` | `true` = interactive approval gate before execution (recommended) |
| `auto_execute_prob` | Probability threshold for auto-execution in `auto` mode (default: 0.80) |
| `market_cap_min` | Minimum market cap for the ticker universe (default: $1B) |
| `expiry_days_max` | Only consider options expiring within this many days (default: 14) |
| `top_n_per_strategy` | Top-ranked picks to surface **per strategy** per scan (default: 5). Total picks = this × number of enabled strategies. |
| `max_picks_per_ticker` | Max picks per ticker across all strategies (default: 1) |
| `max_capital_per_period` | Cap total capital deployed per trading period in dollars (default: $50,000) |
| `stop_loss_multiplier` | Fallback guard for CSP/CC/STRANGLE: exit when loss > `multiplier × entry_premium` (default: 2.0). For spreads the width-relative guard below takes precedence. |
| `stop_loss_max_loss_pct` | Primary guard for spreads (PCS/CCS/IC): exit when unrealised loss > `pct × (width − entry_premium)` (default: 0.80 = 80% of max loss). Set to `null` to use multiplier only. |
| `risk_parameters.max_loss_multiple` | Pre-trade account gate for defined-risk spreads. Rejects picks whose max loss is too large relative to premium. Current index-trading default is `30.0` to avoid filtering out every reasonable far-OTM spread. |
| `risk_parameters.directional_exposure_caps` | Pre-trade gross side-exposure cap. Put-side max loss and call-side max loss are each capped at 50% of the capital budget by default; portfolio stress and symbol stress caps remain the primary tail-risk brakes. |
| `directional_exposure_caps.min_side_cap_dollars` | Dollar floor for each directional side cap. Default `$1,500` keeps small budgets from rejecting every one-contract `$15`-wide spread solely because of percentage sizing. |
| `risk_parameters.portfolio_gamma_risk` | Portfolio-level stress gate. Simulates existing open and pending-close positions plus candidate picks across configured spot and IV shocks before opening more risk. |
| `portfolio_gamma_risk.max_stress_loss_pct` | Hard cap on worst simulated portfolio stress loss as a percentage of account capital / period budget. Default `0.10` = 10%. |
| `portfolio_gamma_risk.min_stress_loss_dollars` | Dollar floor for the global stress cap. Default `$500`. |
| `portfolio_gamma_risk.max_symbol_stress_pct` | Hard cap on worst simulated stress loss concentrated in one symbol. Default `0.05` = 5% of account capital / period budget. |
| `portfolio_gamma_risk.min_symbol_stress_dollars` | Dollar floor for the per-symbol stress cap. Default `$250`. |
| `portfolio_gamma_risk.complex_spread_quantity_cap` | Caps IC/IFLY quantity to `max_quantity` when one contract would consume at least the configured fraction of the symbol stress cap. Default: cap to `1` at 50% of symbol stress capacity. |
| `portfolio_gamma_risk.gamma_loss_to_theta_warning_only` | When `true`, the dashboard and logs warn when 1% gamma loss overwhelms daily theta, but the scanner does not block picks solely for this ratio. |
| `portfolio_gamma_risk.expiry_bucket_cap_enabled` | Enables a hard cap on stress loss concentrated in a single expiry bucket. Default is `false` because the current index-only workflow was too constrained with this hard gate on. |
| `portfolio_gamma_risk.fail_closed` | If stress data cannot be computed, reject new picks instead of opening unknown risk. |
| `risk_parameters.regime_filter` | New-trade throttle based on VIX level/spikes, SPY trend, and realized-volatility expansion. `GREEN` trades normally, `YELLOW` and `ORANGE` reduce scan count and quantity, and `RED` pauses new openings. |
| `dynamic_width.enabled` | Scale spread width to stock price using the tier table below. `false` uses `strike_width` from the strategy config (default: `true`). |
| `dynamic_width.tiers` | List of `{"max_price": X, "width": Y}` entries. First tier where `spot ≤ max_price` wins. Current defaults cap dynamic width at $15. |
| `min_otm_pct.put/call` | Hard floor: the short leg must be at least this % OTM. Prevents near-ATM entries regardless of premium score (default: 0.08 = 8%). |
| `gamma_risk.enabled` | Enable the predictive gamma/theta risk trigger (default: true) |
| `gamma_risk.gamma_theta_ratio_threshold` | Close when `\|gamma\| / \|theta_per_day\| >= this` (default: 1.5) |
| `gamma_risk.min_delta_to_trigger` | Only trigger gamma exit when short leg delta >= this (default: 0.15) |
| `gamma_risk.min_profit_captured_pct` | Require this fraction of premium earned before early close (default: 0.25) |
| `gamma_risk.urgent_delta_threshold` | Bypass profit check and close immediately if short delta >= this (default: 0.30) |
| `sentiment.enabled` | Enable RSI + SMA + momentum sentiment engine |
| `sentiment.skew_factor` | How aggressively to shift per-ticker delta ceilings (0.30 = ±30%) |
| `sentiment.top_n_skew_factor` | How aggressively to reallocate pick counts between PCS and CCS per period (0.50 = up to ±50%) |
| `sentiment.max_skew` | Hard cap on any single delta adjustment (0.50 = never shift more than ±50% of base delta) |

---

## Running the Agent

```bash
# Activate the virtual environment first
# Windows:
.venv\Scripts\activate
# macOS / Linux / Git Bash:
source .venv/bin/activate
```

### Execution Modes

| Mode | Flag | Behaviour |
|---|---|---|
| **approve** (default) | _(none)_ | Scan, show ranked plan table, prompt for approval, execute approved picks |
| **auto** | `--mode auto` | Automatically execute any pick with `prob_win >= auto_execute_prob` |
| **scan-only** | `--mode scan-only` | Scan and print the plan; never execute; save picks to `data/pending_picks.json` |

### Trade Safety

| Flag | Behaviour |
|---|---|
| _(none)_ | Dry-run by default — orders are simulated, nothing reaches Alpaca |
| `--dry-run` | Explicit dry-run (same as default) |
| `--live` | Submit **real** orders to Alpaca; requires valid credentials |

### Ticker Universe

| Flag | Behaviour |
|---|---|
| _(none)_ | Uses `config.json` (`universe: "etf"` by default for index-ETF trading) |
| `--universe etf` | Broad ETF universe from NASDAQ / NYSE / NYSE Arca |
| `--universe index` | S&P 500 + NASDAQ-100 + Dow 30 (~517 tickers, cached 24h) |
| `--universe default` | NASDAQ + NYSE stocks filtered by `market_cap_min` (cached 24h) |
| `--tickers AAPL MSFT NVDA` | Explicit list of tickers (overrides `--universe`) |
| `--refresh-universe` | Force-refresh the cached universe (use with `--universe index`) |

### Other Flags

| Flag | Default | Description |
|---|---|---|
| `--config PATH` | `config.json` | Path to config file |
| `--db PATH` | `data/trades.db` | Path to the SQLite trades database |
| `--max-capital DOLLARS` | from config | Override `max_capital_per_period` for this run |
| `--top-n N` | from config | Override number of top picks to surface |

### Agent Examples

```bash
# Interactive approve mode, dry-run (safe default):
python agent.py

# Interactive approve mode, submit real orders when approved:
python agent.py --live

# Auto-execute high-confidence picks, live:
python agent.py --mode auto --live

# Scan only — no execution, no prompts:
python agent.py --mode scan-only

# Scan the configured ETF universe:
python agent.py --universe etf

# Scan specific tickers only:
python agent.py --tickers AAPL MSFT NVDA TSLA

# Override capital cap and top-N picks:
python agent.py --max-capital 30000 --top-n 5

# Use a different database:
python agent.py --db data/paper_trades.db
```

### Approval Gate

When running in `approve` mode, the agent prints a plan table like:

```
  #  Strategy  Symbol  Expiry      Legs                    Credit   Capital   Prob    ROI    Score
  1  IC        AAPL    2025-01-17  145/150 P  160/165 C    $0.85    $500      82.3%   17.0%  0.699
  2  IFLY      MSFT    2025-01-17  390/400(ATM)/410        $2.10    $1000     68.1%   21.0%  1.430
  3  PCS       NVDA    2025-01-17  480/475 P               $0.55    $500      79.4%   11.0%  0.437

  Total capital: $2,000   |   Total premium: $350.00
  Budget used: $2,000 of $50,000 (4.0%)

Enter picks to approve (e.g. 1,3  or  1-3  or  a=all  n=none  q=quit):
```

Enter numbers, ranges (`1-3`), `a` for all, `n` for none, or `q` to quit.

---

## Headless / Daemon Mode

All three processes (`agent.py`, `monitor.py`, `dashboard.py`) support a `--daemon` flag that runs them continuously in the background without requiring a terminal. `start.sh` launches the agent in `--mode auto` by default, so high-confidence picks above `auto_execute_prob` can execute without an approval prompt.

### How It Works

| Process | `--daemon` behaviour |
|---|---|
| `agent.py` | Wakes once per day at `schedule.run_time` (default 09:35 ET, weekdays only). In `approve` mode it sends a trade plan email and waits for approval; in `auto` mode it executes picks above `auto_execute_prob` after all account-level gates pass. |
| `monitor.py` | Checks stop-loss and gamma/theta triggers every `monitor_schedule.run_interval_minutes` (default 15 min) during market hours. Sends a position-closed email when an exit fires. Sends an EOD risk-report email on the first cycle after market close each trading day. |
| `dashboard.py` | Runs Flask with no auto-reloader and suppresses console noise — suitable for long-running background processes. |

### Step 1: Set Environment Variables

Never put credentials in `config.json`. Use environment variables instead:

```bash
# Alpaca API credentials
export ALPACA_API_KEY=your_api_key_here
export ALPACA_API_SECRET=your_api_secret_here
export ALPACA_PAPER=true          # true = paper trading

# Gmail app password for email notifications
export OPTIONWHEEL_EMAIL_PASSWORD="xxxx xxxx xxxx xxxx"
```

> **Gmail App Password**: Go to [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords), create an app password for "Mail", and paste the 16-character code above. Enable IMAP in Gmail settings (`Settings → See all settings → Forwarding and POP/IMAP → Enable IMAP`).

#### Windows (PowerShell)

```powershell
$env:ALPACA_API_KEY      = "your_api_key_here"
$env:ALPACA_API_SECRET   = "your_api_secret_here"
$env:ALPACA_PAPER        = "true"
$env:OPTIONWHEEL_EMAIL_PASSWORD = "xxxx xxxx xxxx xxxx"
```

### Step 2: Configure `config.json`

```json
"email": {
    "smtp_host": "smtp.gmail.com",
    "smtp_port": 587,
    "imap_host": "imap.gmail.com",
    "imap_port": 993,
    "from_addr": "you@gmail.com",
    "to_addr":   "you@gmail.com",
    "app_password": "",
    "approval_timeout_seconds": 21600,
    "approval_poll_interval_seconds": 15
},
"schedule": {
    "run_time":      "09:35",
    "timezone":      "US/Eastern",
    "weekdays_only": true
},
"monitor_schedule": {
    "run_interval_minutes": 15,
    "market_open":          "09:30",
    "market_close":         "16:00",
    "timezone":             "US/Eastern",
    "weekdays_only":        true
}
```

> Leave `app_password` blank — the agent reads `OPTIONWHEEL_EMAIL_PASSWORD` from the environment automatically.

### Step 3: Start All Three Processes

#### macOS / Linux using `start.sh`

```bash
# Live auto mode: agent uses --daemon --mode auto --live
./start.sh

# Dry-run auto mode: scans and simulates orders without reaching Alpaca
./start.sh --dry-run
```

#### macOS / Linux (nohup)

```bash
# Activate virtual environment
source .venv/bin/activate

# Agent — scans once per day, auto-executes picks above auto_execute_prob
nohup python agent.py --daemon --mode auto --live >> logs/agent.log 2>&1 &

# Position monitor — stop-loss checks every 15 min during market hours
nohup python monitor.py --daemon --live >> logs/monitor.log 2>&1 &

# Dashboard — browse trades at http://localhost:5050
nohup python dashboard.py --daemon --port 5050 >> logs/dashboard.log 2>&1 &

echo "All three processes started. Logs in logs/"
```

#### macOS / Linux (screen)

```bash
source .venv/bin/activate

# Start each in its own named screen session
screen -dmS agent   bash -c "source .venv/bin/activate && python agent.py --daemon --mode auto --live"
screen -dmS monitor bash -c "source .venv/bin/activate && python monitor.py --daemon --live"
screen -dmS dash    bash -c "source .venv/bin/activate && python dashboard.py --daemon --port 5050"

# Attach to a session to check output
screen -r agent

# List all sessions
screen -ls
```

#### Windows (PowerShell, background jobs)

```powershell
# Start all three as background jobs
$agent   = Start-Job { cd $using:PWD; .\.venv\Scripts\python.exe agent.py --daemon --mode auto --live 2>&1 | Tee-Object logs\agent.log }
$monitor = Start-Job { cd $using:PWD; .\.venv\Scripts\python.exe monitor.py --daemon --live 2>&1 | Tee-Object logs\monitor.log }
$dash    = Start-Job { cd $using:PWD; .\.venv\Scripts\python.exe dashboard.py --daemon --port 5050 2>&1 | Tee-Object logs\dashboard.log }

# Check status
Get-Job

# View output
Receive-Job $agent -Keep
```

### Step 4: Approve Mode Email Workflow

When the agent is launched with `--mode approve`, the workflow is email-based:

1. **Each morning** (at the configured `run_time`) the agent emails you a numbered trade plan:
   ```
   OptionWheel Trade Plan — 2026-03-15
   ┌────────────────────────────────────────────────────────────────────┐
   │  Reply to this email to approve picks:                            │
   │    a or all   → approve all picks                                 │
   │    n or none  → skip all picks                                    │
   │    1,3        → approve picks #1 and #3                           │
   │    2-4        → approve picks #2 through #4                       │
   └────────────────────────────────────────────────────────────────────┘
   #1  PCS  AAPL  2026-04-17  credit=$0.45  capital=$500  prob=82%
   #2  IC   MSFT  2026-04-17  credit=$0.85  capital=$500  prob=79%
   ...
   ```

2. **Reply to the email** with your approval choice (e.g. `1,3` or `all`). Quoted lines are stripped — just reply naturally.

3. **Trade execution emails** arrive for each approved pick with full details (strategy, legs, premium, order ID).

4. **Position closed emails** arrive when a stop-loss or gamma/theta trigger fires, including the P&L and risk metrics at the time of exit.

> **Timeout**: If no reply arrives within `approval_timeout_seconds` (default 6 h), the agent skips the cycle and logs a warning. It will try again the next trading day.

When launched with `--mode auto` (the `start.sh` default), the agent still sends execution and close notifications, but it does not wait for an approval reply before placing qualifying trades.

### Checking Logs

```bash
# Tail the agent log
tail -f logs/agent.log

# Tail the monitor log
tail -f logs/monitor.log

# Search for errors
grep -i error logs/agent.log
grep -i "position closed" logs/monitor.log
```

---

## Manual Position Management

The agent can inspect and close open positions directly from the command line without running a full scan. All commands default to **dry-run** — append `--live` only when you intend to send real broker orders.

### Position Commands

| Command | Description |
|---|---|
| `--list-open` | Print all open positions (ID, strategy, symbol, expiry, entry premium) and exit |
| `--close ID [ID …]` | Close one or more positions by trade ID |
| `--close-all` | Close **every** currently open position |

### Examples

```bash
# See all open positions:
python agent.py --list-open

# Close a single position (dry-run — no broker order):
python agent.py --close 7

# Close multiple positions (dry-run):
python agent.py --close 7 12 15

# Close positions with a real broker order:
python agent.py --close 7 12 --live

# Close everything (dry-run):
python agent.py --close-all

# Close everything with real orders (prompts for confirmation):
python agent.py --close-all --live
```

### How It Works

1. Each position is priced live from the yfinance option chain to determine a limit price.
2. `AlpacaExecutor.execute_close_position()` sends a buy-to-close multi-leg order via Alpaca (or prints `[DRY RUN]` without sending anything).
3. The trade is immediately marked `CLOSED` in `data/trades.db` with the realised P&L.
4. `--live` mode shows a confirmation prompt before sending real orders for `--close-all`.

You can also close positions from the **Dashboard** — see the [Open Positions tab](#dashboard).

---

## Sentiment Analysis

The sentiment engine runs on every ticker at scan time and adjusts two things simultaneously: the **delta ceilings** for individual picks and the **number of picks** allocated between PCS and CCS for the whole period.

### Signals

Three signals are combined into a composite score in `[-1, +1]`:

| Signal | Weight | Bullish condition | Bearish condition |
|---|---|---|---|
| RSI-14 | 35% | RSI > 60 | RSI < 40 |
| Price vs SMA-20 | 35% | Price above SMA (up to +5% = max score) | Price below SMA |
| 5-day momentum | 30% | Positive 5-day return | Negative 5-day return |

### Layer 1: Per-ticker delta ceiling skew

Once the composite score exceeds `bull_threshold` or `bear_threshold` (default 0.20), each ticker's max_delta ceiling is shifted by `skew_factor × strength` (clamped to `±max_skew`):

| Sentiment | PCS max_delta | CCS max_delta |
|---|---|---|
| **BULL** | ↑ raised — allows more put premium (puts safer in rising market) | ↓ lowered — tighter calls (calls riskier in rising market) |
| **BEAR** | ↓ lowered | ↑ raised |
| **NEUTRAL** | unchanged | unchanged |

### Layer 2: Per-period pick-count allocation

After scanning all tickers, the mean signed sentiment score across all candidates sets the PCS/CCS quota for the period (`top_n_skew_factor` default 0.50):

| Aggregate score | PCS picks (base 10) | CCS picks (base 10) |
|---|---|---|
| `+1.0` (strong BULL) | **15** | **5** |
| `+0.50` | 12 | 8 |
| `0.0` (NEUTRAL) | 10 | 10 |
| `−0.50` | 8 | 12 |
| `−1.0` (strong BEAR) | **5** | **15** |

IC, IFLY, and other strategies always receive the base `top_n_picks` quota regardless of sentiment.

### Disabling sentiment

```bash
# Agent
python agent.py --no-sentiment
```

Or set `"sentiment": { "enabled": false }` in `config.json`.

---

## Risk Management

Risk management has two jobs:

1. **Before entry**, the agent checks whether the new pick would make the whole account too exposed.
2. **After entry**, the position monitor checks whether an existing position should be closed.

The position monitor runs **at the start of every agent cycle**, before scanning for new picks. It checks every open position against two independent exit triggers and closes any that breach either condition.

Two layers work together: the **stop-loss** reacts to realised losses; the **gamma/theta risk trigger** acts predictively, detecting when a position's risk profile has deteriorated *before* the loss fully materialises.

---

## Portfolio-Level Pre-Trade Risk Gates

Before opening new positions, the agent reconciles the local database with Alpaca, then evaluates the portfolio as one book of risk. The local SQLite database is the system of record for trade lifecycle and P&L; Alpaca is the broker truth for live positions, order state, available quantity, and current market snapshots. The reconciler keeps those views aligned before the scanner allocates more capital.

The pre-trade gates include both fully open positions and `PENDING_CLOSE` positions. This matters because a close order can reserve option quantity at Alpaca before the database row is fully closed; treating that trade as already gone can create duplicate orders or overstated available capital.

### Gates

| Gate | Default | What it protects |
|---|---:|---|
| Max-loss multiple | `30.0×` premium | Rejects spreads whose max loss is too large relative to the collected credit. |
| Put exposure cap | Greater of `50%` of capital budget or `$1,500` | Limits gross PCS / put-side max loss concentration. |
| Call exposure cap | Greater of `50%` of capital budget or `$1,500` | Limits gross CCS / call-side max loss concentration. |
| Regime filter | `GREEN` normal, `YELLOW` 65% qty / 70% top-N, `ORANGE` 30% qty / 35% top-N, `RED` pause | Throttles new short-premium risk when VIX/trend/realized-vol conditions turn hostile. |
| Portfolio stress loss | Greater of `5%` of capital budget or `$500` | Simulates combined portfolio P&L under `±1%`, `±2%`, `±3%`, and `±5%` spot shocks plus a `+10` volatility-point shock. |
| Symbol stress loss | Greater of `2%` of capital budget or `$250` | Applies the same stress model per symbol so one ETF cannot dominate tail risk. |
| Complex-spread quantity cap | IC/IFLY capped to `1` when one contract uses ≥50% of symbol stress cap | Reduces QQQ/SPY-style condor concentration during higher-stress setups. |
| Near-expiry stress | Greater of `0.25%` of capital budget or `$250` | Limits concentrated stress loss in positions with `DTE <= 2`. |
| Gamma loss / daily theta | Warning only | Displays whether a 1% gamma loss overwhelms daily theta, but does not block picks by default. |
| Expiry bucket cap | Disabled | Can be enabled to cap stress loss by expiry, but is off by default to avoid over-constraining index-only scans. |
| Dynamic width | `$15` max | Allows enough credit on index ETF spreads while preventing very wide default strikes. |

The stress model nets signed delta, gamma, theta, and vega across the whole book. That means PCS and CCS on the same symbol can offset directional delta in the stress calculation. The offset is not treated as free risk: both spreads are still short gamma, so a same-symbol put/call mix can reduce directional exposure while still increasing convexity and gap risk.

### Scanner Strictness Check

The current defaults were loosened after a scanner strictness analysis on May 3, 2026. With the prior `6×` max-loss-multiple gate, the index-only scanner found 86 raw PCS candidates across SPY, QQQ, IWM, and DIA but allowed **0 final picks**. The observed max-loss multiples had a minimum around `14.82×` and a median around `64.79×`, so the old hard gate was filtering out every far-OTM index spread before the portfolio stress gate could do the more useful account-level work.

The new default of `30×`, plus 5% directional caps and the hard portfolio stress cap, intentionally allows more reasonable candidates through while still rejecting trades that would make the account fragile under gap-style moves.

---

## Stop-Loss Monitor

### Rules

Two guards protect every open position. For spreads (PCS / CCS / IC / IFLY), the **width-relative guard** is primary. For non-spread positions (CSP, CC, STRANGLE) the **premium-multiplier** applies.

#### Width-relative guard (spreads)

```
If unrealised loss > stop_loss_max_loss_pct × (spread_width − entry_premium) → close
```

Default `stop_loss_max_loss_pct = 0.80`, meaning: exit when 80% of the maximum possible loss (spread width minus credit received) has been realised.

**Why this matters:** A fixed `2× premium` multiplier fires at only `2 × $0.40 = $0.80` above entry on a low-premium far-OTM spread — that's 8% of a $10-wide spread. Any bid/ask wiggle can trigger it. The width-relative guard fires at `0.80 × $9.60 = $7.68` above entry, giving the position room proportional to its actual capital at risk regardless of how much premium was collected.

#### Premium-multiplier guard (non-spreads / fallback)

```
If current cost-to-close > (1 + stop_loss_multiplier) × entry_premium → close
```

Default multiplier = **2.0**. Applies to CSP and CC positions where there is no defined spread width. Also applies to any spread where `stop_loss_max_loss_pct` is set to `null` in config.

**Example (PCS):** $10-wide spread, collected $0.40 credit. Width-relative guard fires when mark reaches `$0.40 + 0.80 × $9.60 = $8.08` (80% of max loss). The spread would need to move nearly fully in-the-money to trigger.

### How it works (live agent)

1. Fetches current option prices via Alpaca (falls back to yfinance) for each open position
2. Computes the mark-to-market cost-to-close (bid/ask mid of the spread)
3. If the threshold is breached, submits a buy-to-close multi-leg order via Alpaca
4. Records the exit P&L in `data/trades.db`

### Configuration

```json
"risk_parameters": {
  "stop_loss_multiplier": 2.0,
  "stop_loss_max_loss_pct": 0.80
}
```

Set `"stop_loss_max_loss_pct": null` to disable the width-relative guard and revert to the legacy multiplier-only behaviour.

---

## Gamma / Theta Risk Trigger

The gamma risk trigger is a **predictive** early-exit mechanism. While the stop-loss reacts after a loss has occurred, this trigger closes a position when its *risk-to-reward profile* deteriorates — even while the position is still marginally profitable — to lock in remaining premium before a gamma spike wipes it out.

### The Core Idea: Why Gamma Kills Short-Premium Sellers

When you sell an option, you collect **theta** (time decay) as daily income. But you are simultaneously **short gamma**, meaning large moves in the underlying hurt you *more than linearly*. Near expiry, if your short strike is approached by the stock price, gamma explodes — a single bad day can erase many days of theta income.

The trigger monitors whether gamma is outpacing your theta income using the **gamma/theta ratio**:

```
gamma_theta_ratio  =  |net_gamma|  /  |net_theta_per_day|
```

A ratio of **1.5** means you are carrying 1.5× more gamma risk than your daily theta income compensates. At this point, one adverse move erases 1.5 days of earnings — the trade is no longer paying you fairly for the risk.

### Risk Score

The full composite risk score amplifies the ratio when the short leg is moving in-the-money:

```
delta_penalty  =  max(0,  |short_delta| − 0.15)  /  0.15
risk_score     =  gamma_theta_ratio  ×  (1  +  delta_penalty)
```

| Short delta | Delta penalty | risk_score multiplier |
|---|---|---|
| 0.10 (safely OTM) | 0 | 1.0× |
| 0.20 | 0.33 | 1.33× |
| 0.30 | 1.0 | 2.0× |
| 0.45 (dangerous) | 2.0 | 3.0× |

Delta is the most direct measure of expiry risk — a short put with delta = −0.30 has roughly a 30% chance of expiring in-the-money. The penalty ensures the risk score rises sharply when the position is both high-gamma *and* directionally threatened.

### Trigger Conditions

All three of the following must be satisfied for the trigger to fire:

```
gamma_theta_ratio  >=  gamma_theta_ratio_threshold   (default 1.5)
|short_delta|      >=  min_delta_to_trigger           (default 0.15)
profit_captured    >=  min_profit_captured_pct        (default 25%)
                    OR  |short_delta|  >=  urgent_delta_threshold  (default 0.30)
```

**Condition 1 — gamma/theta ratio**: The position's gamma risk must exceed the threshold, indicating the trade is no longer compensating you fairly.

**Condition 2 — minimum delta**: Gates the trigger so it only fires when the short leg has actually moved toward ATM. A high ratio on a deeply OTM position (delta = 0.05) is not actionable — the position is still very safe.

**Condition 3 — profit captured (with urgent bypass)**:

- **Normal mode** (`|short_delta| < 0.30`): require that at least 25% of the entry premium has been captured before allowing an early close. This prevents the trigger from closing a freshly opened position that hasn't had time to decay.
- **Urgent bypass** (`|short_delta| >= 0.30`): if the short leg has drifted to 30 delta (30% ITM probability), close regardless of profit captured — the priority shifts from locking in gains to stopping the bleeding.

### Example: How It Saves Premium

A CSP entered with 7 DTE, 0.10 delta, $0.40 credit:

| Day | DTE | Short Δ | Mark | Unrealised P&L | γ/θ ratio | Action |
|-----|-----|---------|------|----------------|-----------|--------|
| 0 | 7 | 0.10 | $0.40 | $0.00 | 0.8 | Hold |
| 2 | 5 | 0.14 | $0.28 | +$0.12 (30%) | 1.2 | Hold |
| 4 | 3 | 0.22 | $0.25 | +$0.15 (37%) | **1.7** | **CLOSE** — gamma trigger |
| 6 | 1 | 0.38 | $0.35 | +$0.05 (12%) | 4.2 | (stop-loss would fire soon) |

Without the gamma trigger, you'd hold through day 6 hoping for $0.05 more, but any overnight news could push the stock through your strike. The trigger on day 4 captures $0.15 (37.5% of the $0.40) with the position still safely OTM, before the final-days gamma spike makes the trade treacherous.

### What the Greeks Mean for Spreads

For multi-leg strategies, the Greeks are netted across all legs:

| Strategy | net_gamma | net_theta | Interpretation |
|---|---|---|---|
| **CSP** (short put) | negative | positive | Gamma hurts, theta helps — typical short-premium profile |
| **PCS** (short + long put) | less negative than CSP | less positive | Long put reduces both gamma risk and theta income |
| **IC** (short put + short call) | most negative | most positive | Highest gamma risk; highest theta income from two short legs |

The long legs in a spread (PCS, IC, IFLY) reduce gamma exposure — this is why spreads are safer than naked options near expiry even though they collect less premium.

### Configuration

```json
"risk_parameters": {
  "gamma_risk": {
    "enabled": true,
    "gamma_theta_ratio_threshold": 1.5,
    "min_delta_to_trigger": 0.15,
    "min_profit_captured_pct": 0.25,
    "urgent_delta_threshold": 0.30
  }
}
```

| Key | Default | Description |
|---|---|---|
| `enabled` | `true` | Set `false` to disable the gamma trigger entirely (stop-loss still active) |
| `gamma_theta_ratio_threshold` | `1.5` | Close when `\|net_gamma\| / \|net_theta_per_day\| >= this`. Lower = more aggressive protection |
| `min_delta_to_trigger` | `0.15` | Only fire when the short leg's absolute delta is at least this value — prevents triggering on safely OTM positions |
| `min_profit_captured_pct` | `0.25` | Require this fraction of entry premium to be already earned before allowing early close |
| `urgent_delta_threshold` | `0.30` | Bypass the profit requirement and close immediately if short delta exceeds this (position is in urgent danger) |

### Two-Trigger Summary

| Trigger | When it fires | Goal |
|---|---|---|
| **Stop-loss** | `loss > multiplier × entry_premium` | Hard floor — prevent catastrophic losses after a large move against you |
| **Gamma risk** | `γ/θ ratio + delta + profit conditions met` | Predictive exit — lock in remaining premium before gamma spike erases it |

---

## HFT Mode

HFT mode switches **all data fetching exclusively to Alpaca** — no yfinance calls anywhere. This eliminates yfinance rate limits and latency during the scanning and monitoring cycle, and gives you broker-native Greeks (delta, gamma, theta, vega) from Alpaca's option snapshots instead of Black-Scholes estimates.

### When to use it

| | Non-HFT (default) | HFT mode |
|---|---|---|
| **Data source** | yfinance (chain), Alpaca (fallback) | Alpaca only — no yfinance |
| **Greeks** | Black-Scholes (IV-derived) | Broker-native from Alpaca snapshots |
| **Position monitor** | Per-position API calls | Single batch prefetch for all positions |
| **On Alpaca failure** | Falls back to yfinance silently | Raises `RuntimeError`, logs the error, skips the position |
| **Best for** | Paper trading, casual use | Live trading, tight monitoring loops |

### Prerequisites

1. Valid Alpaca credentials with **options data** access (paper or live account).
2. `alpaca-py` installed — it is in `requirements.txt`, so `pip install -r requirements.txt` covers it.

### Configuration

Add `"hft_mode": true` at the top level of `config.json`, plus an optional `"hft"` tuning block:

```json
{
  "hft_mode": true,

  "hft": {
    "max_retries":               3,
    "retry_base_delay_seconds":  2.0
  },

  "alpaca": {
    "api_key":    "",
    "api_secret": "",
    "paper":      true
  }
}
```

Or set credentials via environment variables (recommended — keeps secrets out of `config.json`):

```bash
export ALPACA_API_KEY="your_key"
export ALPACA_API_SECRET="your_secret"
```

### HFT config reference

| Key | Default | Description |
|---|---|---|
| `hft_mode` | `false` | Set to `true` to enable HFT mode |
| `hft.max_retries` | `3` | Number of Alpaca retry attempts before raising `RuntimeError` and skipping the position |
| `hft.retry_base_delay_seconds` | `2.0` | Base delay between retries (exponential back-off: attempt N waits `base × 2^(N-1)` seconds) |

### What changes at runtime

- **Scanner**: fetches option chains and spot prices from Alpaca only; logs a hard error (instead of silently falling back) if Alpaca is unavailable.
- **Position monitor**: pre-fetches snapshots for *all* open positions in **two Alpaca API calls** (one for puts, one for calls) at the start of each cycle instead of one call per position — significantly faster with many open positions.
- **Greeks**: delta used in the gamma/theta risk trigger comes directly from Alpaca's snapshot rather than being estimated via Black-Scholes.
- **Startup validation**: if `hft_mode: true` but Alpaca credentials are missing or invalid, the agent refuses to start with a clear error message.

---

## Dashboard

A local web dashboard lets you browse, filter, and analyse every trade stored in `data/trades.db` — P&L over time, win rates, strategy breakdowns, and a searchable trade log.

```bash
python dashboard.py
```

Then open **http://localhost:5050** in your browser.

### Dashboard Options

| Flag | Default | Description |
|---|---|---|
| `--port PORT` | `5000` | Port to listen on |
| `--host HOST` | `127.0.0.1` | Host to bind to (use `0.0.0.0` for LAN access) |
| `--db PATH` | `data/trades.db` | Path to the SQLite trade database |
| `--config PATH` | `config.json` | Path to config file (used for close-order credentials) |
| `--debug` | `false` | Enable Flask debug / auto-reload |

### Dashboard Features

| Tab | What you see |
|---|---|
| **Overview** | Summary stat cards (total trades, total P&L, win rate, avg premium, best/worst trade), cumulative P&L line chart, trades-by-strategy doughnut chart, per-strategy breakdown cards |
| **Open Positions** | Live P&L for every open position — spot price, current option mark, unrealized P&L, P&L %, DTE. Refreshes on demand. Includes **Close** and **Close All** buttons. |
| **Trades** | Filterable, sortable, paginated trade log — filter by symbol, strategy, status, date range |
| **Analytics** | Time-filtered P&L, win rate, trade count, premium/cost/net P&L, capital deployed, capital remaining, and capital deployed per strategy. |
| **Risk Monitor** | Live Greeks per position plus portfolio gamma stress: scenario losses, daily theta, 1% loss/theta ratio, near-expiry stress, and expiry-bucket concentration. Per-position status badges (SAFE / WATCH / WARNING / TRIGGER / STOP_LOSS). Refreshes on demand. |

### Analytics Tab

The **Analytics** tab uses the database ledger as the source of truth for closed / expired P&L and capital usage. The time filter supports daily, weekly, quarterly, six-month, yearly, and all-time views. P&L is computed as premium received minus close cost for closed or expired options, so profit-take closes, stop-loss closes, and expirations all flow through the same calculation.

Capital analytics include total capital deployed, capital remaining against `max_capital_per_period`, and capital deployed per strategy. Losses are shown with a negative sign as well as styling, so the numbers remain clear without relying on color.

### Open Positions Tab

The **Open Positions** tab fetches live option-chain data from yfinance and shows:

- **Spot price** — current underlying price
- **Entry premium** — credit collected at entry
- **Current mark** — mid-market cost to close the spread now
- **Contracts** — number of option contracts in the position
- **Unrealized P&L** — `(entry_premium − current_mark) × 100 × contracts`
- **P&L %** — unrealized P&L as a percentage of entry premium
- **DTE** — days to expiry (colour-coded: red ≤ 5, yellow ≤ 14)

#### Closing Positions from the Dashboard

Every row has a **Close** button and there is a **Close All Positions** button in the action bar. Both show a confirmation modal with a **"Dry-run only"** checkbox (checked by default). Uncheck it to send real broker orders via Alpaca.

### Risk Monitor Tab

The **Risk Monitor** tab computes live Greeks for every open and pending-close position, then summarizes the whole portfolio through the same stress logic used by the pre-trade account gate. Click **Refresh risk data** to update all metrics in real time.

#### Portfolio Stress Summary

| Metric | What it shows |
|---|---|
| **Worst Stress Loss** | Worst simulated portfolio loss across the configured spot and IV shock scenarios |
| **1% Stress Loss** | Estimated loss from a 1% underlying move using net delta/gamma/vega |
| **Daily Theta** | Net daily theta income across open and pending-close positions |
| **1% Loss / Theta** | How many days of theta a 1% gamma-style loss can erase. Warning-only by default. |
| **Near-Expiry Stress** | Stress loss concentrated in positions with `DTE <= portfolio_gamma_risk.near_expiry_dte` |
| **Max Expiry Bucket** | Largest stress loss concentrated in one expiry date |
| **Max Symbol Stress** | Largest stress loss concentrated in one underlying symbol |
| **Stress Scenarios** | Per-shock loss table for `±1%`, `±2%`, `±3%`, and `±5%` moves |
| **Expiry Buckets** | Stress loss grouped by expiry date, useful for spotting same-week concentration |
| **Symbol Buckets** | Stress loss grouped by underlying symbol, useful for spotting QQQ/SPY-style concentration |

#### Columns

| Column | What it shows |
|---|---|
| **Short Δ** | Absolute delta of the short leg(s). Colour-coded: green < 0.10, yellow 0.10–0.20, orange 0.20–0.30, red > 0.30 |
| **Net Δ / Net Vega** | Signed portfolio contribution for the position. These feed the portfolio-level stress model. |
| **γ/θ Ratio** | `|net_gamma| / |net_theta_per_day|`. Green < 0.8, yellow 0.8–1.5, orange 1.5–2.0, red > 2.0 |
| **Risk Score** | `gamma_theta_ratio × (1 + delta_penalty)`. Shown as a colour-coded bar + number |
| **Θ/day ($)** | Net daily theta income per contract (×100). Positive = earning; negative = paying away |
| **Profit %** | Percentage of entry premium already secured (`(premium − mark) / premium`) |
| **SL Distance** | How far the current mark is from the stop-loss trigger, as % of premium. Negative = already past stop |
| **P&L** | Total unrealized P&L across all contracts |
| **Status** | Exit trigger evaluation (see below) |

#### Status Badges

| Badge | Meaning |
|---|---|
| **SAFE** | γ/θ < 60% of threshold and short Δ < 40% of min-delta — well within bounds |
| **WATCH** | Drifting toward thresholds — begin monitoring more closely |
| **WARNING** | Within 70% of the ratio or delta threshold — consider planning an exit |
| **TRIGGER** | Gamma risk trigger would fire right now — early exit recommended |
| **STOP LOSS** | Stop-loss threshold exceeded — close immediately |

#### Config Thresholds Display

The action bar shows a live pill strip of your active config values so you always know what thresholds are being evaluated against your positions: `SL mult · γ/θ threshold · Min Δ · Urgent Δ · Min profit`.

### Filtering Trades

Use the filter bar in the **Trades** tab:
- **Symbol** — partial match (e.g. `AAPL`, `NV`)
- **Strategy** — dropdown: IC, IFLY, PCS, CCS, CSP, …
- **Status** — EXECUTED, DRY_RUN, PENDING, CLOSED, CANCELLED
- **Date range** — from / to date pickers

---

## How It Works

1. **Universe** — The default config uses the ETF universe for index-ETF trading. `src/universe.py` can also fetch NASDAQ + NYSE/AMEX stocks filtered by `market_cap_min` (cached 24h), and `src/universe_indices.py` provides S&P 500 + NASDAQ-100 + Dow 30 constituents scraped from Wikipedia and cached in `data/index_cache.json`.

2. **Sentiment** — `src/sentiment.py` computes a composite score per ticker from RSI-14, price-vs-SMA-20, and 5-day momentum. BULL/BEAR labels and strength (0–1) are used to: (a) skew the per-ticker max_delta ceilings passed to each strategy scanner, and (b) adjust the PCS/CCS pick quotas for the whole period based on the aggregate mood.

3. **Risk monitor** — `monitor.py` checks every open trade in `data/trades.db` every 15 minutes during market hours using two independent triggers: (a) **Stop-loss** — closes any position whose mark-to-market loss exceeds `stop_loss_multiplier × entry_premium`; (b) **Gamma/theta risk** — uses `src/greeks.py` to compute live Greeks from the option chain's implied volatility, then closes any position whose gamma/theta ratio and short-leg delta indicate the risk profile has deteriorated beyond the configured thresholds, locking in remaining premium before a gamma spike erases it.

4. **Reconcile** — `src/position_reconciler.py` refreshes local trade state from Alpaca before new risk is opened. Open and pending-close rows stay in the capital and risk calculations until Alpaca confirms the position is gone.

5. **Scan** — `src/scanner.py` begins by batch-downloading 30-day close-price history for all tickers in a **single `yf.download()` request** (instead of N individual calls) to pre-populate a price/HV30 cache, and simultaneously pre-warms the sentiment cache. Only the per-expiry `option_chain()` calls — which cannot be batched — still hit yfinance individually, at up to 10 concurrent threads. An in-process option-chain cache (5-min TTL) amortises re-scans within the same session. On a 500-ticker universe this cuts scan time from ~5 min to ~1.5 min.

6. **Rank and account-gate** — `get_top_picks()` collects all candidates, applies the sentiment-driven PCS/CCS quotas, sorts by `score = (premium / spread_width) × prob_win²`, then the agent applies max-loss-multiple, directional exposure, capital budget, and portfolio gamma stress gates before presenting or executing picks.

7. **Approve / Execute** — In `approve` mode the plan table is shown and the user selects picks to execute. In `auto` mode, picks above `auto_execute_prob` are executed without prompting. In `scan-only` mode, picks are saved to `data/pending_picks.json`.

8. **Order** — `src/executor.py` translates each pick dict into the correct Alpaca multi-leg COMBO order (4-leg condors/butterflies, 2-leg spreads, 1-leg CSP/CC) using `PositionIntent.SELL_TO_OPEN`. Close orders use `PositionIntent.BUY_TO_CLOSE`.

9. **Log** — Every executed trade is recorded in a local SQLite database (`data/trades.db`) with leg strikes, status tracking, stop-loss exit price, close order tracking, and realised P&L.

---

## Project Structure

```
optionwheel/
├── agent.py                   # Main entry point — stop-loss check → scan → approve → execute
├── dashboard.py               # Flask web dashboard (positions, risk monitor, manual close)
├── config.json                # All user-configurable settings
├── .env.example               # Credential template (copy to .env — never committed)
├── LICENSE                    # Proprietary license — all rights reserved
├── requirements.txt           # Python dependencies
├── setup.bat                  # Windows install & launch script
├── setup.sh                   # macOS / Linux / Git Bash install & launch script
├── src/
│   ├── scanner.py             # Option scanning, B-S probability, all strategies, sentiment hooks
│   ├── executor.py            # AlpacaExecutor — places / closes orders via alpaca-py
│   ├── sentiment.py           # RSI + SMA + momentum sentiment engine; delta + top-N skew
│   ├── position_monitor.py    # PositionMonitor — stop-loss, profit-take, gamma/theta triggers
│   ├── position_reconciler.py # Reconciles PENDING_CLOSE rows against live Alpaca positions
│   ├── risk_service.py        # PositionRiskService — shared Greek + mark computation (HFT-aware)
│   ├── portfolio_risk.py      # PortfolioRiskService — account-level stress, exposure, concentration gates
│   ├── alpaca_data.py         # AlpacaDataClient — option chains, snapshots, greeks via Alpaca API
│   ├── notifier.py            # EmailNotifier — trade plan, execution, risk report via Gmail SMTP/IMAP
│   ├── greeks.py              # Black-Scholes Greeks (delta, gamma, theta, vega) + position risk score
│   ├── universe.py            # Dynamic NASDAQ/NYSE universe with market-cap filter
│   ├── universe_indices.py    # S&P 500 + NASDAQ-100 + Russell 1000 + Dow 30 index constituents
│   ├── database.py            # SQLite trade log (legs, stop-loss fields, P&L, close order tracking)
│   ├── utils.py               # Config loader, logger
│   ├── market_data/           # HFT-aware data layer
│   │   ├── adapter.py         # DataAdapter — unified data fetch (Alpaca snapshots or yfinance/chain)
│   │   └── base.py            # OptionChain dataclass and shared types
│   ├── notify/                # Email formatting helpers
│   │   ├── formatter.py       # Risk legend, position rows, HTML/text formatting
│   │   └── sender.py          # Low-level SMTP send with exponential back-off retry
│   ├── risk_rules/            # Modular risk classification pipeline
│   │   ├── pipeline.py        # Orchestrates all risk axes into a final risk level
│   │   ├── classify.py        # Three-axis classifier (stop proximity, profit, gamma/theta)
│   │   ├── stop_loss.py       # Stop-loss proximity rule
│   │   ├── profit_take.py     # Profit-capture rule (≥25% captured → SAFE)
│   │   ├── gamma_risk.py      # Gamma/theta ratio rule
│   │   ├── intrinsic_value.py # Intrinsic value / ITM detection
│   │   ├── leg_specs.py       # Per-strategy leg specification helpers
│   │   └── mark.py            # Mark price computation (mid / conservative ask-bid)
│   ├── scan_filters/          # Pluggable scanner pre-filters
│   │   ├── atr.py             # ATR / volatility filter
│   │   ├── earnings.py        # Earnings blackout filter
│   │   ├── liquidity.py       # Bid-ask spread and OI liquidity filter
│   │   ├── otm.py             # OTM distance filter
│   │   └── probability.py     # B-S probability-of-profit filter
│   └── scan_strategies/       # Per-strategy option chain scanning
│       ├── csp.py             # Cash-secured put
│       ├── covered_call.py    # Covered call
│       ├── spreads.py         # PCS / CCS credit spreads
│       ├── iron_condor.py     # Iron condor
│       ├── iron_butterfly.py  # Iron butterfly
│       └── strangle.py        # Short strangle
├── data/
│   ├── universe_cache.json    # NASDAQ/NYSE ticker cache (24-hour TTL)
│   ├── index_cache.json       # Index constituent cache (24-hour TTL)
│   ├── pending_picks.json     # Picks saved in scan-only mode
│   └── trades.db              # SQLite trade database
└── tests/                     # 1074 passing unit tests + 1 skipped integration-style test
    ├── conftest.py                  # Shared fixtures
    ├── test_agent_account_safety.py # Account-level gates: capital, exposure, max-loss, pending-close
    ├── test_alpaca_data.py          # AlpacaDataClient — chains, snapshots, Greeks
    ├── test_alpaca_scanner.py       # Scanner with Alpaca data source
    ├── test_credit_spreads.py
    ├── test_csp.py
    ├── test_dashboard.py            # Dashboard APIs, analytics, P&L, portfolio risk monitor
    ├── test_data_adapter.py         # DataAdapter HFT vs non-HFT paths, no-options cache
    ├── test_database.py
    ├── test_executor.py
    ├── test_greeks.py
    ├── test_new_strategies.py       # IC, IFLY, STRANGLE, CC, executor
    ├── test_notifier.py             # Email formatting, risk classification, SMTP retry
    ├── test_position_monitor.py     # PositionMonitor lifecycle, dry-run, live mode
    ├── test_portfolio_risk.py       # Portfolio stress scenarios, warning-only gates, expiry buckets
    ├── test_reconciler.py           # PENDING_CLOSE reconciliation, crash recovery
    ├── test_risk_service.py         # PositionRiskService Greek + mark computation
    ├── test_scanner.py
    ├── test_scanner_extended.py
    ├── test_scanner_filters.py      # Pluggable scan filter unit tests
    ├── test_scanner_get_top_picks.py # Blacklist, top-N selection, HFT prefetch
    ├── test_sentiment.py            # RSI, delta skew, disabled mode
    ├── test_stop_loss.py            # Stop-loss + profit-take triggers, two-phase close
    ├── test_universe.py
    ├── test_utils.py
    └── test_vix_filter.py
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Expected: **1074 tests pass, 1 test skipped**.
