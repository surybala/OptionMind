#!/usr/bin/env bash
# setup.sh — install dependencies, validate config, run tests, launch OptionMind
# Works on Linux, macOS, and Windows Git Bash.
set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC}   $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }
info() { echo -e "${CYAN}[INFO]${NC}  $*"; }

echo "============================================"
echo "   OptionMind | Setup & Launch"
echo "============================================"
echo

# ── 1. Locate Python 3.10+ ───────────────────────────────────────────────────
PYTHON=""
for cmd in python3 python py; do
    if command -v "$cmd" &>/dev/null; then
        if "$cmd" -c "import sys; exit(0 if sys.version_info>=(3,10) else 1)" 2>/dev/null; then
            PYTHON="$cmd"
            break
        fi
    fi
done

# Fallback: common Windows paths when running under Git Bash
if [[ -z "$PYTHON" ]]; then
    WIN_PATHS=(
        "${LOCALAPPDATA:-}/Python/bin/python.exe"
        "${LOCALAPPDATA:-}/Programs/Python/Python313/python.exe"
        "${LOCALAPPDATA:-}/Programs/Python/Python312/python.exe"
        "${LOCALAPPDATA:-}/Programs/Python/Python311/python.exe"
        "${LOCALAPPDATA:-}/Programs/Python/Python310/python.exe"
        "/c/Python312/python.exe"
    )
    for p in "${WIN_PATHS[@]}"; do
        if [[ -x "$p" ]] && "$p" -c "import sys; exit(0 if sys.version_info>=(3,10) else 1)" 2>/dev/null; then
            PYTHON="$p"
            break
        fi
    done
fi

[[ -z "$PYTHON" ]] && err "Python 3.10+ not found. Download from https://python.org"
ok "Python: $PYTHON ($("$PYTHON" --version 2>&1))"
echo

# ── 2. Virtual environment ────────────────────────────────────────────────────
VENV=".venv"
if [[ ! -d "$VENV" ]]; then
    info "Creating virtual environment at $VENV ..."
    "$PYTHON" -m venv "$VENV"
    ok "Virtual environment created."
else
    ok "Virtual environment already exists."
fi
echo

# Activate (Git Bash uses Scripts/activate; Linux/macOS use bin/activate)
if [[ -f "$VENV/Scripts/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VENV/Scripts/activate"
else
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi

# ── 3. Install dependencies ───────────────────────────────────────────────────
info "Upgrading pip ..."
python -m pip install --upgrade pip --quiet

info "Installing dependencies from requirements.txt ..."
pip install -r requirements.txt
ok "All dependencies installed."
echo

# ── 4. Data directory ─────────────────────────────────────────────────────────
mkdir -p data
ok "data/ directory ready."
echo

# ── 5. Validate Alpaca credentials ───────────────────────────────────────────
info "Checking Alpaca credentials in config.json ..."
python - <<'PY'
import json, sys

try:
    cfg = json.load(open('config.json'))
except Exception as e:
    print(f"[ERROR] Cannot read config.json: {e}")
    sys.exit(1)

a      = cfg.get('alpaca', {})
key    = a.get('api_key',    '')
secret = a.get('api_secret', '')
paper  = a.get('paper', True)

missing = [name for name, val in [('api_key', key), ('api_secret', secret)] if not val]
if missing:
    print(f"[WARN]  Missing Alpaca credentials: {missing}")
    print("        Edit config.json and add your api_key and api_secret.")
    print("        The agent will still run but no live orders will be placed.")
else:
    mode = "PAPER" if paper else "LIVE ***"
    print(f"[OK]    Credentials present | mode: {mode}")
PY
echo

# ── 6. Test suite ─────────────────────────────────────────────────────────────
info "Running test suite to verify the installation ..."
python -m pytest tests/ -q --tb=short
echo

# ── 7. Launch ─────────────────────────────────────────────────────────────────
echo "============================================"
echo "   Launching OptionMind Agent"
echo "============================================"
echo
python agent.py

echo
ok "OptionMind finished."
