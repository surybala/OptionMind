#!/usr/bin/env bash
# start.sh — Launch all three OptionWheel daemons
# Usage: bash start.sh [--dry-run]
#
# Flags:
#   --dry-run   Start agent and monitor in dry-run mode (no live orders).
#               Dashboard always starts without --live (it has its own toggle).
#
# Logs are written to ./logs/{agent,monitor,dashboard}.log with automatic
# rotation (10 MB per file, 5 backups kept → up to ~55 MB per process).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DRY_RUN=false
for arg in "$@"; do
  [[ "$arg" == "--dry-run" ]] && DRY_RUN=true
done

LIVE_FLAG=""
if [[ "$DRY_RUN" == false ]]; then
  LIVE_FLAG="--live"
fi

# Activate virtual environment
source .venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting OptionWheel daemons (dry_run=$DRY_RUN, agent_mode=auto)..."

# Agent — scans once per day and auto-executes picks above auto_execute_prob
nohup python agent.py --daemon --mode auto $LIVE_FLAG --log-file logs/agent.log > /dev/null 2>&1 &
echo "  agent.py     PID $! → logs/agent.log (rotated at 10 MB, 5 backups)"

# Position monitor — stop-loss checks every 15 min during market hours
nohup python monitor.py --daemon $LIVE_FLAG --log-file logs/monitor.log > /dev/null 2>&1 &
echo "  monitor.py   PID $! → logs/monitor.log (rotated at 10 MB, 5 backups)"

# Dashboard — browse trades at http://localhost:5000
nohup python dashboard.py --daemon --log-file logs/dashboard.log > /dev/null 2>&1 &
echo "  dashboard.py PID $! → logs/dashboard.log (rotated at 10 MB, 5 backups)"

echo ""
echo "All daemons started. To tail logs:"
echo "  tail -f logs/agent.log logs/monitor.log logs/dashboard.log"
