"""Run a saved shell training command from a JSON artifact."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a saved training command from a JSON artifact.")
    parser.add_argument("--input-json", required=True, help="Path to the JSON artifact.")
    parser.add_argument(
        "--command-key",
        default="training_command",
        help="JSON key containing the shell command to execute.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    command = payload.get(args.command_key)
    if not isinstance(command, str) or not command.strip():
        raise ValueError(f"Missing non-empty string command at key {args.command_key!r}")
    print(command)
    subprocess.run(
        command,
        shell=True,
        check=True,
        executable="/bin/zsh",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
