from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from gf_ml_system.config import DEFAULT_SYMBOLS, PROJECT_ROOT
else:
    from .config import DEFAULT_SYMBOLS, PROJECT_ROOT


DEFAULT_CONNECTOR_SCRIPT = Path(
    "/home/kyle-black/GaleForce_TrendEngine/src/trade_station_connect/src/get_bars2.py",
)
DEFAULT_CONNECTOR_PYTHON = Path("/home/kyle-black/GaleForce_TrendEngine/env/bin/python")


def _connector_supports_flag(connector_python: Path, connector_script: Path, flag: str) -> bool:
    result = subprocess.run(
        [str(connector_python), str(connector_script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    help_text = f"{result.stdout}\n{result.stderr}"
    return flag in help_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pull daily TradeStation futures bars using the existing GF connector style "
            "and write DP CSVs into this repo."
        ),
    )
    parser.add_argument("--futures", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--unit", default="Daily", choices=["Minute", "Daily", "Weekly", "Monthly"])
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--barsback", type=int, default=20000)
    parser.add_argument("--include-current-day", action="store_true")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "data" / "historical_data"))
    parser.add_argument("--connector-script", default=str(DEFAULT_CONNECTOR_SCRIPT))
    parser.add_argument(
        "--connector-python",
        default=str(DEFAULT_CONNECTOR_PYTHON if DEFAULT_CONNECTOR_PYTHON.exists() else Path(sys.executable)),
    )
    parser.add_argument(
        "--extra-connector-args",
        default="",
        help="Optional raw args passed through to get_bars2.py, for example '--session-template 103XC'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    connector_script = Path(args.connector_script).expanduser()
    if not connector_script.exists():
        raise FileNotFoundError(f"TradeStation connector script not found: {connector_script}")

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    command = [
        str(Path(args.connector_python).expanduser()),
        str(connector_script),
        "--futures",
        args.futures,
        "--unit",
        args.unit,
        "--interval",
        str(args.interval),
        "--barsback",
        str(args.barsback),
        "--output-root",
        str(output_root),
        "--no-dashboard-update",
    ]
    if args.include_current_day and _connector_supports_flag(
        Path(args.connector_python).expanduser(),
        connector_script,
        "--include-current-day",
    ):
        command.append("--include-current-day")
    elif args.include_current_day:
        print(
            "Connector does not expose --include-current-day; continuing with connector defaults.",
            file=sys.stderr,
        )
    if args.extra_connector_args:
        command.extend(shlex.split(args.extra_connector_args))

    printable = " ".join(shlex.quote(part) for part in command)
    print(printable)
    subprocess.run(command, check=True, cwd=connector_script.parents[3])


if __name__ == "__main__":
    main()
