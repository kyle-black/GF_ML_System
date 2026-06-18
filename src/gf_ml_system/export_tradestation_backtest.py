from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from gf_ml_system.config import DEFAULT_OUTPUT_DIR
else:
    from .config import DEFAULT_OUTPUT_DIR


EASYLANGUAGE_COLUMNS = [
    "el_date",
    "ts_date",
    "date_iso",
    "prob_up_close",
    "prob_down_close",
    "prob_up_0_25_atr",
    "prob_down_0_25_atr",
    "prob_up_0_5_atr",
    "prob_down_0_5_atr",
    "prob_up_0_75_atr",
    "prob_down_0_75_atr",
    "prob_up_1_atr",
    "prob_down_1_atr",
    "highest_up_atr_probability",
    "highest_down_atr_probability",
]
UP_PROBABILITY_COLUMNS = [
    "prob_up_close",
    "prob_up_0_25_atr",
    "prob_up_0_5_atr",
    "prob_up_0_75_atr",
    "prob_up_1_atr",
]
DOWN_PROBABILITY_COLUMNS = [
    "prob_down_close",
    "prob_down_0_25_atr",
    "prob_down_0_5_atr",
    "prob_down_0_75_atr",
    "prob_down_1_atr",
]


def _el_date(date_values: pd.Series) -> pd.Series:
    return (
        (date_values.dt.year - 1900) * 10000
        + date_values.dt.month * 100
        + date_values.dt.day
    ).astype(int)


def _probability_column(frame: pd.DataFrame, *columns: str) -> pd.Series:
    for column in columns:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=frame.index)


def _normalize_side_policy(side_policy: str) -> str:
    normalized = side_policy.strip().lower().replace("-", "_")
    if normalized not in {"both", "long_only", "short_only"}:
        raise ValueError("side_policy must be one of: both, long-only, short-only")
    return normalized


def _apply_side_policy(export: pd.DataFrame, side_policy: str) -> pd.DataFrame:
    normalized = _normalize_side_policy(side_policy)
    if normalized == "long_only":
        export.loc[:, DOWN_PROBABILITY_COLUMNS] = 0.0
    elif normalized == "short_only":
        export.loc[:, UP_PROBABILITY_COLUMNS] = 0.0
    return export


def build_easylanguage_backtest_export(
    walk_forward_predictions: pd.DataFrame,
    *,
    side_policy: str = "both",
) -> pd.DataFrame:
    source = walk_forward_predictions.copy()
    if "signal_date" not in source.columns:
        raise ValueError("walk_forward_predictions must include a signal_date column.")

    date_values = pd.to_datetime(source["signal_date"], errors="coerce")
    source = source.loc[date_values.notna()].copy()
    date_values = pd.to_datetime(source["signal_date"], errors="coerce")
    if source.empty:
        return pd.DataFrame(columns=EASYLANGUAGE_COLUMNS)

    export = pd.DataFrame(index=source.index)
    export["el_date"] = _el_date(date_values)
    export["ts_date"] = date_values.dt.strftime("%Y%m%d").astype(int)
    export["date_iso"] = date_values.dt.strftime("%Y-%m-%d")
    export["prob_up_close"] = _probability_column(source, "prob_up_close")
    export["prob_down_close"] = _probability_column(source, "prob_down_close")
    export["prob_up_0_25_atr"] = _probability_column(source, "prob_up_0_25atr", "prob_up_oc_0_25atr")
    export["prob_down_0_25_atr"] = _probability_column(source, "prob_down_0_25atr", "prob_down_oc_0_25atr")
    export["prob_up_0_5_atr"] = _probability_column(source, "prob_up_0_5atr", "prob_up_oc_0_5atr")
    export["prob_down_0_5_atr"] = _probability_column(source, "prob_down_0_5atr", "prob_down_oc_0_5atr")
    export["prob_up_0_75_atr"] = 0.0
    export["prob_down_0_75_atr"] = 0.0
    export["prob_up_1_atr"] = 0.0
    export["prob_down_1_atr"] = 0.0
    export = _apply_side_policy(export, side_policy)

    up_columns = [
        "prob_up_0_25_atr",
        "prob_up_0_5_atr",
        "prob_up_0_75_atr",
        "prob_up_1_atr",
    ]
    down_columns = [
        "prob_down_0_25_atr",
        "prob_down_0_5_atr",
        "prob_down_0_75_atr",
        "prob_down_1_atr",
    ]
    export["highest_up_atr_probability"] = export[up_columns].max(axis=1)
    export["highest_down_atr_probability"] = export[down_columns].max(axis=1)
    return export.loc[:, EASYLANGUAGE_COLUMNS].sort_values("el_date").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export only walk-forward testing predictions to the 15-column "
            "EasyLanguage CSV format used by the TradeStation backtest strategy."
        ),
    )
    parser.add_argument(
        "--run-name",
        default="es_next_day_atr_tes_intermarket_xgboost",
        help="Run folder name under data/ml/tes_intermarket.",
    )
    parser.add_argument(
        "--run-root",
        default="",
        help="Explicit run folder. Overrides --run-name when provided.",
    )
    parser.add_argument(
        "--input",
        default="",
        help="Explicit walk_forward_predictions.csv path. Overrides --run-root/--run-name.",
    )
    parser.add_argument(
        "--output",
        default="",
        help=(
            "Output CSV path. Defaults to <run-root>/tradestation_easylanguage_backtest.csv, "
            "or adds a side-policy suffix for long-only/short-only exports."
        ),
    )
    parser.add_argument(
        "--side-policy",
        default="both",
        choices=("both", "long-only", "short-only"),
        help="Zero one side's probabilities for TradeStation testing without changing model predictions.",
    )
    parser.add_argument(
        "--with-header",
        action="store_true",
        help="Include a header row. The pasted EasyLanguage strategy does not require it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root) if args.run_root else DEFAULT_OUTPUT_DIR / args.run_name
    input_path = Path(args.input) if args.input else run_root / "walk_forward_predictions.csv"
    side_policy = _normalize_side_policy(args.side_policy)
    if args.output:
        output_path = Path(args.output)
    else:
        suffix = "" if side_policy == "both" else f"_{side_policy}"
        output_path = run_root / f"tradestation_easylanguage_backtest{suffix}.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Walk-forward prediction file not found: {input_path}")

    export = build_easylanguage_backtest_export(pd.read_csv(input_path), side_policy=side_policy)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(output_path, index=False, header=args.with_header)
    print(f"Wrote {len(export)} TradeStation testing rows to {output_path}")


if __name__ == "__main__":
    main()
