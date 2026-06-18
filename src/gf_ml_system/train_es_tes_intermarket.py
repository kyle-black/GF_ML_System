from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from gf_ml_system.config import (
        DEFAULT_DATA_ROOTS,
        DEFAULT_INTRADAY_60M_ROOTS,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_SYMBOLS,
    )
    from gf_ml_system.data import find_latest_symbol_file, parse_data_roots, parse_file_map, parse_symbol_list
    from gf_ml_system.model import WalkForwardConfig, train_pipeline
else:
    from .config import DEFAULT_DATA_ROOTS, DEFAULT_INTRADAY_60M_ROOTS, DEFAULT_OUTPUT_DIR, DEFAULT_SYMBOLS
    from .data import find_latest_symbol_file, parse_data_roots, parse_file_map, parse_symbol_list
    from .model import WalkForwardConfig, train_pipeline


def _parse_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(token.strip()) for token in value.split(",") if token.strip())
    if not values:
        raise ValueError("At least one ATR multiplier is required.")
    return values


def _parse_optional_symbol_list(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    return parse_symbol_list(value)


def _extend_symbols_with_available_optionals(
    symbols: tuple[str, ...],
    optional_symbols: tuple[str, ...],
    *,
    data_roots: tuple[Path, ...],
    data_file_map: dict[str, Path],
) -> tuple[str, ...]:
    selected = list(symbols)
    selected_set = set(symbols)
    for symbol in optional_symbols:
        if symbol in selected_set:
            continue
        path = data_file_map.get(symbol)
        if path is not None and path.exists():
            selected.append(symbol)
            selected_set.add(symbol)
            continue
        try:
            find_latest_symbol_file(data_roots, symbol)
        except FileNotFoundError:
            print(f"Skipping optional symbol {symbol}: no daily DP file found.", file=sys.stderr)
            continue
        selected.append(symbol)
        selected_set.add(symbol)
    return tuple(selected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the first ES TES intermarket model: ES next-day ATR up/down labels, "
            "TES indicators from ES,NQ,TY,US,GC,SI, and 50-day walk-forward test blocks."
        ),
    )
    parser.add_argument("--run-name", default="es_next_day_atr_tes_intermarket")
    parser.add_argument("--target-symbol", default="ES")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument(
        "--optional-symbols",
        default="",
        help="Comma-separated symbols to include only when daily data is locally available, e.g. VX.",
    )
    parser.add_argument(
        "--data-roots",
        default=",".join(str(path) for path in DEFAULT_DATA_ROOTS),
        help="Comma-separated DP CSV roots. Each root should contain SYMBOL/DP-SYMBOL-MMDDYY-ED.csv files.",
    )
    parser.add_argument(
        "--data-file-map",
        default="",
        help="Optional explicit CSV files, e.g. ES=/tmp/es.csv,NQ=/tmp/nq.csv.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--start-date", default="2008-01-01")
    parser.add_argument("--end-date", default="")
    parser.add_argument(
        "--target-mode",
        default="atr_touch",
        choices=("atr_touch", "open_close", "open_close_atr", "combined"),
        help=(
            "atr_touch trains next-day high/low ATR-touch labels; open_close trains "
            "next-day open-to-close direction labels; open_close_atr trains "
            "thresholded next-day open-to-close ATR labels; combined trains all families."
        ),
    )
    parser.add_argument("--atr-window", type=int, default=10)
    parser.add_argument("--atr-multipliers", default="0.25,0.5")
    parser.add_argument("--min-train-rows", type=int, default=750)
    parser.add_argument(
        "--rolling-train-rows",
        type=int,
        default=750,
        help="Rows in each rolling train window. Use 0 for expanding walk-forward.",
    )
    parser.add_argument("--test-rows", type=int, default=50)
    parser.add_argument("--step-rows", type=int, default=50)
    parser.add_argument("--context-ffill-limit", type=int, default=2)
    parser.add_argument(
        "--include-enhanced-features",
        action="store_true",
        help="Add RTH session-shape, cross-market divergence, ATR distribution, regime, and daily-structure features.",
    )
    parser.add_argument(
        "--include-atr-distribution-features",
        action="store_true",
        help="Add rolling ATR distribution features without the full enhanced feature pack.",
    )
    parser.add_argument(
        "--include-intraday-60m",
        action="store_true",
        help=(
            "Add same-day 60-minute RTH features. Bars are filtered to the configured "
            "start/end times and collapsed to the last completed intraday bar per signal date."
        ),
    )
    parser.add_argument(
        "--intraday-60m-roots",
        default=",".join(str(path) for path in DEFAULT_INTRADAY_60M_ROOTS),
        help="Comma-separated 60-minute DP CSV roots. All DP files per symbol are merged by timestamp.",
    )
    parser.add_argument(
        "--intraday-60m-file-map",
        default="",
        help="Optional explicit 60-minute CSV files, e.g. ES=/tmp/es_60m.csv,NQ=/tmp/nq_60m.csv.",
    )
    parser.add_argument("--intraday-rth-start", default="10:00")
    parser.add_argument("--intraday-rth-end", default="16:00")
    parser.add_argument(
        "--allow-missing-intraday-symbols",
        action="store_true",
        help="Skip 60-minute features for symbols with daily data but no intraday DP files.",
    )
    parser.add_argument("--model-type", default="ridge", choices=("ridge", "xgboost"))
    parser.add_argument("--ridge-l2", type=float, default=10.0)
    parser.add_argument("--xgb-eta", type=float, default=0.05)
    parser.add_argument("--xgb-max-depth", type=int, default=3)
    parser.add_argument("--xgb-num-boost-round", type=int, default=120)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--xgb-min-child-weight", type=float, default=5.0)
    parser.add_argument("--xgb-reg-lambda", type=float, default=5.0)
    parser.add_argument("--xgb-reg-alpha", type=float, default=0.0)
    parser.add_argument("--xgb-device", default="cpu")
    parser.add_argument(
        "--max-folds",
        type=int,
        default=0,
        help="Debug limit for walk-forward folds. Use 0 for all folds.",
    )
    parser.add_argument(
        "--no-target-features",
        action="store_true",
        help="Use only non-target symbols as features. By default ES indicators are included too.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_roots = parse_data_roots(args.data_roots, DEFAULT_DATA_ROOTS)
    data_file_map = parse_file_map(args.data_file_map)
    symbols = _extend_symbols_with_available_optionals(
        parse_symbol_list(args.symbols),
        _parse_optional_symbol_list(args.optional_symbols),
        data_roots=data_roots,
        data_file_map=data_file_map,
    )
    config = WalkForwardConfig(
        target_symbol=args.target_symbol.upper(),
        symbols=symbols,
        data_roots=data_roots,
        data_file_map=data_file_map,
        run_name=args.run_name,
        output_dir=Path(args.output_dir),
        start_date=args.start_date,
        end_date=args.end_date or None,
        target_mode=args.target_mode,
        atr_window=args.atr_window,
        atr_multipliers=_parse_floats(args.atr_multipliers),
        min_train_rows=args.min_train_rows,
        rolling_train_rows=args.rolling_train_rows,
        test_rows=args.test_rows,
        step_rows=args.step_rows,
        context_ffill_limit=args.context_ffill_limit,
        include_target_features=not args.no_target_features,
        include_enhanced_features=args.include_enhanced_features,
        include_atr_distribution_features=args.include_atr_distribution_features,
        include_intraday_60m=args.include_intraday_60m,
        allow_missing_intraday_symbols=args.allow_missing_intraday_symbols,
        intraday_60m_roots=parse_data_roots(args.intraday_60m_roots, DEFAULT_INTRADAY_60M_ROOTS),
        intraday_60m_file_map=parse_file_map(args.intraday_60m_file_map),
        intraday_60m_start_time=args.intraday_rth_start,
        intraday_60m_end_time=args.intraday_rth_end,
        model_type=args.model_type,
        ridge_l2=args.ridge_l2,
        xgb_eta=args.xgb_eta,
        xgb_max_depth=args.xgb_max_depth,
        xgb_num_boost_round=args.xgb_num_boost_round,
        xgb_subsample=args.xgb_subsample,
        xgb_colsample_bytree=args.xgb_colsample_bytree,
        xgb_min_child_weight=args.xgb_min_child_weight,
        xgb_reg_lambda=args.xgb_reg_lambda,
        xgb_reg_alpha=args.xgb_reg_alpha,
        xgb_device=args.xgb_device,
        max_folds=args.max_folds,
    )
    metrics = train_pipeline(config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
