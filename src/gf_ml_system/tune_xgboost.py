from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from gf_ml_system.backtest import BacktestConfig, run_backtest
    from gf_ml_system.config import (
        DEFAULT_DATA_ROOTS,
        DEFAULT_INTRADAY_60M_ROOTS,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_SYMBOLS,
    )
    from gf_ml_system.data import find_latest_symbol_file, parse_data_roots, parse_file_map, parse_symbol_list
    from gf_ml_system.ensemble import score_ensemble
    from gf_ml_system.export_tradestation_backtest import build_easylanguage_backtest_export
    from gf_ml_system.model import (
        WalkForwardConfig,
        binary_metrics,
        build_dataset,
        run_walk_forward,
    )
else:
    from .backtest import BacktestConfig, run_backtest
    from .config import DEFAULT_DATA_ROOTS, DEFAULT_INTRADAY_60M_ROOTS, DEFAULT_OUTPUT_DIR, DEFAULT_SYMBOLS
    from .data import find_latest_symbol_file, parse_data_roots, parse_file_map, parse_symbol_list
    from .ensemble import score_ensemble
    from .export_tradestation_backtest import build_easylanguage_backtest_export
    from .model import WalkForwardConfig, binary_metrics, build_dataset, run_walk_forward


@dataclass(frozen=True)
class XGBoostCandidate:
    name: str
    eta: float
    max_depth: int
    num_boost_round: int
    subsample: float
    colsample_bytree: float
    min_child_weight: float
    reg_lambda: float
    reg_alpha: float = 0.0


FOCUSED_CANDIDATES = (
    XGBoostCandidate(
        name="baseline_d3_r120_l5",
        eta=0.05,
        max_depth=3,
        num_boost_round=120,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5.0,
        reg_lambda=5.0,
    ),
    XGBoostCandidate(
        name="shallow_reg_d2_r160_m10_l10",
        eta=0.05,
        max_depth=2,
        num_boost_round=160,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10.0,
        reg_lambda=10.0,
    ),
    XGBoostCandidate(
        name="slow_shallow_d2_r240_m8_l10",
        eta=0.03,
        max_depth=2,
        num_boost_round=240,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=8.0,
        reg_lambda=10.0,
    ),
    XGBoostCandidate(
        name="reg_d3_r160_m8_l10",
        eta=0.05,
        max_depth=3,
        num_boost_round=160,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=8.0,
        reg_lambda=10.0,
    ),
    XGBoostCandidate(
        name="feature_sparse_d3_r160_c65_l10",
        eta=0.05,
        max_depth=3,
        num_boost_round=160,
        subsample=0.8,
        colsample_bytree=0.65,
        min_child_weight=5.0,
        reg_lambda=10.0,
    ),
    XGBoostCandidate(
        name="deep_reg_d4_r140_c65_m10_l15",
        eta=0.04,
        max_depth=4,
        num_boost_round=140,
        subsample=0.8,
        colsample_bytree=0.65,
        min_child_weight=10.0,
        reg_lambda=15.0,
    ),
    XGBoostCandidate(
        name="light_reg_d3_r120_m3_l3",
        eta=0.05,
        max_depth=3,
        num_boost_round=120,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3.0,
        reg_lambda=3.0,
    ),
)


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


def _parse_thresholds(value: str) -> list[float]:
    thresholds = [float(token.strip()) for token in value.split(",") if token.strip()]
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    return thresholds


def _candidate_config(base: WalkForwardConfig, candidate: XGBoostCandidate) -> WalkForwardConfig:
    return replace(
        base,
        run_name=f"{base.run_name}_{candidate.name}",
        model_type="xgboost",
        xgb_eta=candidate.eta,
        xgb_max_depth=candidate.max_depth,
        xgb_num_boost_round=candidate.num_boost_round,
        xgb_subsample=candidate.subsample,
        xgb_colsample_bytree=candidate.colsample_bytree,
        xgb_min_child_weight=candidate.min_child_weight,
        xgb_reg_lambda=candidate.reg_lambda,
        xgb_reg_alpha=candidate.reg_alpha,
    )


def _metrics_for_predictions(
    config: WalkForwardConfig,
    predictions: pd.DataFrame,
    label_columns: list[str],
    probability_columns: dict[str, str],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "run_name": config.run_name,
        "target_symbol": config.target_symbol.upper(),
        "target_mode": config.target_mode,
        "model_type": config.model_type,
        "xgb_params": {
            "eta": config.xgb_eta,
            "max_depth": config.xgb_max_depth,
            "num_boost_round": config.xgb_num_boost_round,
            "subsample": config.xgb_subsample,
            "colsample_bytree": config.xgb_colsample_bytree,
            "min_child_weight": config.xgb_min_child_weight,
            "reg_lambda": config.xgb_reg_lambda,
            "reg_alpha": config.xgb_reg_alpha,
        },
        "walk_forward": {},
    }
    for label_column in label_columns:
        probability_column = probability_columns[label_column]
        metrics["walk_forward"][label_column] = binary_metrics(
            predictions[label_column].to_numpy(dtype=float),
            predictions[probability_column].to_numpy(dtype=float),
        )
    return metrics


def _threshold_grid(
    predictions: pd.DataFrame,
    *,
    long_thresholds: list[float],
    short_thresholds: list[float],
    atr_stop_multiple: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for long_threshold in long_thresholds:
        for short_threshold in short_thresholds:
            row = {
                "long_threshold": float(long_threshold),
                "short_threshold": float(short_threshold),
            }
            row.update(
                score_ensemble(
                    predictions,
                    long_threshold=long_threshold,
                    short_threshold=short_threshold,
                    atr_stop_multiple=atr_stop_multiple,
                ),
            )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["gross_points", "profit_factor", "trades"],
        ascending=[False, False, False],
    )


def _best_grid_row(grid: pd.DataFrame, *, min_trades: int, min_profit_factor: float) -> pd.Series:
    filtered = grid.copy()
    filtered["profit_factor"] = pd.to_numeric(filtered["profit_factor"], errors="coerce")
    filtered = filtered.loc[
        (filtered["trades"] >= min_trades)
        & (filtered["profit_factor"].fillna(0.0) >= min_profit_factor)
    ]
    if filtered.empty:
        filtered = grid.copy()
    return filtered.sort_values(
        ["gross_points", "profit_factor", "trades"],
        ascending=[False, False, False],
    ).iloc[0]


def _run_fixed_backtest(
    *,
    predictions_path: Path,
    output_dir: Path,
    long_threshold: float,
    short_threshold: float,
    contracts: int,
    point_value: float,
    commission_per_contract_side: float,
    atr_stop_multiple: float,
) -> dict[str, Any]:
    return run_backtest(
        BacktestConfig(
            input_path=predictions_path,
            output_dir=output_dir,
            symbol="ES",
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            contracts=contracts,
            point_value=point_value,
            commission_per_contract_side=commission_per_contract_side,
            use_atr_stop=atr_stop_multiple > 0.0,
            atr_stop_multiple=atr_stop_multiple,
            hold_bars=1,
            same_bar_priority="stop",
        ),
    )


def _leaderboard_row(
    *,
    candidate: XGBoostCandidate,
    metrics: dict[str, Any],
    fixed_score: dict[str, Any],
    best_grid: pd.Series,
    backtest_summary: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    overall = backtest_summary["overall"]
    up_metrics = metrics["walk_forward"].get("target_up_oc_0_25atr", {})
    down_metrics = metrics["walk_forward"].get("target_down_oc_0_25atr", {})
    return {
        "candidate": candidate.name,
        **asdict(candidate),
        "up_auc": up_metrics.get("auc"),
        "up_top_20pct_hit_rate": up_metrics.get("top_20pct_hit_rate"),
        "down_auc": down_metrics.get("auc"),
        "down_top_20pct_hit_rate": down_metrics.get("top_20pct_hit_rate"),
        "fixed_trades": fixed_score.get("trades"),
        "fixed_gross_points": fixed_score.get("gross_points"),
        "fixed_profit_factor": fixed_score.get("profit_factor"),
        "fixed_win_rate": fixed_score.get("win_rate"),
        "fixed_max_dd_points": fixed_score.get("max_drawdown_points"),
        "best_grid_long_threshold": best_grid["long_threshold"],
        "best_grid_short_threshold": best_grid["short_threshold"],
        "best_grid_trades": best_grid["trades"],
        "best_grid_gross_points": best_grid["gross_points"],
        "best_grid_profit_factor": best_grid["profit_factor"],
        "backtest_trades": overall["trades"],
        "backtest_net_pnl": overall["net_pnl"],
        "backtest_gross_points": overall["gross_points"],
        "backtest_profit_factor": overall["profit_factor"],
        "backtest_win_rate": overall["win_rate"],
        "backtest_max_drawdown": overall["max_drawdown_net_pnl"],
        "backtest_avg_trade_net_pnl": overall["avg_trade_net_pnl"],
        "output_dir": str(output_root),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Focused XGBoost tuning for the TES intermarket pipeline.")
    parser.add_argument("--run-name", default="es_next_open_to_close_025atr_tes_daily_rth60_xgboost_wf20_train500_xgb_tune")
    parser.add_argument("--target-symbol", default="ES")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--optional-symbols", default="", help="Symbols to include only when daily data is available.")
    parser.add_argument("--data-roots", default=",".join(str(path) for path in DEFAULT_DATA_ROOTS))
    parser.add_argument("--data-file-map", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--start-date", default="2008-01-01")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--target-mode", default="open_close_atr", choices=("atr_touch", "open_close", "open_close_atr", "combined"))
    parser.add_argument("--atr-window", type=int, default=10)
    parser.add_argument("--atr-multipliers", default="0.25")
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--rolling-train-rows", type=int, default=500)
    parser.add_argument("--test-rows", type=int, default=20)
    parser.add_argument("--step-rows", type=int, default=20)
    parser.add_argument("--context-ffill-limit", type=int, default=2)
    parser.add_argument("--include-enhanced-features", action="store_true")
    parser.add_argument("--include-atr-distribution-features", action="store_true")
    parser.add_argument("--include-intraday-60m", action="store_true", default=True)
    parser.add_argument("--no-intraday-60m", action="store_false", dest="include_intraday_60m")
    parser.add_argument("--intraday-60m-roots", default=",".join(str(path) for path in DEFAULT_INTRADAY_60M_ROOTS))
    parser.add_argument("--intraday-60m-file-map", default="")
    parser.add_argument("--intraday-rth-start", default="10:00")
    parser.add_argument("--intraday-rth-end", default="16:00")
    parser.add_argument("--allow-missing-intraday-symbols", action="store_true")
    parser.add_argument("--xgb-device", default="cpu")
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=0, help="Debug limit. Use 0 for all focused candidates.")
    parser.add_argument("--long-threshold", type=float, default=0.425)
    parser.add_argument("--short-threshold", type=float, default=0.45)
    parser.add_argument("--long-threshold-grid", default="0.35,0.375,0.4,0.425,0.45,0.475,0.5,0.525,0.55,0.575,0.6")
    parser.add_argument("--short-threshold-grid", default="0.35,0.375,0.4,0.425,0.45,0.475,0.5,0.525,0.55,0.575,0.6")
    parser.add_argument("--grid-min-trades", type=int, default=500)
    parser.add_argument("--grid-min-profit-factor", type=float, default=1.20)
    parser.add_argument("--contracts", type=int, default=25)
    parser.add_argument("--point-value", type=float, default=50.0)
    parser.add_argument("--commission-per-contract-side", type=float, default=2.12)
    parser.add_argument("--atr-stop-multiple", type=float, default=1.0)
    parser.add_argument("--with-header", action="store_true")
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
    base_config = WalkForwardConfig(
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
        include_target_features=True,
        include_enhanced_features=args.include_enhanced_features,
        include_atr_distribution_features=args.include_atr_distribution_features,
        include_intraday_60m=args.include_intraday_60m,
        allow_missing_intraday_symbols=args.allow_missing_intraday_symbols,
        intraday_60m_roots=parse_data_roots(args.intraday_60m_roots, DEFAULT_INTRADAY_60M_ROOTS),
        intraday_60m_file_map=parse_file_map(args.intraday_60m_file_map),
        intraday_60m_start_time=args.intraday_rth_start,
        intraday_60m_end_time=args.intraday_rth_end,
        model_type="xgboost",
        xgb_device=args.xgb_device,
        max_folds=args.max_folds,
    )

    output_root = Path(args.output_dir) / args.run_name
    output_root.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(base_config)
    dataset.frame.to_csv(output_root / "training_dataset.csv", index=False)

    candidates = list(FOCUSED_CANDIDATES)
    if args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]

    long_thresholds = _parse_thresholds(args.long_threshold_grid)
    short_thresholds = _parse_thresholds(args.short_threshold_grid)
    leaderboard_rows: list[dict[str, Any]] = []

    for index, candidate in enumerate(candidates, start=1):
        print(f"[{index}/{len(candidates)}] Running {candidate.name}", flush=True)
        candidate_config = _candidate_config(base_config, candidate)
        predictions, folds = run_walk_forward(candidate_config, dataset)
        metrics = _metrics_for_predictions(
            candidate_config,
            predictions,
            dataset.label_columns,
            dataset.probability_columns,
        )
        candidate_root = output_root / candidate.name
        candidate_root.mkdir(parents=True, exist_ok=True)
        predictions_path = candidate_root / "walk_forward_predictions.csv"
        predictions.to_csv(predictions_path, index=False)
        folds.to_csv(candidate_root / "walk_forward_folds.csv", index=False)
        (candidate_root / "metrics.json").write_text(json.dumps(metrics, indent=2))
        build_easylanguage_backtest_export(predictions).to_csv(
            candidate_root / "tradestation_easylanguage_backtest.csv",
            index=False,
            header=args.with_header,
        )

        grid = _threshold_grid(
            predictions,
            long_thresholds=long_thresholds,
            short_thresholds=short_thresholds,
            atr_stop_multiple=args.atr_stop_multiple,
        )
        grid.to_csv(candidate_root / "threshold_grid.csv", index=False)
        fixed_score = score_ensemble(
            predictions,
            long_threshold=args.long_threshold,
            short_threshold=args.short_threshold,
            atr_stop_multiple=args.atr_stop_multiple,
        )
        best_grid = _best_grid_row(
            grid,
            min_trades=args.grid_min_trades,
            min_profit_factor=args.grid_min_profit_factor,
        )
        backtest_summary = _run_fixed_backtest(
            predictions_path=predictions_path,
            output_dir=candidate_root / "backtest_fixed_thresholds",
            long_threshold=args.long_threshold,
            short_threshold=args.short_threshold,
            contracts=args.contracts,
            point_value=args.point_value,
            commission_per_contract_side=args.commission_per_contract_side,
            atr_stop_multiple=args.atr_stop_multiple,
        )
        leaderboard_rows.append(
            _leaderboard_row(
                candidate=candidate,
                metrics=metrics,
                fixed_score=fixed_score,
                best_grid=best_grid,
                backtest_summary=backtest_summary,
                output_root=candidate_root,
            ),
        )
        pd.DataFrame(leaderboard_rows).sort_values(
            ["backtest_net_pnl", "backtest_profit_factor"],
            ascending=[False, False],
        ).to_csv(output_root / "leaderboard.csv", index=False)

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        ["backtest_net_pnl", "backtest_profit_factor"],
        ascending=[False, False],
    )
    leaderboard.to_csv(output_root / "leaderboard.csv", index=False)
    metadata = {
        "config": base_config.to_json_dict(),
        "fixed_thresholds": {
            "long_threshold": args.long_threshold,
            "short_threshold": args.short_threshold,
            "atr_stop_multiple": args.atr_stop_multiple,
            "contracts": args.contracts,
            "point_value": args.point_value,
            "commission_per_contract_side": args.commission_per_contract_side,
        },
        "candidates": [asdict(candidate) for candidate in candidates],
        "output_root": str(output_root),
    }
    (output_root / "tune_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"output_root": str(output_root), "leaderboard": leaderboard.to_dict(orient="records")}, indent=2))


if __name__ == "__main__":
    main()
