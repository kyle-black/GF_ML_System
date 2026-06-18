from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from gf_ml_system.config import DEFAULT_OUTPUT_DIR
    from gf_ml_system.data import read_price_csv
else:
    from .config import DEFAULT_OUTPUT_DIR
    from .data import read_price_csv


@dataclass(frozen=True)
class BacktestConfig:
    input_path: Path
    output_dir: Path
    price_path: Path | None = None
    symbol: str = "ES"
    long_threshold: float = 0.50
    short_threshold: float = 0.50
    long_selector: int = 2
    short_selector: int = 2
    enable_longs: bool = True
    enable_shorts: bool = True
    require_long_beats_short: bool = True
    require_short_beats_long: bool = True
    signal_row_mode: str = "same_day"
    hold_bars: int = 1
    contracts: int = 1
    point_value: float = 50.0
    commission_per_contract_side: float = 0.0
    use_atr_stop: bool = False
    atr_stop_multiple: float = 1.0
    use_atr_target: bool = False
    atr_target_multiple: float = 1.0
    same_bar_priority: str = "stop"
    start_date: str | None = None
    end_date: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_path"] = str(self.input_path)
        payload["output_dir"] = str(self.output_dir)
        payload["price_path"] = str(self.price_path) if self.price_path else None
        return payload


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


def _normalize_signal_row_mode(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"1", "same", "same_day", "same_session"}:
        return "same_day"
    if normalized in {"2", "prior", "prior_session", "previous"}:
        return "prior_session"
    raise ValueError("signal_row_mode must be same-day or prior-session")


def _normalize_same_bar_priority(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in {"stop", "target", "best", "worst"}:
        raise ValueError("same_bar_priority must be one of: stop, target, best, worst")
    return normalized


def _probability_columns(selector: int, side: str) -> tuple[str, ...]:
    if side not in {"up", "down"}:
        raise ValueError("side must be up or down")
    if selector == 1:
        return (f"prob_{side}_close",)
    if selector == 2:
        return (f"prob_{side}_0_25_atr", f"prob_{side}_0_25atr", f"prob_{side}_oc_0_25atr")
    if selector == 3:
        return (f"prob_{side}_0_5_atr", f"prob_{side}_0_5atr", f"prob_{side}_oc_0_5atr")
    if selector == 4:
        return (f"prob_{side}_0_75_atr", f"prob_{side}_0_75atr", f"prob_{side}_oc_0_75atr")
    if selector == 5:
        return (f"prob_{side}_1_atr", f"prob_{side}_1atr", f"prob_{side}_oc_1atr")
    if selector == 6:
        return (f"highest_{side}_atr_probability",)
    raise ValueError("selector must be an integer from 1 through 6")


def _first_available(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    for column in columns:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=frame.index)


def _el_date(date_values: pd.Series) -> pd.Series:
    return (
        (date_values.dt.year - 1900) * 10000
        + date_values.dt.month * 100
        + date_values.dt.day
    ).astype(int)


def _read_prediction_file(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if "signal_date" in raw.columns:
        frame = raw.copy()
        frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="coerce")
        return frame.loc[frame["signal_date"].notna()].copy()

    # Headerless TradeStation export fallback. This is useful for auditing the
    # exact file that EasyLanguage reads, but price execution still needs a
    # matching price file.
    frame = pd.read_csv(path, header=None, names=EASYLANGUAGE_COLUMNS)
    if len(frame.columns) != len(EASYLANGUAGE_COLUMNS):
        raise ValueError(
            "Input must be a walk_forward_predictions CSV or a 15-column TradeStation export.",
        )
    frame.columns = EASYLANGUAGE_COLUMNS
    frame["signal_date"] = pd.to_datetime(frame["date_iso"], errors="coerce")
    return frame.loc[frame["signal_date"].notna()].copy()


def _build_price_frame_from_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "signal_date",
        "next_open",
        "next_close",
        "target_close",
        "atr_points",
        "next_high_move_atr",
        "next_low_move_atr",
    }
    if not required.issubset(predictions.columns):
        missing = sorted(required - set(predictions.columns))
        raise ValueError(
            "Prediction file does not include enough next-session price fields. "
            f"Provide --price-file or use walk_forward_predictions input. Missing: {missing}",
        )

    source = predictions.copy()
    source["signal_date"] = pd.to_datetime(source["signal_date"], errors="coerce")
    for column in [
        "target_close",
        "next_open",
        "next_close",
        "atr_points",
        "next_high_move_atr",
        "next_low_move_atr",
    ]:
        source[column] = pd.to_numeric(source[column], errors="coerce")
    source = source.dropna(
        subset=[
            "signal_date",
            "target_close",
            "next_open",
            "next_close",
            "atr_points",
            "next_high_move_atr",
            "next_low_move_atr",
        ],
    ).sort_values("signal_date")

    rows: list[dict[str, Any]] = []
    for index, row in source.reset_index(drop=True).iterrows():
        if index + 1 < len(source):
            trade_date = source["signal_date"].iloc[index + 1]
        else:
            trade_date = row["signal_date"] + pd.tseries.offsets.BDay(1)
        rows.append(
            {
                "date": pd.Timestamp(trade_date).normalize(),
                "open": float(row["next_open"]),
                "high": float(row["target_close"] + row["next_high_move_atr"] * row["atr_points"]),
                "low": float(row["target_close"] - row["next_low_move_atr"] * row["atr_points"]),
                "close": float(row["next_close"]),
            },
        )
    price_frame = pd.DataFrame(rows).drop_duplicates(subset=["date"], keep="last")
    return price_frame.set_index("date").sort_index()


def _prepare_prediction_frame(predictions: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    frame = predictions.copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="coerce").dt.normalize()
    frame = frame.loc[frame["signal_date"].notna()].sort_values("signal_date").reset_index(drop=True)
    if config.start_date:
        frame = frame.loc[frame["signal_date"] >= pd.Timestamp(config.start_date)]
    if config.end_date:
        frame = frame.loc[frame["signal_date"] <= pd.Timestamp(config.end_date)]

    frame["long_probability"] = _first_available(frame, _probability_columns(config.long_selector, "up"))
    frame["short_probability"] = _first_available(frame, _probability_columns(config.short_selector, "down"))
    if "atr_points" in frame.columns:
        frame["atr_points"] = pd.to_numeric(frame["atr_points"], errors="coerce")
    else:
        frame["atr_points"] = np.nan
    return frame


def _load_price_frame(config: BacktestConfig, predictions: pd.DataFrame) -> pd.DataFrame:
    if config.price_path:
        price_frame = read_price_csv(config.price_path).copy()
        price_frame.index = pd.to_datetime(price_frame.index).normalize()
        return price_frame.sort_index()
    return _build_price_frame_from_predictions(predictions)


def _next_index(index: pd.DatetimeIndex, date_value: pd.Timestamp, *, mode: str) -> int | None:
    side = "right" if mode == "same_day" else "left"
    position = int(index.searchsorted(date_value, side=side))
    if position >= len(index):
        return None
    return position


def _resolve_same_bar_exit(
    *,
    direction: int,
    stop_price: float,
    target_price: float,
    stop_hit: bool,
    target_hit: bool,
    priority: str,
) -> tuple[float, str]:
    if stop_hit and target_hit:
        if priority == "target":
            return target_price, "atr_target_same_bar"
        if priority == "best":
            return (max(stop_price, target_price), "best_same_bar") if direction == 1 else (
                min(stop_price, target_price),
                "best_same_bar",
            )
        if priority == "worst":
            return (min(stop_price, target_price), "worst_same_bar") if direction == 1 else (
                max(stop_price, target_price),
                "worst_same_bar",
            )
        return stop_price, "atr_stop_same_bar"
    if stop_hit:
        return stop_price, "atr_stop"
    if target_hit:
        return target_price, "atr_target"
    raise ValueError("No same-bar exit was hit.")


def _simulate_trade(
    *,
    signal_row: pd.Series,
    direction: int,
    entry_idx: int,
    price_frame: pd.DataFrame,
    config: BacktestConfig,
) -> dict[str, Any] | None:
    hold_bars = max(1, int(config.hold_bars))
    exit_idx = min(entry_idx + hold_bars - 1, len(price_frame) - 1)
    price_slice = price_frame.iloc[entry_idx : exit_idx + 1]
    if price_slice.empty:
        return None

    entry_date = price_slice.index[0]
    planned_exit_date = price_slice.index[-1]
    entry_price = float(price_slice["open"].iloc[0])
    exit_price = float(price_slice["close"].iloc[-1])
    exit_date = planned_exit_date
    exit_reason = "hold_bars_exit"
    atr_points = float(signal_row.get("atr_points", np.nan))

    stop_price = np.nan
    target_price = np.nan
    if config.use_atr_stop:
        if not np.isfinite(atr_points) or atr_points <= 0:
            raise ValueError("ATR stop requested but atr_points is missing or non-positive.")
        stop_price = entry_price - direction * config.atr_stop_multiple * atr_points
    if config.use_atr_target:
        if not np.isfinite(atr_points) or atr_points <= 0:
            raise ValueError("ATR target requested but atr_points is missing or non-positive.")
        target_price = entry_price + direction * config.atr_target_multiple * atr_points

    for current_date, bar in price_slice.iterrows():
        stop_hit = False
        target_hit = False
        if config.use_atr_stop:
            if direction == 1:
                stop_hit = float(bar["low"]) <= stop_price
            else:
                stop_hit = float(bar["high"]) >= stop_price
        if config.use_atr_target:
            if direction == 1:
                target_hit = float(bar["high"]) >= target_price
            else:
                target_hit = float(bar["low"]) <= target_price
        if stop_hit or target_hit:
            exit_price, exit_reason = _resolve_same_bar_exit(
                direction=direction,
                stop_price=stop_price,
                target_price=target_price,
                stop_hit=stop_hit,
                target_hit=target_hit,
                priority=config.same_bar_priority,
            )
            exit_date = current_date
            break

    gross_points = direction * (exit_price - entry_price)
    gross_pnl = gross_points * config.point_value * config.contracts
    commission = 2.0 * config.commission_per_contract_side * config.contracts
    net_pnl = gross_pnl - commission
    runup_points = np.nan
    drawdown_points = np.nan
    if direction == 1:
        runup_points = float(price_slice["high"].max()) - entry_price
        drawdown_points = float(price_slice["low"].min()) - entry_price
    else:
        runup_points = entry_price - float(price_slice["low"].min())
        drawdown_points = entry_price - float(price_slice["high"].max())

    return {
        "signal_date": signal_row["signal_date"],
        "entry_date": entry_date,
        "exit_date": exit_date,
        "side": "long" if direction == 1 else "short",
        "direction": direction,
        "long_probability": float(signal_row["long_probability"]),
        "short_probability": float(signal_row["short_probability"]),
        "atr_points": atr_points,
        "entry_price": entry_price,
        "exit_price": float(exit_price),
        "exit_reason": exit_reason,
        "hold_bars": hold_bars,
        "contracts": config.contracts,
        "point_value": config.point_value,
        "gross_points": float(gross_points),
        "gross_pnl": float(gross_pnl),
        "commission": float(commission),
        "net_pnl": float(net_pnl),
        "runup_points": float(runup_points),
        "drawdown_points": float(drawdown_points),
        "stop_price": float(stop_price) if np.isfinite(stop_price) else np.nan,
        "target_price": float(target_price) if np.isfinite(target_price) else np.nan,
    }


def run_backtest(config: BacktestConfig) -> dict[str, Any]:
    signal_row_mode = _normalize_signal_row_mode(config.signal_row_mode)
    same_bar_priority = _normalize_same_bar_priority(config.same_bar_priority)
    config = BacktestConfig(**{**config.to_json_dict(), "input_path": config.input_path, "output_dir": config.output_dir, "price_path": config.price_path, "signal_row_mode": signal_row_mode, "same_bar_priority": same_bar_priority})

    predictions_raw = _read_prediction_file(config.input_path)
    predictions = _prepare_prediction_frame(predictions_raw, config)
    price_frame = _load_price_frame(config, predictions)
    price_index = price_frame.index

    trades: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    for _, signal_row in predictions.iterrows():
        long_signal = config.enable_longs and signal_row["long_probability"] >= config.long_threshold
        short_signal = config.enable_shorts and signal_row["short_probability"] >= config.short_threshold
        if config.require_long_beats_short:
            long_signal = long_signal and signal_row["long_probability"] > signal_row["short_probability"]
        if config.require_short_beats_long:
            short_signal = short_signal and signal_row["short_probability"] > signal_row["long_probability"]
        if long_signal and short_signal:
            skipped_rows.append(
                {
                    "signal_date": signal_row["signal_date"],
                    "reason": "both_sides_active",
                    "long_probability": signal_row["long_probability"],
                    "short_probability": signal_row["short_probability"],
                },
            )
            continue
        if not long_signal and not short_signal:
            continue

        entry_idx = _next_index(price_index, signal_row["signal_date"], mode=signal_row_mode)
        if entry_idx is None:
            skipped_rows.append(
                {
                    "signal_date": signal_row["signal_date"],
                    "reason": "no_future_price_bar",
                    "long_probability": signal_row["long_probability"],
                    "short_probability": signal_row["short_probability"],
                },
            )
            continue
        direction = 1 if long_signal else -1
        trade = _simulate_trade(
            signal_row=signal_row,
            direction=direction,
            entry_idx=entry_idx,
            price_frame=price_frame,
            config=config,
        )
        if trade is not None:
            trades.append(trade)

    trades_frame = pd.DataFrame(trades)
    if not trades_frame.empty:
        trades_frame = trades_frame.sort_values(["entry_date", "signal_date"]).reset_index(drop=True)
        trades_frame["trade_number"] = np.arange(1, len(trades_frame) + 1)
        trades_frame["cum_net_pnl"] = trades_frame["net_pnl"].cumsum()
        trades_frame["cum_gross_points"] = trades_frame["gross_points"].cumsum()
        trades_frame["equity_peak"] = trades_frame["cum_net_pnl"].cummax()
        trades_frame["drawdown_net_pnl"] = trades_frame["cum_net_pnl"] - trades_frame["equity_peak"]
    else:
        trades_frame = pd.DataFrame(
            columns=[
                "trade_number",
                "signal_date",
                "entry_date",
                "exit_date",
                "side",
                "gross_points",
                "gross_pnl",
                "commission",
                "net_pnl",
                "cum_net_pnl",
            ],
        )

    skipped_frame = pd.DataFrame(skipped_rows)
    annual = _period_summary(trades_frame, period="Y")
    monthly = _period_summary(trades_frame, period="M")
    side_summary = _side_summary(trades_frame)
    summary = {
        "config": config.to_json_dict(),
        "input_rows": int(len(predictions)),
        "price_rows": int(len(price_frame)),
        "price_start_date": str(price_frame.index.min().date()) if not price_frame.empty else None,
        "price_end_date": str(price_frame.index.max().date()) if not price_frame.empty else None,
        "skipped_rows": int(len(skipped_frame)),
        "overall": _summary_stats(trades_frame),
        "by_side": side_summary,
    }

    config.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "trades": config.output_dir / "trades.csv",
        "skipped_rows": config.output_dir / "skipped_rows.csv",
        "annual": config.output_dir / "annual.csv",
        "monthly": config.output_dir / "monthly.csv",
        "summary": config.output_dir / "summary.json",
    }
    trades_frame.to_csv(paths["trades"], index=False)
    skipped_frame.to_csv(paths["skipped_rows"], index=False)
    annual.to_csv(paths["annual"], index=False)
    monthly.to_csv(paths["monthly"], index=False)
    paths["summary"].write_text(json.dumps(summary, indent=2))
    summary["output_paths"] = {key: str(path) for key, path in paths.items()}
    return summary


def _summary_stats(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {
            "trades": 0,
            "net_pnl": 0.0,
            "gross_pnl": 0.0,
            "gross_points": 0.0,
            "profit_factor": None,
            "win_rate": None,
            "avg_trade_net_pnl": None,
            "avg_trade_points": None,
            "max_drawdown_net_pnl": 0.0,
        }
    wins = trades.loc[trades["net_pnl"] > 0, "net_pnl"]
    losses = trades.loc[trades["net_pnl"] < 0, "net_pnl"]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    return {
        "trades": int(len(trades)),
        "winning_trades": int((trades["net_pnl"] > 0).sum()),
        "losing_trades": int((trades["net_pnl"] < 0).sum()),
        "net_pnl": float(trades["net_pnl"].sum()),
        "gross_pnl": float(trades["gross_pnl"].sum()),
        "commission": float(trades["commission"].sum()),
        "gross_points": float(trades["gross_points"].sum()),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss else None,
        "win_rate": float((trades["net_pnl"] > 0).mean()),
        "avg_trade_net_pnl": float(trades["net_pnl"].mean()),
        "avg_trade_points": float(trades["gross_points"].mean()),
        "largest_win": float(trades["net_pnl"].max()),
        "largest_loss": float(trades["net_pnl"].min()),
        "max_drawdown_net_pnl": float(trades["drawdown_net_pnl"].min()),
        "start_date": str(pd.to_datetime(trades["entry_date"]).min().date()),
        "end_date": str(pd.to_datetime(trades["exit_date"]).max().date()),
    }


def _side_summary(trades: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for side in ("long", "short"):
        output[side] = _summary_stats(trades.loc[trades["side"] == side].copy()) if not trades.empty else _summary_stats(trades)
    return output


def _period_summary(trades: pd.DataFrame, *, period: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    frame = trades.copy()
    frame["exit_date"] = pd.to_datetime(frame["exit_date"])
    frame["period"] = frame["exit_date"].dt.to_period(period).astype(str)
    rows: list[dict[str, Any]] = []
    for period_value, group in frame.groupby("period", sort=True):
        stats = _summary_stats(group.copy())
        rows.append({"period": period_value, **stats})
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local open-to-close ML signal backtest from walk-forward predictions.",
    )
    parser.add_argument("--input", default="", help="Prediction CSV or 15-column TradeStation CSV.")
    parser.add_argument("--run-name", default="", help="Run folder under data/ml/tes_intermarket.")
    parser.add_argument(
        "--prediction-file",
        default="walk_forward_predictions.csv",
        help="Prediction file name inside --run-name when --input is omitted.",
    )
    parser.add_argument("--output-dir", default="", help="Output folder for summary/trades CSVs.")
    parser.add_argument("--price-file", default="", help="Optional daily OHLC CSV. Defaults to next_* prediction fields.")
    parser.add_argument("--symbol", default="ES")
    parser.add_argument("--long-threshold", type=float, default=0.50)
    parser.add_argument("--short-threshold", type=float, default=0.50)
    parser.add_argument("--long-selector", type=int, default=2)
    parser.add_argument("--short-selector", type=int, default=2)
    parser.add_argument("--disable-longs", action="store_true")
    parser.add_argument("--disable-shorts", action="store_true")
    parser.add_argument("--allow-long-without-beating-short", action="store_true")
    parser.add_argument("--allow-short-without-beating-long", action="store_true")
    parser.add_argument("--signal-row-mode", default="same-day", choices=("same-day", "prior-session"))
    parser.add_argument("--hold-bars", type=int, default=1)
    parser.add_argument("--contracts", type=int, default=1)
    parser.add_argument("--point-value", type=float, default=50.0)
    parser.add_argument("--commission-per-contract-side", type=float, default=0.0)
    parser.add_argument("--use-atr-stop", action="store_true")
    parser.add_argument("--atr-stop-multiple", type=float, default=1.0)
    parser.add_argument("--use-atr-target", action="store_true")
    parser.add_argument("--atr-target-multiple", type=float, default=1.0)
    parser.add_argument("--same-bar-priority", default="stop", choices=("stop", "target", "best", "worst"))
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    return parser.parse_args()


def _resolve_input_path(args: argparse.Namespace) -> Path:
    if args.input:
        return Path(args.input)
    if not args.run_name:
        raise ValueError("Provide --input or --run-name.")
    return DEFAULT_OUTPUT_DIR / args.run_name / args.prediction_file


def main() -> None:
    args = parse_args()
    input_path = _resolve_input_path(args)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stem = input_path.stem.replace("walk_forward_predictions_", "")
        output_dir = input_path.parent / "backtests" / stem
    config = BacktestConfig(
        input_path=input_path,
        output_dir=output_dir,
        price_path=Path(args.price_file) if args.price_file else None,
        symbol=args.symbol.upper(),
        long_threshold=args.long_threshold,
        short_threshold=args.short_threshold,
        long_selector=args.long_selector,
        short_selector=args.short_selector,
        enable_longs=not args.disable_longs,
        enable_shorts=not args.disable_shorts,
        require_long_beats_short=not args.allow_long_without_beating_short,
        require_short_beats_long=not args.allow_short_without_beating_long,
        signal_row_mode=args.signal_row_mode,
        hold_bars=args.hold_bars,
        contracts=args.contracts,
        point_value=args.point_value,
        commission_per_contract_side=args.commission_per_contract_side,
        use_atr_stop=args.use_atr_stop,
        atr_stop_multiple=args.atr_stop_multiple,
        use_atr_target=args.use_atr_target,
        atr_target_multiple=args.atr_target_multiple,
        same_bar_priority=args.same_bar_priority,
        start_date=args.start_date or None,
        end_date=args.end_date or None,
    )
    summary = run_backtest(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
