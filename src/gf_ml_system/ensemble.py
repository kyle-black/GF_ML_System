from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from gf_ml_system.export_tradestation_backtest import build_easylanguage_backtest_export
else:
    from .export_tradestation_backtest import build_easylanguage_backtest_export


BASE_COLUMNS = [
    "signal_date",
    "target_close",
    "next_open",
    "next_close",
    "atr_points",
    "next_high_move_atr",
    "next_low_move_atr",
    "next_close_return_atr",
    "next_open_to_close_return_points",
    "next_open_to_close_return_atr",
    "target_up_oc_0_25atr",
    "target_down_oc_0_25atr",
]
UP_COLUMN = "prob_up_oc_0_25atr"
DOWN_COLUMN = "prob_down_oc_0_25atr"


@dataclass(frozen=True)
class EnsembleMember:
    name: str
    path: Path


def _parse_member(value: str) -> EnsembleMember:
    if "=" not in value:
        raise ValueError("--member values must be name=/path/to/walk_forward_predictions.csv")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("--member name cannot be empty")
    return EnsembleMember(name=name, path=Path(path.strip()))


def _read_member(member: EnsembleMember, *, include_base: bool) -> pd.DataFrame:
    if not member.path.exists():
        raise FileNotFoundError(f"Prediction file not found for {member.name}: {member.path}")
    source = pd.read_csv(member.path)
    required = {"signal_date", UP_COLUMN, DOWN_COLUMN}
    if include_base:
        required.update(BASE_COLUMNS)
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"{member.path} is missing required columns: {sorted(missing)}")
    columns = BASE_COLUMNS if include_base else ["signal_date"]
    frame = source.loc[:, columns + [UP_COLUMN, DOWN_COLUMN]].copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame = frame.loc[frame["signal_date"].notna()].drop_duplicates(subset=["signal_date"], keep="last")
    return frame.rename(
        columns={
            UP_COLUMN: f"{member.name}__{UP_COLUMN}",
            DOWN_COLUMN: f"{member.name}__{DOWN_COLUMN}",
        },
    )


def load_member_predictions(members: list[EnsembleMember]) -> pd.DataFrame:
    if len(members) < 2:
        raise ValueError("At least two --member inputs are required.")
    merged = _read_member(members[0], include_base=True)
    for member in members[1:]:
        merged = merged.merge(_read_member(member, include_base=False), on="signal_date", how="inner")
    return merged.sort_values("signal_date").reset_index(drop=True)


def _parse_weights(value: str, member_names: list[str]) -> np.ndarray:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise ValueError("Weight list cannot be empty.")
    if all("=" in token for token in tokens):
        mapping: dict[str, float] = {}
        for token in tokens:
            name, raw_weight = token.split("=", 1)
            mapping[name.strip()] = float(raw_weight)
        missing = set(member_names) - set(mapping)
        if missing:
            raise ValueError(f"Missing weights for members: {sorted(missing)}")
        weights = np.array([mapping[name] for name in member_names], dtype=float)
    else:
        if len(tokens) != len(member_names):
            raise ValueError(
                f"Expected {len(member_names)} comma-separated weights for members {member_names}; got {len(tokens)}",
            )
        weights = np.array([float(token) for token in tokens], dtype=float)
    return _normalize_weights(weights)


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    if np.any(weights < 0.0):
        raise ValueError("Weights must be non-negative.")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("At least one weight must be positive.")
    return weights / total


def _simplex_weights(member_count: int, step: float) -> list[np.ndarray]:
    if step <= 0.0 or step > 1.0:
        raise ValueError("--search-grid-step must be in (0, 1].")
    units = round(1.0 / step)
    if not np.isclose(units * step, 1.0):
        raise ValueError("--search-grid-step must divide 1.0 evenly, e.g. 0.1 or 0.05.")

    rows: list[np.ndarray] = []

    def rec(prefix: list[int], remaining: int, slots_left: int) -> None:
        if slots_left == 1:
            rows.append(np.array([*prefix, remaining], dtype=float) / units)
            return
        for value in range(remaining + 1):
            rec([*prefix, value], remaining - value, slots_left - 1)

    rec([], int(units), member_count)
    return rows


def build_weighted_ensemble(
    merged: pd.DataFrame,
    *,
    member_names: list[str],
    long_weights: np.ndarray,
    short_weights: np.ndarray,
) -> pd.DataFrame:
    long_matrix = merged[[f"{name}__{UP_COLUMN}" for name in member_names]].apply(
        pd.to_numeric,
        errors="coerce",
    ).fillna(0.0)
    short_matrix = merged[[f"{name}__{DOWN_COLUMN}" for name in member_names]].apply(
        pd.to_numeric,
        errors="coerce",
    ).fillna(0.0)
    output = merged.loc[:, BASE_COLUMNS].copy()
    output[UP_COLUMN] = np.clip(long_matrix.to_numpy(dtype=float) @ long_weights, 0.0, 1.0)
    output[DOWN_COLUMN] = np.clip(short_matrix.to_numpy(dtype=float) @ short_weights, 0.0, 1.0)
    return output


def _approx_trade_points(
    frame: pd.DataFrame,
    *,
    long_threshold: float,
    short_threshold: float,
    atr_stop_multiple: float,
) -> pd.Series:
    long_prob = pd.to_numeric(frame[UP_COLUMN], errors="coerce").fillna(0.0)
    short_prob = pd.to_numeric(frame[DOWN_COLUMN], errors="coerce").fillna(0.0)
    long_signal = (long_prob >= long_threshold) & (long_prob > short_prob)
    short_signal = (short_prob >= short_threshold) & (short_prob > long_prob)
    direction = pd.Series(0, index=frame.index, dtype=int)
    direction.loc[long_signal] = 1
    direction.loc[short_signal] = -1

    next_open = pd.to_numeric(frame["next_open"], errors="coerce")
    next_close = pd.to_numeric(frame["next_close"], errors="coerce")
    target_close = pd.to_numeric(frame["target_close"], errors="coerce")
    atr = pd.to_numeric(frame["atr_points"], errors="coerce")
    high = target_close + pd.to_numeric(frame["next_high_move_atr"], errors="coerce") * atr
    low = target_close - pd.to_numeric(frame["next_low_move_atr"], errors="coerce") * atr

    points = direction.astype(float) * (next_close - next_open)
    if atr_stop_multiple > 0.0:
        long_stop_hit = long_signal & (low <= next_open - atr_stop_multiple * atr)
        short_stop_hit = short_signal & (high >= next_open + atr_stop_multiple * atr)
        points.loc[long_stop_hit | short_stop_hit] = -atr_stop_multiple * atr.loc[long_stop_hit | short_stop_hit]
    return points.loc[direction != 0].dropna()


def _score_points(points: pd.Series) -> dict[str, Any]:
    if points.empty:
        return {
            "trades": 0,
            "gross_points": 0.0,
            "profit_factor": None,
            "win_rate": None,
            "avg_points": None,
            "max_drawdown_points": 0.0,
        }
    wins = points.loc[points > 0.0]
    losses = points.loc[points < 0.0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    cumulative = points.cumsum()
    drawdown = cumulative - cumulative.cummax()
    return {
        "trades": int(len(points)),
        "gross_points": float(points.sum()),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss else None,
        "win_rate": float((points > 0.0).mean()),
        "avg_points": float(points.mean()),
        "max_drawdown_points": float(drawdown.min()),
    }


def _score_points_array(points: np.ndarray) -> dict[str, Any]:
    points = points[np.isfinite(points)]
    if points.size == 0:
        return {
            "trades": 0,
            "gross_points": 0.0,
            "profit_factor": None,
            "win_rate": None,
            "avg_points": None,
            "max_drawdown_points": 0.0,
        }
    wins = points[points > 0.0]
    losses = points[points < 0.0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    cumulative = np.cumsum(points)
    drawdown = cumulative - np.maximum.accumulate(cumulative)
    return {
        "trades": int(points.size),
        "gross_points": float(points.sum()),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss else None,
        "win_rate": float(np.mean(points > 0.0)),
        "avg_points": float(points.mean()),
        "max_drawdown_points": float(drawdown.min()),
    }


def score_ensemble(
    frame: pd.DataFrame,
    *,
    long_threshold: float,
    short_threshold: float,
    atr_stop_multiple: float,
) -> dict[str, Any]:
    return _score_points(
        _approx_trade_points(
            frame,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            atr_stop_multiple=atr_stop_multiple,
        ),
    )


def _metric_value(row: dict[str, Any], metric: str) -> float:
    if metric == "gross_points":
        return float(row["gross_points"])
    if metric == "profit_factor":
        return float(row["profit_factor"] or 0.0)
    if metric == "points_per_dd":
        drawdown = abs(float(row["max_drawdown_points"]))
        return float(row["gross_points"]) / drawdown if drawdown else 0.0
    raise ValueError("--metric must be one of: gross_points, profit_factor, points_per_dd")


def search_weight_grid(
    merged: pd.DataFrame,
    *,
    member_names: list[str],
    step: float,
    long_threshold: float,
    short_threshold: float,
    atr_stop_multiple: float,
    min_trades: int,
    min_profit_factor: float,
    metric: str,
) -> pd.DataFrame:
    weights = _simplex_weights(len(member_names), step)
    weight_array = np.vstack(weights)
    up_matrix = merged[[f"{name}__{UP_COLUMN}" for name in member_names]].apply(
        pd.to_numeric,
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)
    down_matrix = merged[[f"{name}__{DOWN_COLUMN}" for name in member_names]].apply(
        pd.to_numeric,
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)
    up_scores = weight_array @ up_matrix.T
    down_scores = weight_array @ down_matrix.T

    next_open = pd.to_numeric(merged["next_open"], errors="coerce").to_numpy(dtype=float)
    next_close = pd.to_numeric(merged["next_close"], errors="coerce").to_numpy(dtype=float)
    target_close = pd.to_numeric(merged["target_close"], errors="coerce").to_numpy(dtype=float)
    atr = pd.to_numeric(merged["atr_points"], errors="coerce").to_numpy(dtype=float)
    high = target_close + pd.to_numeric(merged["next_high_move_atr"], errors="coerce").to_numpy(dtype=float) * atr
    low = target_close - pd.to_numeric(merged["next_low_move_atr"], errors="coerce").to_numpy(dtype=float) * atr

    rows: list[dict[str, Any]] = []
    for long_index, short_index in product(range(len(weights)), repeat=2):
        long_weights = weights[long_index]
        short_weights = weights[short_index]
        long_prob = up_scores[long_index]
        short_prob = down_scores[short_index]
        long_signal = (long_prob >= long_threshold) & (long_prob > short_prob)
        short_signal = (short_prob >= short_threshold) & (short_prob > long_prob)
        direction = np.zeros(len(merged), dtype=float)
        direction[long_signal] = 1.0
        direction[short_signal] = -1.0
        points = direction * (next_close - next_open)
        if atr_stop_multiple > 0.0:
            long_stop_hit = long_signal & (low <= next_open - atr_stop_multiple * atr)
            short_stop_hit = short_signal & (high >= next_open + atr_stop_multiple * atr)
            stop_hit = long_stop_hit | short_stop_hit
            points[stop_hit] = -atr_stop_multiple * atr[stop_hit]
        stats = _score_points_array(points[direction != 0.0])
        eligible = stats["trades"] >= min_trades and (
            stats["profit_factor"] is not None and stats["profit_factor"] >= min_profit_factor
        )
        row = {
            "long_threshold": long_threshold,
            "short_threshold": short_threshold,
            "eligible": bool(eligible),
            **{f"long_weight_{name}": float(weight) for name, weight in zip(member_names, long_weights, strict=True)},
            **{f"short_weight_{name}": float(weight) for name, weight in zip(member_names, short_weights, strict=True)},
            **stats,
        }
        row["metric_value"] = _metric_value(row, metric)
        rows.append(row)
    results = pd.DataFrame(rows)
    return results.sort_values(["eligible", "metric_value", "gross_points"], ascending=[False, False, False])


def _weights_from_row(row: pd.Series, member_names: list[str], side: str) -> np.ndarray:
    return np.array([float(row[f"{side}_weight_{name}"]) for name in member_names], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build weighted ensembles from walk-forward prediction files.")
    parser.add_argument("--member", action="append", default=[], help="Repeat as name=/path/to/walk_forward_predictions.csv")
    parser.add_argument("--long-weights", default="", help="Comma weights in member order, or name=weight pairs.")
    parser.add_argument("--short-weights", default="", help="Comma weights in member order, or name=weight pairs.")
    parser.add_argument("--search-grid-step", type=float, default=0.0, help="Optional simplex grid step, e.g. 0.1.")
    parser.add_argument("--long-threshold", type=float, default=0.425)
    parser.add_argument("--short-threshold", type=float, default=0.45)
    parser.add_argument("--atr-stop-multiple", type=float, default=1.0)
    parser.add_argument("--min-trades", type=int, default=500)
    parser.add_argument("--min-profit-factor", type=float, default=1.35)
    parser.add_argument("--metric", default="gross_points", choices=("gross_points", "profit_factor", "points_per_dd"))
    parser.add_argument("--output-predictions", required=True)
    parser.add_argument("--output-tradestation", default="")
    parser.add_argument("--output-grid", default="")
    parser.add_argument("--metadata", default="")
    parser.add_argument("--with-header", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    members = [_parse_member(value) for value in args.member]
    member_names = [member.name for member in members]
    merged = load_member_predictions(members)

    grid: pd.DataFrame | None = None
    selected_row: dict[str, Any] | None = None
    if args.search_grid_step > 0.0:
        grid = search_weight_grid(
            merged,
            member_names=member_names,
            step=args.search_grid_step,
            long_threshold=args.long_threshold,
            short_threshold=args.short_threshold,
            atr_stop_multiple=args.atr_stop_multiple,
            min_trades=args.min_trades,
            min_profit_factor=args.min_profit_factor,
            metric=args.metric,
        )
        if args.output_grid:
            output_grid = Path(args.output_grid)
            output_grid.parent.mkdir(parents=True, exist_ok=True)
            grid.to_csv(output_grid, index=False)
        best = grid.iloc[0]
        long_weights = _weights_from_row(best, member_names, "long")
        short_weights = _weights_from_row(best, member_names, "short")
        selected_row = best.to_dict()
    else:
        if not args.long_weights or not args.short_weights:
            raise ValueError("Provide --long-weights/--short-weights or use --search-grid-step.")
        long_weights = _parse_weights(args.long_weights, member_names)
        short_weights = _parse_weights(args.short_weights, member_names)

    ensemble = build_weighted_ensemble(
        merged,
        member_names=member_names,
        long_weights=long_weights,
        short_weights=short_weights,
    )
    prediction_path = Path(args.output_predictions)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble.to_csv(prediction_path, index=False)

    tradestation_path = Path(args.output_tradestation) if args.output_tradestation else None
    if tradestation_path:
        tradestation_path.parent.mkdir(parents=True, exist_ok=True)
        build_easylanguage_backtest_export(ensemble).to_csv(
            tradestation_path,
            index=False,
            header=args.with_header,
        )

    metadata = {
        "members": {member.name: str(member.path) for member in members},
        "member_rows_inner_join": int(len(merged)),
        "long_weights": {name: float(weight) for name, weight in zip(member_names, long_weights, strict=True)},
        "short_weights": {name: float(weight) for name, weight in zip(member_names, short_weights, strict=True)},
        "long_threshold": args.long_threshold,
        "short_threshold": args.short_threshold,
        "atr_stop_multiple": args.atr_stop_multiple,
        "approx_score": score_ensemble(
            ensemble,
            long_threshold=args.long_threshold,
            short_threshold=args.short_threshold,
            atr_stop_multiple=args.atr_stop_multiple,
        ),
        "selected_grid_row": selected_row,
        "output_predictions": str(prediction_path),
        "output_tradestation": str(tradestation_path) if tradestation_path else None,
    }
    metadata_path = Path(args.metadata) if args.metadata else prediction_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
