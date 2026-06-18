from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from gf_ml_system.export_tradestation_backtest import build_easylanguage_backtest_export
else:
    from .export_tradestation_backtest import build_easylanguage_backtest_export


UP_COLUMN = "prob_up_oc_0_25atr"
DOWN_COLUMN = "prob_down_oc_0_25atr"
STRESS_FEATURE_COLUMNS = [
    "ES__rth60__range_pct",
    "ES__rth60__return",
    "NQ__rth60__range_pct",
    "NQ__rth60__return",
    "TY__rth60__range_pct",
    "TY__rth60__return",
    "US__rth60__range_pct",
    "US__rth60__return",
    "GC__rth60__range_pct",
    "GC__rth60__return",
    "SI__rth60__range_pct",
    "SI__rth60__return",
]


@dataclass(frozen=True)
class OverlayConfig:
    base_predictions: Path
    training_dataset: Path
    source_predictions: Path
    agreement_predictions: tuple[Path, ...]
    output_predictions: Path
    output_tradestation: Path | None = None
    metadata: Path | None = None
    action: str = "switch"
    stress_min_score: int = 5
    agreement_min: int = 1
    long_gate: float = 0.45
    short_gate: float = 0.45
    percentile_window: int = 750
    percentile_min_periods: int = 250
    atr_percentile_gate: float = 0.90
    es_range_percentile_gate: float = 0.90
    es_abs_return_percentile_gate: float = 0.90
    nq_abs_return_percentile_gate: float = 0.90
    cross_asset_range_percentile_gate: float = 0.75
    with_header: bool = False

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in [
            "base_predictions",
            "training_dataset",
            "source_predictions",
            "output_predictions",
            "output_tradestation",
            "metadata",
        ]:
            value = payload[key]
            payload[key] = str(value) if value is not None else None
        payload["agreement_predictions"] = [str(path) for path in self.agreement_predictions]
        return payload


def _normalize_action(action: str) -> str:
    normalized = action.strip().lower().replace("-", "_")
    if normalized not in {"switch", "add", "filter"}:
        raise ValueError("--action must be one of: switch, add, filter")
    return normalized


def _read_predictions(path: Path, *, prefix: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"signal_date", UP_COLUMN, DOWN_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="coerce")
    frame = frame.loc[frame["signal_date"].notna()].sort_values("signal_date").reset_index(drop=True)
    if prefix:
        frame = frame.loc[:, ["signal_date", UP_COLUMN, DOWN_COLUMN]].rename(
            columns={
                UP_COLUMN: f"{prefix}__{UP_COLUMN}",
                DOWN_COLUMN: f"{prefix}__{DOWN_COLUMN}",
            },
        )
    return frame


def _rolling_percentile(series: pd.Series, *, window: int, min_periods: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.rolling(window, min_periods=min_periods).apply(
        lambda array: float(np.mean(array <= array[-1])),
        raw=True,
    )


def build_stress_features(base: pd.DataFrame, training_dataset: pd.DataFrame, config: OverlayConfig) -> pd.DataFrame:
    training = training_dataset.copy()
    training["signal_date"] = pd.to_datetime(training["signal_date"], errors="coerce")
    missing = set(STRESS_FEATURE_COLUMNS) - set(training.columns)
    if missing:
        raise ValueError(f"Training dataset is missing stress feature columns: {sorted(missing)}")

    features = base.loc[:, ["signal_date", "atr_points"]].copy()
    features = features.merge(
        training.loc[:, ["signal_date", *STRESS_FEATURE_COLUMNS]],
        on="signal_date",
        how="left",
    )
    window_kwargs = {
        "window": config.percentile_window,
        "min_periods": config.percentile_min_periods,
    }
    features["atr_percentile"] = _rolling_percentile(features["atr_points"], **window_kwargs)
    features["es_rth_range_percentile"] = _rolling_percentile(features["ES__rth60__range_pct"], **window_kwargs)
    features["es_rth_abs_return_percentile"] = _rolling_percentile(
        features["ES__rth60__return"].abs(),
        **window_kwargs,
    )
    features["nq_rth_abs_return_percentile"] = _rolling_percentile(
        features["NQ__rth60__return"].abs(),
        **window_kwargs,
    )
    range_percentile_columns: list[str] = []
    for symbol in ("ES", "NQ", "TY", "US", "GC", "SI"):
        column = f"{symbol}__rth60__range_pct"
        percentile_column = f"{symbol.lower()}_rth_range_percentile"
        features[percentile_column] = _rolling_percentile(features[column], **window_kwargs)
        range_percentile_columns.append(percentile_column)
    features["cross_asset_range_percentile_mean"] = features[range_percentile_columns].mean(axis=1)

    features["stress_score"] = 0
    features["stress_score"] += (features["atr_percentile"] >= config.atr_percentile_gate).astype(int)
    features["stress_score"] += (
        features["es_rth_range_percentile"] >= config.es_range_percentile_gate
    ).astype(int)
    features["stress_score"] += (
        features["es_rth_abs_return_percentile"] >= config.es_abs_return_percentile_gate
    ).astype(int)
    features["stress_score"] += (
        features["nq_rth_abs_return_percentile"] >= config.nq_abs_return_percentile_gate
    ).astype(int)
    features["stress_score"] += (
        features["cross_asset_range_percentile_mean"] >= config.cross_asset_range_percentile_gate
    ).astype(int)
    return features


def _agreement_counts(frame: pd.DataFrame, agreement_prefixes: list[str]) -> tuple[pd.Series, pd.Series]:
    long_votes = pd.Series(0, index=frame.index, dtype=int)
    short_votes = pd.Series(0, index=frame.index, dtype=int)
    for prefix in agreement_prefixes:
        up = pd.to_numeric(frame[f"{prefix}__{UP_COLUMN}"], errors="coerce").fillna(0.0)
        down = pd.to_numeric(frame[f"{prefix}__{DOWN_COLUMN}"], errors="coerce").fillna(0.0)
        long_votes += (up > down).astype(int)
        short_votes += (down > up).astype(int)
    return long_votes, short_votes


def apply_overlay(config: OverlayConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    action = _normalize_action(config.action)
    base = _read_predictions(config.base_predictions)
    source = _read_predictions(config.source_predictions, prefix="source")
    training = pd.read_csv(config.training_dataset)
    stress = build_stress_features(base, training, config)

    frame = base.merge(
        stress.loc[:, ["signal_date", "stress_score"]],
        on="signal_date",
        how="left",
    )
    frame = frame.merge(source, on="signal_date", how="left")

    agreement_prefixes: list[str] = []
    for index, path in enumerate(config.agreement_predictions, start=1):
        prefix = f"agree{index}"
        frame = frame.merge(_read_predictions(path, prefix=prefix), on="signal_date", how="left")
        agreement_prefixes.append(prefix)

    output = base.copy()
    stress_mask = frame["stress_score"].fillna(0).astype(int) >= config.stress_min_score
    source_up = pd.to_numeric(frame[f"source__{UP_COLUMN}"], errors="coerce").fillna(0.0)
    source_down = pd.to_numeric(frame[f"source__{DOWN_COLUMN}"], errors="coerce").fillna(0.0)
    if agreement_prefixes:
        long_votes, short_votes = _agreement_counts(frame, agreement_prefixes)
    else:
        long_votes = pd.Series(config.agreement_min, index=frame.index, dtype=int)
        short_votes = pd.Series(config.agreement_min, index=frame.index, dtype=int)

    long_ok = (
        stress_mask
        & (source_up >= config.long_gate)
        & (source_up > source_down)
        & (long_votes >= config.agreement_min)
    )
    short_ok = (
        stress_mask
        & (source_down >= config.short_gate)
        & (source_down > source_up)
        & (short_votes >= config.agreement_min)
    )

    if action == "switch":
        output.loc[stress_mask, [UP_COLUMN, DOWN_COLUMN]] = 0.0
        output.loc[long_ok, UP_COLUMN] = source_up.loc[long_ok]
        output.loc[short_ok, DOWN_COLUMN] = source_down.loc[short_ok]
    elif action == "add":
        inactive = stress_mask & output[UP_COLUMN].eq(0.0) & output[DOWN_COLUMN].eq(0.0)
        output.loc[inactive & long_ok, UP_COLUMN] = source_up.loc[inactive & long_ok]
        output.loc[inactive & short_ok, DOWN_COLUMN] = source_down.loc[inactive & short_ok]
    else:
        active_stress = stress_mask & (output[UP_COLUMN].gt(0.0) | output[DOWN_COLUMN].gt(0.0))
        keep = long_ok | short_ok
        output.loc[active_stress & ~keep, [UP_COLUMN, DOWN_COLUMN]] = 0.0

    output["stress_score"] = frame["stress_score"].fillna(0).astype(int)
    output["stress_overlay_action"] = "base"
    output.loc[stress_mask, "stress_overlay_action"] = action
    output.loc[long_ok | short_ok, "stress_overlay_action"] = f"{action}_accepted"
    output.loc[stress_mask & ~(long_ok | short_ok), "stress_overlay_action"] = f"{action}_rejected"

    active_before = base[UP_COLUMN].gt(0.0) | base[DOWN_COLUMN].gt(0.0)
    active_after = output[UP_COLUMN].gt(0.0) | output[DOWN_COLUMN].gt(0.0)
    diagnostics = {
        "stress_rows": int(stress_mask.sum()),
        "stress_active_before": int((stress_mask & active_before).sum()),
        "stress_active_after": int((stress_mask & active_after).sum()),
        "rows_added": int((~active_before & active_after).sum()),
        "rows_removed": int((active_before & ~active_after).sum()),
        "long_accepts": int(long_ok.sum()),
        "short_accepts": int(short_ok.sum()),
        "stress_score_counts": {
            str(int(score)): int(count)
            for score, count in frame["stress_score"].fillna(0).astype(int).value_counts().sort_index().items()
        },
    }
    return output, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a signal-day stress-regime overlay to ML prediction CSVs.")
    parser.add_argument("--base-predictions", required=True)
    parser.add_argument("--training-dataset", required=True)
    parser.add_argument("--source-predictions", required=True)
    parser.add_argument("--agreement-predictions", action="append", default=[])
    parser.add_argument("--output-predictions", required=True)
    parser.add_argument("--output-tradestation", default="")
    parser.add_argument("--metadata", default="")
    parser.add_argument("--action", default="switch", choices=("switch", "add", "filter"))
    parser.add_argument("--stress-min-score", type=int, default=5)
    parser.add_argument("--agreement-min", type=int, default=1)
    parser.add_argument("--long-gate", type=float, default=0.45)
    parser.add_argument("--short-gate", type=float, default=0.45)
    parser.add_argument("--percentile-window", type=int, default=750)
    parser.add_argument("--percentile-min-periods", type=int, default=250)
    parser.add_argument("--atr-percentile-gate", type=float, default=0.90)
    parser.add_argument("--es-range-percentile-gate", type=float, default=0.90)
    parser.add_argument("--es-abs-return-percentile-gate", type=float, default=0.90)
    parser.add_argument("--nq-abs-return-percentile-gate", type=float, default=0.90)
    parser.add_argument("--cross-asset-range-percentile-gate", type=float, default=0.75)
    parser.add_argument("--with-header", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = OverlayConfig(
        base_predictions=Path(args.base_predictions),
        training_dataset=Path(args.training_dataset),
        source_predictions=Path(args.source_predictions),
        agreement_predictions=tuple(Path(path) for path in args.agreement_predictions),
        output_predictions=Path(args.output_predictions),
        output_tradestation=Path(args.output_tradestation) if args.output_tradestation else None,
        metadata=Path(args.metadata) if args.metadata else None,
        action=args.action,
        stress_min_score=args.stress_min_score,
        agreement_min=args.agreement_min,
        long_gate=args.long_gate,
        short_gate=args.short_gate,
        percentile_window=args.percentile_window,
        percentile_min_periods=args.percentile_min_periods,
        atr_percentile_gate=args.atr_percentile_gate,
        es_range_percentile_gate=args.es_range_percentile_gate,
        es_abs_return_percentile_gate=args.es_abs_return_percentile_gate,
        nq_abs_return_percentile_gate=args.nq_abs_return_percentile_gate,
        cross_asset_range_percentile_gate=args.cross_asset_range_percentile_gate,
        with_header=args.with_header,
    )
    output, diagnostics = apply_overlay(config)
    config.output_predictions.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(config.output_predictions, index=False)
    if config.output_tradestation:
        config.output_tradestation.parent.mkdir(parents=True, exist_ok=True)
        build_easylanguage_backtest_export(output).to_csv(
            config.output_tradestation,
            index=False,
            header=config.with_header,
        )

    metadata = {
        "config": config.to_json_dict(),
        "diagnostics": diagnostics,
        "output_predictions": str(config.output_predictions),
        "output_tradestation": str(config.output_tradestation) if config.output_tradestation else None,
    }
    metadata_path = config.metadata or config.output_predictions.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
