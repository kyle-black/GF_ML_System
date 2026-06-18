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
    from gf_ml_system.model import XGBoostClassifier, binary_metrics
else:
    from .export_tradestation_backtest import build_easylanguage_backtest_export
    from .model import XGBoostClassifier, binary_metrics


UP_COLUMN = "prob_up_oc_0_25atr"
DOWN_COLUMN = "prob_down_oc_0_25atr"
MICRO_PROB_COLUMN = "micro_rally_probability"
MICRO_SIGNAL_COLUMN = "micro_rally_signal"
MICRO_TARGET_COLUMN = "micro_rally_target"
MICRO_ADVERSE_COLUMN = "next_open_to_low_adverse_atr"


@dataclass(frozen=True)
class SourcePrediction:
    name: str
    path: Path


@dataclass(frozen=True)
class MicroRallyConfig:
    base_predictions: Path
    training_dataset: Path
    source_predictions: tuple[SourcePrediction, ...]
    output_predictions: Path
    output_tradestation: Path | None = None
    metadata: Path | None = None
    long_threshold: float = 0.425
    short_threshold: float = 0.45
    micro_threshold: float = 0.55
    output_probability: float = 0.426
    target_return_atr: float = 0.50
    max_adverse_atr: float = 1.00
    min_train_rows: int = 400
    rolling_train_rows: int = 500
    step_rows: int = 20
    xgb_eta: float = 0.03
    xgb_max_depth: int = 2
    xgb_num_boost_round: int = 120
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_min_child_weight: float = 10.0
    xgb_reg_lambda: float = 10.0
    xgb_reg_alpha: float = 0.0
    xgb_device: str = "cpu"
    zero_short_on_micro_signal: bool = True
    with_header: bool = False

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["base_predictions"] = str(self.base_predictions)
        payload["training_dataset"] = str(self.training_dataset)
        payload["source_predictions"] = [
            {"name": source.name, "path": str(source.path)}
            for source in self.source_predictions
        ]
        payload["output_predictions"] = str(self.output_predictions)
        payload["output_tradestation"] = str(self.output_tradestation) if self.output_tradestation else None
        payload["metadata"] = str(self.metadata) if self.metadata else None
        return payload


def _parse_source_prediction(value: str) -> SourcePrediction:
    if "=" not in value:
        raise ValueError("--source-prediction values must look like name=/path/to/walk_forward_predictions.csv")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("Source prediction name cannot be empty.")
    return SourcePrediction(name=name, path=Path(raw_path.strip()))


def _read_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    frame = pd.read_csv(path)
    missing = {"signal_date", UP_COLUMN, DOWN_COLUMN} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="coerce")
    frame = frame.loc[frame["signal_date"].notna()].sort_values("signal_date").reset_index(drop=True)
    return frame


def _read_training_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Training dataset not found: {path}")
    frame = pd.read_csv(path)
    if "signal_date" not in frame.columns:
        raise ValueError(f"{path} must include signal_date.")
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="coerce")
    frame = frame.loc[frame["signal_date"].notna()].sort_values("signal_date").reset_index(drop=True)
    return frame


def _add_source_features(frame: pd.DataFrame, source: SourcePrediction) -> pd.DataFrame:
    predictions = _read_predictions(source.path).loc[:, ["signal_date", UP_COLUMN, DOWN_COLUMN]].rename(
        columns={
            UP_COLUMN: f"{source.name}__up_probability",
            DOWN_COLUMN: f"{source.name}__down_probability",
        },
    )
    output = frame.merge(predictions, on="signal_date", how="left")
    up = pd.to_numeric(output[f"{source.name}__up_probability"], errors="coerce").fillna(0.0)
    down = pd.to_numeric(output[f"{source.name}__down_probability"], errors="coerce").fillna(0.0)
    output[f"{source.name}__probability_margin"] = up - down
    output[f"{source.name}__probability_sum"] = up + down
    return output


def _base_signal_masks(frame: pd.DataFrame, config: MicroRallyConfig) -> tuple[pd.Series, pd.Series, pd.Series]:
    up = pd.to_numeric(frame[UP_COLUMN], errors="coerce").fillna(0.0)
    down = pd.to_numeric(frame[DOWN_COLUMN], errors="coerce").fillna(0.0)
    long_signal = up.ge(config.long_threshold) & up.gt(down)
    short_signal = down.ge(config.short_threshold) & down.gt(up)
    flat = ~(long_signal | short_signal)
    return long_signal, short_signal, flat


def _add_micro_target(frame: pd.DataFrame, config: MicroRallyConfig) -> pd.DataFrame:
    output = frame.copy()
    next_open = pd.to_numeric(output["next_open"], errors="coerce")
    target_close = pd.to_numeric(output["target_close"], errors="coerce")
    next_low_move_atr = pd.to_numeric(output["next_low_move_atr"], errors="coerce")
    atr = pd.to_numeric(output["atr_points"], errors="coerce")
    next_low = target_close - next_low_move_atr * atr
    output[MICRO_ADVERSE_COLUMN] = (next_open - next_low) / atr.replace(0.0, np.nan)
    return_atr = pd.to_numeric(output["next_open_to_close_return_atr"], errors="coerce")
    output[MICRO_TARGET_COLUMN] = (
        return_atr.ge(config.target_return_atr)
        & output[MICRO_ADVERSE_COLUMN].le(config.max_adverse_atr)
    ).astype(float)
    valid = return_atr.notna() & output[MICRO_ADVERSE_COLUMN].notna()
    output.loc[~valid, MICRO_TARGET_COLUMN] = np.nan
    return output


def _eligible_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "signal_date",
        "target_close",
        "target_high",
        "target_low",
        "next_open",
        "next_high",
        "next_low",
        "next_close",
        "core_long_signal",
        "core_short_signal",
        "core_flat",
        MICRO_PROB_COLUMN,
        MICRO_SIGNAL_COLUMN,
        MICRO_TARGET_COLUMN,
        MICRO_ADVERSE_COLUMN,
    }
    excluded_prefixes = (
        "next_",
        "target_up",
        "target_down",
    )
    feature_columns: list[str] = []
    for column in frame.columns:
        if column in excluded:
            continue
        if column.endswith("__training"):
            continue
        if column.startswith(excluded_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            feature_columns.append(column)
    return feature_columns


def _build_frame(config: MicroRallyConfig) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    base = _read_predictions(config.base_predictions)
    training = _read_training_dataset(config.training_dataset)

    _, _, flat = _base_signal_masks(base, config)
    base_features = base.loc[:, ["signal_date", UP_COLUMN, DOWN_COLUMN]].copy()
    base_features = base_features.rename(
        columns={
            UP_COLUMN: "core__up_probability",
            DOWN_COLUMN: "core__down_probability",
        },
    )
    base_features["core__probability_margin"] = (
        pd.to_numeric(base_features["core__up_probability"], errors="coerce").fillna(0.0)
        - pd.to_numeric(base_features["core__down_probability"], errors="coerce").fillna(0.0)
    )
    base_features["core__probability_sum"] = (
        pd.to_numeric(base_features["core__up_probability"], errors="coerce").fillna(0.0)
        + pd.to_numeric(base_features["core__down_probability"], errors="coerce").fillna(0.0)
    )

    merged = base.merge(training, on="signal_date", how="left", suffixes=("", "__training"))
    merged = merged.merge(base_features, on="signal_date", how="left")
    for source in config.source_predictions:
        merged = _add_source_features(merged, source)

    long_signal, short_signal, flat = _base_signal_masks(merged, config)
    merged["core_long_signal"] = long_signal.astype(float)
    merged["core_short_signal"] = short_signal.astype(float)
    merged["core_flat"] = flat.astype(float)
    merged = _add_micro_target(merged, config)
    feature_columns = _eligible_feature_columns(merged)
    return base, merged, feature_columns


def _fit_model(config: MicroRallyConfig) -> XGBoostClassifier:
    return XGBoostClassifier(
        eta=config.xgb_eta,
        max_depth=config.xgb_max_depth,
        num_boost_round=config.xgb_num_boost_round,
        subsample=config.xgb_subsample,
        colsample_bytree=config.xgb_colsample_bytree,
        min_child_weight=config.xgb_min_child_weight,
        reg_lambda=config.xgb_reg_lambda,
        reg_alpha=config.xgb_reg_alpha,
        device=config.xgb_device,
    )


def _walk_forward_micro_probability(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    config: MicroRallyConfig,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    probabilities = pd.Series(0.0, index=frame.index, dtype=float)
    eligible = frame["core_flat"].eq(1.0) & frame[MICRO_TARGET_COLUMN].notna()
    folds: list[dict[str, Any]] = []
    importances: list[pd.DataFrame] = []

    for test_start in range(0, len(frame), config.step_rows):
        train = frame.iloc[:test_start].loc[eligible.iloc[:test_start]].copy()
        if config.rolling_train_rows > 0:
            train = train.tail(config.rolling_train_rows)
        if len(train) < config.min_train_rows:
            continue

        test_slice = frame.iloc[test_start : test_start + config.step_rows]
        test = test_slice.loc[eligible.iloc[test_start : test_start + config.step_rows]].copy()
        if test.empty:
            continue

        model = _fit_model(config).fit(train, feature_columns, MICRO_TARGET_COLUMN)
        fold_probabilities = model.predict_proba(test, feature_columns)
        probabilities.loc[test.index] = fold_probabilities
        metrics = binary_metrics(test[MICRO_TARGET_COLUMN].to_numpy(dtype=float), fold_probabilities)
        folds.append(
            {
                "fold": len(folds) + 1,
                "train_start_date": train["signal_date"].iloc[0].date().isoformat(),
                "train_end_date": train["signal_date"].iloc[-1].date().isoformat(),
                "test_start_date": test["signal_date"].iloc[0].date().isoformat(),
                "test_end_date": test["signal_date"].iloc[-1].date().isoformat(),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "train_positive_rate": float(train[MICRO_TARGET_COLUMN].mean()),
                "test_positive_rate": float(test[MICRO_TARGET_COLUMN].mean()),
                **{f"test_{key}": value for key, value in metrics.items()},
            },
        )
        fold_importance = pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": [
                    model._importance_by_feature(feature_columns).get(feature, 0.0)
                    for feature in feature_columns
                ],
                "fold": len(folds),
            },
        )
        importances.append(fold_importance)

    folds_frame = pd.DataFrame(folds)
    if importances:
        importance_frame = (
            pd.concat(importances, ignore_index=True)
            .groupby("feature", as_index=False)["importance"]
            .mean()
            .sort_values("importance", ascending=False)
        )
    else:
        importance_frame = pd.DataFrame(columns=["feature", "importance"])
    return probabilities, folds_frame, importance_frame


def apply_micro_rally_overlay(config: MicroRallyConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    base, frame, feature_columns = _build_frame(config)
    micro_probability, folds, importance = _walk_forward_micro_probability(
        frame,
        feature_columns=feature_columns,
        config=config,
    )

    output = base.copy()
    output[MICRO_PROB_COLUMN] = micro_probability.reindex(output.index).fillna(0.0)
    output[MICRO_TARGET_COLUMN] = frame[MICRO_TARGET_COLUMN].reindex(output.index)
    output[MICRO_ADVERSE_COLUMN] = frame[MICRO_ADVERSE_COLUMN].reindex(output.index)
    output["core_flat"] = frame["core_flat"].reindex(output.index).fillna(0.0)

    micro_signal = output["core_flat"].eq(1.0) & output[MICRO_PROB_COLUMN].ge(config.micro_threshold)
    output[MICRO_SIGNAL_COLUMN] = micro_signal.astype(int)
    if micro_signal.any():
        output.loc[micro_signal, UP_COLUMN] = np.maximum(
            pd.to_numeric(output.loc[micro_signal, UP_COLUMN], errors="coerce").fillna(0.0),
            config.output_probability,
        )
        if config.zero_short_on_micro_signal:
            output.loc[micro_signal, DOWN_COLUMN] = 0.0
    output["micro_rally_action"] = "base"
    output.loc[micro_signal, "micro_rally_action"] = "micro_long"

    valid_prediction_mask = output[MICRO_PROB_COLUMN].gt(0.0) & output[MICRO_TARGET_COLUMN].notna()
    diagnostics = {
        "feature_count": int(len(feature_columns)),
        "fold_count": int(len(folds)),
        "micro_probability_rows": int(valid_prediction_mask.sum()),
        "micro_signal_rows": int(micro_signal.sum()),
        "micro_signal_positive_rate": (
            float(output.loc[micro_signal, MICRO_TARGET_COLUMN].mean())
            if micro_signal.any()
            else None
        ),
        "micro_signal_avg_next_open_to_close_return_atr": (
            float(pd.to_numeric(output.loc[micro_signal, "next_open_to_close_return_atr"], errors="coerce").mean())
            if micro_signal.any()
            else None
        ),
        "micro_signal_caught_05atr_rallies": int(
            (
                micro_signal
                & pd.to_numeric(output["next_open_to_close_return_atr"], errors="coerce").ge(0.50)
            ).sum(),
        ),
        "overall_micro_model_metrics": binary_metrics(
            output.loc[valid_prediction_mask, MICRO_TARGET_COLUMN].to_numpy(dtype=float),
            output.loc[valid_prediction_mask, MICRO_PROB_COLUMN].to_numpy(dtype=float),
        ),
    }
    return output, folds, importance, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a flat-day micro-rally overlay for ES open-to-close ML signals.")
    parser.add_argument("--base-predictions", required=True)
    parser.add_argument("--training-dataset", required=True)
    parser.add_argument("--source-prediction", action="append", default=[])
    parser.add_argument("--output-predictions", required=True)
    parser.add_argument("--output-tradestation", default="")
    parser.add_argument("--metadata", default="")
    parser.add_argument("--long-threshold", type=float, default=0.425)
    parser.add_argument("--short-threshold", type=float, default=0.45)
    parser.add_argument("--micro-threshold", type=float, default=0.55)
    parser.add_argument("--output-probability", type=float, default=0.426)
    parser.add_argument("--target-return-atr", type=float, default=0.50)
    parser.add_argument("--max-adverse-atr", type=float, default=1.00)
    parser.add_argument("--min-train-rows", type=int, default=400)
    parser.add_argument("--rolling-train-rows", type=int, default=500)
    parser.add_argument("--step-rows", type=int, default=20)
    parser.add_argument("--xgb-eta", type=float, default=0.03)
    parser.add_argument("--xgb-max-depth", type=int, default=2)
    parser.add_argument("--xgb-num-boost-round", type=int, default=120)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--xgb-min-child-weight", type=float, default=10.0)
    parser.add_argument("--xgb-reg-lambda", type=float, default=10.0)
    parser.add_argument("--xgb-reg-alpha", type=float, default=0.0)
    parser.add_argument("--xgb-device", default="cpu")
    parser.add_argument("--keep-short-probability", action="store_true")
    parser.add_argument("--with-header", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MicroRallyConfig(
        base_predictions=Path(args.base_predictions),
        training_dataset=Path(args.training_dataset),
        source_predictions=tuple(_parse_source_prediction(value) for value in args.source_prediction),
        output_predictions=Path(args.output_predictions),
        output_tradestation=Path(args.output_tradestation) if args.output_tradestation else None,
        metadata=Path(args.metadata) if args.metadata else None,
        long_threshold=args.long_threshold,
        short_threshold=args.short_threshold,
        micro_threshold=args.micro_threshold,
        output_probability=args.output_probability,
        target_return_atr=args.target_return_atr,
        max_adverse_atr=args.max_adverse_atr,
        min_train_rows=args.min_train_rows,
        rolling_train_rows=args.rolling_train_rows,
        step_rows=args.step_rows,
        xgb_eta=args.xgb_eta,
        xgb_max_depth=args.xgb_max_depth,
        xgb_num_boost_round=args.xgb_num_boost_round,
        xgb_subsample=args.xgb_subsample,
        xgb_colsample_bytree=args.xgb_colsample_bytree,
        xgb_min_child_weight=args.xgb_min_child_weight,
        xgb_reg_lambda=args.xgb_reg_lambda,
        xgb_reg_alpha=args.xgb_reg_alpha,
        xgb_device=args.xgb_device,
        zero_short_on_micro_signal=not args.keep_short_probability,
        with_header=args.with_header,
    )
    output, folds, importance, diagnostics = apply_micro_rally_overlay(config)
    config.output_predictions.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(config.output_predictions, index=False)
    if config.output_tradestation:
        config.output_tradestation.parent.mkdir(parents=True, exist_ok=True)
        build_easylanguage_backtest_export(output).to_csv(
            config.output_tradestation,
            index=False,
            header=config.with_header,
        )

    folds_path = config.output_predictions.with_name(f"{config.output_predictions.stem}_folds.csv")
    importance_path = config.output_predictions.with_name(f"{config.output_predictions.stem}_feature_importance.csv")
    folds.to_csv(folds_path, index=False)
    importance.to_csv(importance_path, index=False)

    metadata = {
        "config": config.to_json_dict(),
        "diagnostics": diagnostics,
        "output_predictions": str(config.output_predictions),
        "output_tradestation": str(config.output_tradestation) if config.output_tradestation else None,
        "folds": str(folds_path),
        "feature_importance": str(importance_path),
    }
    metadata_path = config.metadata or config.output_predictions.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
