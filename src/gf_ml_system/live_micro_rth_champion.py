from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from gf_ml_system.config import (
        DEFAULT_DATA_ROOTS,
        DEFAULT_INTRADAY_60M_ROOTS,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_SYMBOLS,
        PROJECT_ROOT,
    )
    from gf_ml_system.data import parse_data_roots, parse_file_map, parse_symbol_list
    from gf_ml_system.export_tradestation_backtest import (
        EASYLANGUAGE_COLUMNS,
        build_easylanguage_backtest_export,
    )
    from gf_ml_system.model import WalkForwardConfig, XGBoostClassifier, build_dataset
    from gf_ml_system.regime_overlay import OverlayConfig, build_stress_features
else:
    from .config import DEFAULT_DATA_ROOTS, DEFAULT_INTRADAY_60M_ROOTS, DEFAULT_OUTPUT_DIR, DEFAULT_SYMBOLS, PROJECT_ROOT
    from .data import parse_data_roots, parse_file_map, parse_symbol_list
    from .export_tradestation_backtest import EASYLANGUAGE_COLUMNS, build_easylanguage_backtest_export
    from .model import WalkForwardConfig, XGBoostClassifier, build_dataset
    from .regime_overlay import OverlayConfig, build_stress_features


NY_TIMEZONE = ZoneInfo("America/New_York")
CHAMPION_NAME = "micro_rth_esret003_close70_nqpos"
UP_COLUMN = "prob_up_oc_0_25atr"
DOWN_COLUMN = "prob_down_oc_0_25atr"
UP_LABEL = "target_up_oc_0_25atr"
DOWN_LABEL = "target_down_oc_0_25atr"
MULTIHORIZON_ROOT = DEFAULT_OUTPUT_DIR / "es_next_open_to_close_025atr_tes_daily_rth60_xgboost_wf20_multihorizon"
DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_DIR / f"live_{CHAMPION_NAME}"
HISTORICAL_TRADESTATION = MULTIHORIZON_ROOT / f"tradestation_easylanguage_backtest_{CHAMPION_NAME}.csv"
HISTORICAL_META_CANDIDATES = MULTIHORIZON_ROOT / "walk_forward_predictions_true_ensemble_meta_filter.csv"

SOURCE_RUNS = {
    "train150": {
        "artifact": DEFAULT_OUTPUT_DIR / "es_next_open_to_close_025atr_tes_daily_rth60_xgboost_wf20_train150" / "model_artifact.json",
        "train_rows": 150,
    },
    "train250": {
        "artifact": DEFAULT_OUTPUT_DIR / "es_next_open_to_close_025atr_tes_daily_rth60_xgboost_wf20_train250" / "model_artifact.json",
        "train_rows": 250,
    },
    "train500": {
        "artifact": DEFAULT_OUTPUT_DIR / "es_next_open_to_close_025atr_tes_daily_rth60_xgboost_wf20_train500" / "model_artifact.json",
        "train_rows": 500,
    },
    "train750": {
        "artifact": DEFAULT_OUTPUT_DIR / "es_next_open_to_close_025atr_tes_daily_rth60_xgboost_wf20" / "model_artifact.json",
        "train_rows": 750,
    },
}
ENSEMBLE_UP_WEIGHTS = {"train150": 0.0, "train250": 0.6, "train500": 0.3, "train750": 0.1}
ENSEMBLE_DOWN_WEIGHTS = {"train150": 0.0, "train250": 0.4, "train500": 0.4, "train750": 0.2}
AGREEMENT_RUNS = ("train150", "train250", "train500")

META_LONG_THRESHOLD = 0.425
META_SHORT_THRESHOLD = 0.45
META_PROB_THRESHOLD = 0.475
META_TARGET_COLUMN = "meta_target_win_1atr_stop"
META_CONFIG = {
    "long": {"min_train_rows": 500, "rolling_train_rows": 1000},
    "short": {"min_train_rows": 180, "rolling_train_rows": 600},
}
META_XGB_PARAMS = {
    "eta": 0.05,
    "max_depth": 2,
    "num_boost_round": 60,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 12.0,
    "reg_lambda": 20.0,
    "reg_alpha": 1.0,
    "device": "cpu",
}

MICRO_OUTPUT_PROBABILITY = 0.426
MICRO_ES_RTH_RETURN_MIN = 0.003
MICRO_ES_RTH_CLOSE_POSITION_MIN = 0.70
MICRO_NQ_RTH_RETURN_MIN = 0.0
CURRENT_DAY_READY_TIME = time(hour=16, minute=5)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return str(value)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return json.loads(path.read_text())


def _model_from_artifact_config(config: dict[str, Any]) -> XGBoostClassifier:
    return XGBoostClassifier(
        eta=float(config.get("xgb_eta", 0.05)),
        max_depth=int(config.get("xgb_max_depth", 3)),
        num_boost_round=int(config.get("xgb_num_boost_round", 120)),
        subsample=float(config.get("xgb_subsample", 0.8)),
        colsample_bytree=float(config.get("xgb_colsample_bytree", 0.8)),
        min_child_weight=float(config.get("xgb_min_child_weight", 5.0)),
        reg_lambda=float(config.get("xgb_reg_lambda", 5.0)),
        reg_alpha=float(config.get("xgb_reg_alpha", 0.0)),
        device=str(config.get("xgb_device", "cpu")),
    )


def _meta_model() -> XGBoostClassifier:
    return XGBoostClassifier(**META_XGB_PARAMS)


def _rolling_percentile(series: pd.Series, *, window: int, min_periods: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.rolling(window, min_periods=min_periods).apply(
        lambda sample: float(pd.Series(sample).rank(pct=True).iloc[-1]),
        raw=False,
    )


def _run_data_pull(
    *,
    futures: str,
    unit: str,
    interval: int,
    barsback: int,
    output_root: Path,
    include_current_day: bool,
    extra_connector_args: str,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "gf_ml_system.pull_tradestation_data",
        "--futures",
        futures,
        "--unit",
        unit,
        "--interval",
        str(interval),
        "--barsback",
        str(barsback),
        "--output-root",
        str(output_root),
    ]
    if include_current_day:
        command.append("--include-current-day")
    if extra_connector_args:
        command.extend(["--extra-connector-args", extra_connector_args])
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    return {
        "unit": unit,
        "interval": interval,
        "barsback": barsback,
        "output_root": str(output_root),
        "include_current_day_requested": include_current_day,
        "command": command,
    }


def _pull_data_if_requested(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.pull_data:
        return []
    include_current_day = not args.exclude_current_day
    pulls = [
        _run_data_pull(
            futures=args.symbols,
            unit="Daily",
            interval=1,
            barsback=args.daily_barsback,
            output_root=PROJECT_ROOT / "data" / "historical_data",
            include_current_day=include_current_day,
            extra_connector_args=args.extra_connector_args,
        ),
    ]
    if not args.skip_intraday_pull:
        pulls.append(
            _run_data_pull(
                futures=args.symbols,
                unit="Minute",
                interval=60,
                barsback=args.intraday_barsback,
                output_root=PROJECT_ROOT / "data" / "historical_data_60min",
                include_current_day=include_current_day,
                extra_connector_args=args.extra_connector_args,
            ),
        )
    return pulls


def _build_live_dataset(args: argparse.Namespace):
    config = WalkForwardConfig(
        target_symbol="ES",
        symbols=parse_symbol_list(args.symbols),
        data_roots=parse_data_roots(args.data_roots, DEFAULT_DATA_ROOTS),
        data_file_map=parse_file_map(args.data_file_map),
        run_name="live_micro_rth_champion_dataset",
        output_dir=DEFAULT_OUTPUT_DIR,
        start_date=args.start_date,
        end_date=args.end_date or None,
        target_mode="open_close_atr",
        atr_window=10,
        atr_multipliers=(0.25,),
        min_train_rows=750,
        rolling_train_rows=750,
        test_rows=20,
        step_rows=20,
        context_ffill_limit=2,
        include_target_features=True,
        include_enhanced_features=False,
        include_atr_distribution_features=False,
        include_intraday_60m=True,
        allow_missing_intraday_symbols=args.allow_missing_intraday_symbols,
        intraday_60m_roots=parse_data_roots(args.intraday_60m_roots, DEFAULT_INTRADAY_60M_ROOTS),
        intraday_60m_file_map=parse_file_map(args.intraday_60m_file_map),
        intraday_60m_start_time="10:00",
        intraday_60m_end_time="16:00",
        model_type="xgboost",
    )
    return build_dataset(config)


def _apply_current_day_guard(dataset, args: argparse.Namespace):
    now_ny = datetime.now(NY_TIMEZONE)
    today_iso = now_ny.date().isoformat()
    frame_dates = pd.to_datetime(dataset.frame["signal_date"], errors="coerce")
    latest_signal_date = frame_dates.max()
    guard_info = {
        "today": today_iso,
        "current_day_ready_time": CURRENT_DAY_READY_TIME.strftime("%H:%M"),
        "allow_provisional_current_day": bool(args.allow_provisional_current_day),
        "latest_signal_date_before_guard": latest_signal_date.date().isoformat()
        if pd.notna(latest_signal_date)
        else None,
        "dropped_provisional_current_day": False,
        "cleared_next_outcome_row": None,
    }
    if (
        not args.allow_provisional_current_day
        and pd.notna(latest_signal_date)
        and latest_signal_date.date().isoformat() == today_iso
        and now_ny.time() < CURRENT_DAY_READY_TIME
    ):
        guarded_frame = dataset.frame.loc[frame_dates.dt.date.astype(str) < today_iso].reset_index(drop=True)
        if guarded_frame.empty:
            raise RuntimeError("Current-day guard removed the only available row.")
        guarded_latest_index = guarded_frame.index[-1]
        guarded_latest_date = str(guarded_frame.loc[guarded_latest_index, "signal_date"])
        future_columns = [
            column
            for column in guarded_frame.columns
            if column.startswith("next_") or column.startswith("target_up") or column.startswith("target_down")
        ]
        guarded_frame.loc[guarded_latest_index, future_columns] = np.nan
        dataset = replace(dataset, frame=guarded_frame)
        guard_info["dropped_provisional_current_day"] = True
        guard_info["cleared_next_outcome_row"] = guarded_latest_date
    guarded_latest = pd.to_datetime(dataset.frame["signal_date"], errors="coerce").max()
    guard_info["latest_signal_date_after_guard"] = guarded_latest.date().isoformat() if pd.notna(guarded_latest) else None
    return dataset, guard_info


def _score_source_run(dataset, name: str, run_config: dict[str, Any]) -> dict[str, Any]:
    artifact_path = Path(run_config["artifact"])
    artifact = _read_json(artifact_path)
    feature_columns = list(artifact["feature_columns"])
    label_columns = list(artifact["label_columns"])
    if UP_LABEL not in label_columns or DOWN_LABEL not in label_columns:
        raise ValueError(f"{artifact_path} does not include expected open-to-close 0.25 ATR labels.")

    labeled = dataset.frame.dropna(subset=[UP_LABEL, DOWN_LABEL]).sort_values("signal_date").reset_index(drop=True)
    train_rows = int(run_config["train_rows"])
    train_frame = labeled.tail(train_rows).copy()
    if len(train_frame) < train_rows:
        raise RuntimeError(f"{name} needs {train_rows} labeled rows; only found {len(train_frame)}.")
    latest_frame = dataset.frame.tail(1).copy()
    model_config = artifact.get("config", {})

    probabilities: dict[str, float] = {}
    for label_column in (UP_LABEL, DOWN_LABEL):
        model = _model_from_artifact_config(model_config).fit(train_frame, feature_columns, label_column)
        probabilities[label_column] = float(model.predict_proba(latest_frame, feature_columns)[0])

    return {
        "name": name,
        "artifact": str(artifact_path),
        "train_rows": train_rows,
        "train_start_date": str(train_frame["signal_date"].iloc[0]),
        "train_end_date": str(train_frame["signal_date"].iloc[-1]),
        "feature_count": len(feature_columns),
        "prob_up": probabilities[UP_LABEL],
        "prob_down": probabilities[DOWN_LABEL],
    }


def _score_source_runs(dataset) -> dict[str, dict[str, Any]]:
    return {name: _score_source_run(dataset, name, config) for name, config in SOURCE_RUNS.items()}


def _weighted_probability(scores: dict[str, dict[str, Any]], weights: dict[str, float], key: str) -> float:
    return float(sum(scores[name][key] * weight for name, weight in weights.items()))


def _prediction_base_from_latest(dataset, ensemble_up: float, ensemble_down: float) -> pd.DataFrame:
    latest = dataset.frame.tail(1).copy()
    columns = [
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
        UP_LABEL,
        DOWN_LABEL,
    ]
    output = latest.reindex(columns=columns).copy()
    output[UP_COLUMN] = ensemble_up
    output[DOWN_COLUMN] = ensemble_down
    return output.reset_index(drop=True)


def _add_meta_engineered_columns(frame: pd.DataFrame, source_scores: dict[str, dict[str, Any]]) -> pd.DataFrame:
    output = frame.copy()
    close = pd.to_numeric(output["target_close"], errors="coerce")
    atr = pd.to_numeric(output["atr_points"], errors="coerce")
    output["target_close_return_5"] = close.pct_change(5)
    output["target_close_return_20"] = close.pct_change(20)
    output["atr_pct_of_close"] = atr / close.replace(0.0, np.nan)
    output["atr_rank_252"] = _rolling_percentile(atr, window=252, min_periods=63)
    output["atr_rank_63"] = _rolling_percentile(atr, window=63, min_periods=20)
    output["close_rank_252"] = _rolling_percentile(close, window=252, min_periods=63)

    latest_index = output.index[-1]
    for run_name, prefix in (("train750", "train750"), ("train500", "train500"), ("train250", "train250")):
        output.loc[latest_index, f"{prefix}_up"] = source_scores[run_name]["prob_up"]
        output.loc[latest_index, f"{prefix}_down"] = source_scores[run_name]["prob_down"]

    up_values = np.array([source_scores[name]["prob_up"] for name in ("train750", "train500", "train250")])
    down_values = np.array([source_scores[name]["prob_down"] for name in ("train750", "train500", "train250")])
    output.loc[latest_index, "run_up_mean"] = float(up_values.mean())
    output.loc[latest_index, "run_down_mean"] = float(down_values.mean())
    output.loc[latest_index, "run_up_std"] = float(up_values.std(ddof=0))
    output.loc[latest_index, "run_down_std"] = float(down_values.std(ddof=0))
    output.loc[latest_index, "run_max_prob_mean"] = float(np.maximum(up_values, down_values).mean())
    return output


def _load_meta_training(path: Path) -> tuple[pd.DataFrame, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Meta-filter candidate file not found: {path}")
    training = pd.read_csv(path)
    if "signal_date" not in training.columns:
        raise ValueError(f"{path} must include signal_date.")
    training["signal_date"] = pd.to_datetime(training["signal_date"], errors="coerce")
    training = training.loc[training["signal_date"].notna()].sort_values("signal_date").reset_index(drop=True)

    columns = list(training.columns)
    meta_prob_index = columns.index("meta_prob") if "meta_prob" in columns else len(columns)
    base_feature_columns = ["prob_gap"] + columns[29:meta_prob_index]
    feature_columns = [column for column in base_feature_columns if column in training.columns]
    missing_target = META_TARGET_COLUMN not in training.columns
    if missing_target:
        raise ValueError(f"{path} is missing {META_TARGET_COLUMN}.")
    return training, feature_columns


def _build_live_meta_row(
    dataset,
    source_scores: dict[str, dict[str, Any]],
    ensemble_up: float,
    ensemble_down: float,
) -> pd.DataFrame:
    meta_frame = _add_meta_engineered_columns(dataset.frame.copy(), source_scores)
    row = meta_frame.tail(1).copy().reset_index(drop=True)
    row[UP_COLUMN] = ensemble_up
    row[DOWN_COLUMN] = ensemble_down
    row["up"] = ensemble_up
    row["down"] = ensemble_down
    row["side"] = 1 if ensemble_up >= ensemble_down else -1
    row["side_name"] = "long" if ensemble_up >= ensemble_down else "short"
    row["side_prob"] = max(ensemble_up, ensemble_down)
    row["opp_prob"] = min(ensemble_up, ensemble_down)
    row["prob_gap"] = row["side_prob"] - row["opp_prob"]
    row["prob_sum"] = row["side_prob"] + row["opp_prob"]
    row["prob_ratio"] = row["side_prob"] / row["opp_prob"].replace(0.0, np.nan)
    row["is_long_side"] = float(row["side_name"].iloc[0] == "long")
    row["is_short_side"] = float(row["side_name"].iloc[0] == "short")

    side_is_long = bool(row["side_name"].iloc[0] == "long")
    agreement_count = 0
    for name in ("train750", "train500", "train250"):
        if side_is_long and source_scores[name]["prob_up"] > source_scores[name]["prob_down"]:
            agreement_count += 1
        if not side_is_long and source_scores[name]["prob_down"] > source_scores[name]["prob_up"]:
            agreement_count += 1
    row["run_side_agree_count"] = float(agreement_count)
    row["side_prob_x_gap"] = row["side_prob"] * row["prob_gap"]
    row["side_prob_x_atr_rank_252"] = row["side_prob"] * row["atr_rank_252"]
    row["gap_x_atr_rank_252"] = row["prob_gap"] * row["atr_rank_252"]
    return row


def _score_meta_filter(
    live_meta_row: pd.DataFrame,
    *,
    training_path: Path,
) -> dict[str, Any]:
    training, feature_columns = _load_meta_training(training_path)
    side = str(live_meta_row["side_name"].iloc[0])
    side_config = META_CONFIG[side]
    train_frame = training.loc[
        training["side_name"].eq(side) & pd.to_numeric(training[META_TARGET_COLUMN], errors="coerce").notna()
    ].tail(side_config["rolling_train_rows"]).copy()
    if len(train_frame) < side_config["min_train_rows"]:
        raise RuntimeError(
            f"Meta filter {side} model needs {side_config['min_train_rows']} rows; found {len(train_frame)}.",
        )

    model = _meta_model().fit(train_frame, feature_columns, META_TARGET_COLUMN)
    meta_prob = float(model.predict_proba(live_meta_row, feature_columns)[0])
    return {
        "side": side,
        "meta_probability": meta_prob,
        "feature_count": len(feature_columns),
        "train_rows": int(len(train_frame)),
        "train_start_date": train_frame["signal_date"].iloc[0].date().isoformat(),
        "train_end_date": train_frame["signal_date"].iloc[-1].date().isoformat(),
        "training_path": str(training_path),
    }


def _apply_meta_gate(prediction: pd.DataFrame, meta_result: dict[str, Any]) -> pd.DataFrame:
    output = prediction.copy()
    source_up = float(output[UP_COLUMN].iloc[0])
    source_down = float(output[DOWN_COLUMN].iloc[0])
    output[UP_COLUMN] = 0.0
    output[DOWN_COLUMN] = 0.0
    side = meta_result["side"]
    meta_prob = float(meta_result["meta_probability"])
    if side == "long" and source_up >= META_LONG_THRESHOLD and meta_prob >= META_PROB_THRESHOLD:
        output[UP_COLUMN] = source_up
    elif side == "short" and source_down >= META_SHORT_THRESHOLD and meta_prob >= META_PROB_THRESHOLD:
        output[DOWN_COLUMN] = source_down
    output["meta_filter_probability"] = meta_prob
    output["meta_filter_side"] = side
    output["meta_filter_action"] = "accepted" if output[UP_COLUMN].iloc[0] > 0 or output[DOWN_COLUMN].iloc[0] > 0 else "rejected"
    return output


def _agreement_counts(source_scores: dict[str, dict[str, Any]]) -> tuple[int, int]:
    long_votes = 0
    short_votes = 0
    for name in AGREEMENT_RUNS:
        if source_scores[name]["prob_up"] > source_scores[name]["prob_down"]:
            long_votes += 1
        elif source_scores[name]["prob_down"] > source_scores[name]["prob_up"]:
            short_votes += 1
    return long_votes, short_votes


def _stress_config() -> OverlayConfig:
    dummy = Path("__live__")
    return OverlayConfig(
        base_predictions=dummy,
        training_dataset=dummy,
        source_predictions=dummy,
        agreement_predictions=(),
        output_predictions=dummy,
        action="switch",
        stress_min_score=5,
        agreement_min=1,
        long_gate=0.45,
        short_gate=0.45,
    )


def _apply_stress_switch(
    prediction: pd.DataFrame,
    dataset,
    *,
    source_up: float,
    source_down: float,
    source_scores: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = prediction.copy()
    stress_base = dataset.frame.loc[:, ["signal_date", "atr_points"]].copy()
    stress_base["signal_date"] = pd.to_datetime(stress_base["signal_date"], errors="coerce")
    stress = build_stress_features(stress_base, dataset.frame, _stress_config())
    latest_stress = stress.tail(1).iloc[0]
    stress_score = int(latest_stress["stress_score"]) if pd.notna(latest_stress["stress_score"]) else 0
    long_votes, short_votes = _agreement_counts(source_scores)
    action = "base"

    if stress_score >= 5:
        output.loc[:, [UP_COLUMN, DOWN_COLUMN]] = 0.0
        action = "switch_rejected"
        if source_up >= 0.45 and source_up > source_down and long_votes >= 1:
            output[UP_COLUMN] = source_up
            action = "switch_accepted"
        elif source_down >= 0.45 and source_down > source_up and short_votes >= 1:
            output[DOWN_COLUMN] = source_down
            action = "switch_accepted"

    output["stress_score"] = stress_score
    output["stress_overlay_action"] = action
    return output, {
        "stress_score": stress_score,
        "stress_overlay_action": action,
        "long_votes": long_votes,
        "short_votes": short_votes,
    }


def _apply_micro_rth_overlay(prediction: pd.DataFrame, dataset) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = prediction.copy()
    latest = dataset.frame.tail(1).iloc[0]
    flat = float(output[UP_COLUMN].iloc[0]) == 0.0 and float(output[DOWN_COLUMN].iloc[0]) == 0.0
    es_return = float(pd.to_numeric(pd.Series([latest.get("ES__rth60__return")]), errors="coerce").fillna(0.0).iloc[0])
    es_close_position = float(
        pd.to_numeric(pd.Series([latest.get("ES__rth60__close_position")]), errors="coerce").fillna(0.0).iloc[0],
    )
    nq_return = float(pd.to_numeric(pd.Series([latest.get("NQ__rth60__return")]), errors="coerce").fillna(0.0).iloc[0])
    micro_signal = (
        flat
        and es_return > MICRO_ES_RTH_RETURN_MIN
        and es_close_position > MICRO_ES_RTH_CLOSE_POSITION_MIN
        and nq_return > MICRO_NQ_RTH_RETURN_MIN
    )
    if micro_signal:
        output[UP_COLUMN] = MICRO_OUTPUT_PROBABILITY
        output[DOWN_COLUMN] = 0.0
        output["stress_overlay_action"] = "micro_rth_esret003_close70_nqpos"
    output["micro_rth_signal"] = int(micro_signal)
    return output, {
        "micro_rth_signal": bool(micro_signal),
        "core_was_flat": bool(flat),
        "ES__rth60__return": es_return,
        "ES__rth60__close_position": es_close_position,
        "NQ__rth60__return": nq_return,
    }


def _read_tradestation_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, header=None, names=EASYLANGUAGE_COLUMNS)
    if not frame.empty and str(frame.iloc[0]["el_date"]).strip().lower() == "el_date":
        frame = frame.iloc[1:].copy()
    for column in ("el_date", "ts_date"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["date_iso"] = pd.to_datetime(frame["date_iso"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame = frame.loc[frame["date_iso"].notna()].copy()
    for column in EASYLANGUAGE_COLUMNS:
        if column not in {"date_iso"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["el_date"] = frame["el_date"].astype(int)
    frame["ts_date"] = frame["ts_date"].astype(int)
    return frame.loc[:, EASYLANGUAGE_COLUMNS]


def _upsert_master(
    *,
    master_path: Path,
    latest_trade_station: pd.DataFrame,
    initialize_from_history: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    initialized_from_history = False
    if initialize_from_history and HISTORICAL_TRADESTATION.exists():
        frames.append(_read_tradestation_csv(HISTORICAL_TRADESTATION))
        initialized_from_history = True
    if master_path.exists():
        frames.append(_read_tradestation_csv(master_path))
    frames.append(latest_trade_station)

    combined = pd.concat(frames, ignore_index=True)
    before_rows = len(combined)
    combined = combined.drop_duplicates(subset=["date_iso"], keep="last").sort_values("ts_date").reset_index(drop=True)
    master_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(master_path, index=False, header=False)
    combined.to_csv(master_path.with_name(f"{master_path.stem}_with_header.csv"), index=False, header=True)
    return combined, {
        "master_path": str(master_path),
        "master_header_path": str(master_path.with_name(f"{master_path.stem}_with_header.csv")),
        "initialized_from_history": initialized_from_history,
        "rows_before_dedup": int(before_rows),
        "rows_after_dedup": int(len(combined)),
    }


def _write_outputs(
    *,
    output_root: Path,
    run_id: str,
    prediction: pd.DataFrame,
    metadata: dict[str, Any],
    initialize_master_from_history: bool,
) -> dict[str, str]:
    run_dir = output_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    latest_prediction_path = run_dir / "latest_prediction.csv"
    latest_tradestation_path = run_dir / "tradestation_latest.csv"
    master_path = output_root / "tradestation_master.csv"
    master_snapshot_path = run_dir / "tradestation_master_snapshot.csv"
    metadata_path = run_dir / "run_metadata.json"

    prediction.to_csv(latest_prediction_path, index=False)
    tradestation = build_easylanguage_backtest_export(prediction)
    tradestation.to_csv(latest_tradestation_path, index=False, header=False)
    master, master_info = _upsert_master(
        master_path=master_path,
        latest_trade_station=tradestation,
        initialize_from_history=initialize_master_from_history,
    )
    master.to_csv(master_snapshot_path, index=False, header=False)
    master.to_csv(master_snapshot_path.with_name("tradestation_master_snapshot_with_header.csv"), index=False)

    metadata["outputs"] = {
        "run_dir": str(run_dir),
        "latest_prediction": str(latest_prediction_path),
        "tradestation_latest": str(latest_tradestation_path),
        "tradestation_master_snapshot": str(master_snapshot_path),
        "tradestation_master_snapshot_with_header": str(
            master_snapshot_path.with_name("tradestation_master_snapshot_with_header.csv"),
        ),
        **master_info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=_json_default))

    prediction.to_csv(output_root / "latest_prediction.csv", index=False)
    tradestation.to_csv(output_root / "tradestation_latest.csv", index=False, header=False)
    (output_root / "latest_run_metadata.json").write_text(json.dumps(metadata, indent=2, default=_json_default))
    return {
        "run_dir": str(run_dir),
        "latest_prediction": str(latest_prediction_path),
        "tradestation_latest": str(latest_tradestation_path),
        "tradestation_master": str(master_path),
        "metadata": str(metadata_path),
    }


def build_live_prediction(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    pulls = _pull_data_if_requested(args)
    dataset = _build_live_dataset(args)
    dataset, current_day_guard = _apply_current_day_guard(dataset, args)
    source_scores = _score_source_runs(dataset)
    ensemble_up = _weighted_probability(source_scores, ENSEMBLE_UP_WEIGHTS, "prob_up")
    ensemble_down = _weighted_probability(source_scores, ENSEMBLE_DOWN_WEIGHTS, "prob_down")
    source_prediction = _prediction_base_from_latest(dataset, ensemble_up, ensemble_down)

    live_meta_row = _build_live_meta_row(dataset, source_scores, ensemble_up, ensemble_down)
    meta_result = _score_meta_filter(live_meta_row, training_path=Path(args.meta_training))
    prediction = _apply_meta_gate(source_prediction, meta_result)
    prediction, stress_result = _apply_stress_switch(
        prediction,
        dataset,
        source_up=ensemble_up,
        source_down=ensemble_down,
        source_scores=source_scores,
    )
    prediction, micro_result = _apply_micro_rth_overlay(prediction, dataset)

    now_ny = datetime.now(NY_TIMEZONE)
    latest_signal_date = str(prediction["signal_date"].iloc[0])
    today_iso = now_ny.date().isoformat()
    include_current_day_requested = bool(args.pull_data and not args.exclude_current_day)
    latest_signal_date_is_today = latest_signal_date == today_iso
    metadata = {
        "champion": CHAMPION_NAME,
        "created_at_ny": now_ny.isoformat(),
        "latest_signal_date": latest_signal_date,
        "latest_signal_date_is_today": latest_signal_date_is_today,
        "prediction_is_for_next_session_open_to_close": True,
        "pulls": pulls,
        "include_current_day_requested": include_current_day_requested,
        "provisional_current_day": bool(
            include_current_day_requested
            and latest_signal_date_is_today
            and now_ny.time() < CURRENT_DAY_READY_TIME
        ),
        "current_day_guard": current_day_guard,
        "dataset_rows": int(len(dataset.frame)),
        "dataset_source_files": dataset.source_files,
        "source_scores": source_scores,
        "ensemble": {
            "up_weights": ENSEMBLE_UP_WEIGHTS,
            "down_weights": ENSEMBLE_DOWN_WEIGHTS,
            "prob_up": ensemble_up,
            "prob_down": ensemble_down,
        },
        "meta_filter": meta_result,
        "stress_filter": stress_result,
        "micro_rth_overlay": micro_result,
        "final_prob_up_oc_0_25atr": float(prediction[UP_COLUMN].iloc[0]),
        "final_prob_down_oc_0_25atr": float(prediction[DOWN_COLUMN].iloc[0]),
    }
    return prediction, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen ES champion: true multi-horizon ensemble, true meta filter, "
            "stress switch, and micro_rth_esret003_close70_nqpos overlay."
        ),
    )
    parser.add_argument("--pull-data", action="store_true", help="Refresh daily and 60-minute TradeStation DP files before scoring.")
    parser.add_argument(
        "--exclude-current-day",
        action="store_true",
        help="When pulling data, omit TradeStation's current-day bar. By default live pulls include it.",
    )
    parser.add_argument("--skip-intraday-pull", action="store_true", help="Only pull daily data before scoring.")
    parser.add_argument("--daily-barsback", type=int, default=20000)
    parser.add_argument("--intraday-barsback", type=int, default=1000)
    parser.add_argument("--extra-connector-args", default="")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--data-roots", default=",".join(str(path) for path in DEFAULT_DATA_ROOTS))
    parser.add_argument("--data-file-map", default="")
    parser.add_argument("--intraday-60m-roots", default=",".join(str(path) for path in DEFAULT_INTRADAY_60M_ROOTS))
    parser.add_argument("--intraday-60m-file-map", default="")
    parser.add_argument("--allow-missing-intraday-symbols", action="store_true")
    parser.add_argument("--start-date", default="2008-01-01")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--meta-training", default=str(HISTORICAL_META_CANDIDATES))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--no-initialize-master-from-history",
        action="store_true",
        help="Do not seed the master CSV from the frozen historical TradeStation backtest.",
    )
    parser.add_argument(
        "--allow-provisional-current-day",
        action="store_true",
        help="Allow scoring today's row before the 16:05 New York current-day guard has passed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id or datetime.now(NY_TIMEZONE).strftime("%Y-%m-%d_%H%M%S")
    prediction, metadata = build_live_prediction(args)
    paths = _write_outputs(
        output_root=Path(args.output_root),
        run_id=run_id,
        prediction=prediction,
        metadata=metadata,
        initialize_master_from_history=not args.no_initialize_master_from_history,
    )
    print(json.dumps({"paths": paths, "metadata": metadata}, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
