from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_DATA_ROOTS,
    DEFAULT_INTRADAY_60M_ROOTS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SYMBOLS,
    DEFAULT_TARGET_SYMBOL,
)
from .data import load_intraday_symbol_frames, load_symbol_frames
from .indicators import TES_INDICATOR_COLUMNS, build_indicator_frame

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - optional dependency
    xgb = None


@dataclass(frozen=True)
class WalkForwardConfig:
    target_symbol: str = DEFAULT_TARGET_SYMBOL
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS
    data_roots: tuple[Path, ...] = DEFAULT_DATA_ROOTS
    data_file_map: dict[str, Path] = field(default_factory=dict)
    run_name: str = "es_next_day_atr_tes_intermarket"
    output_dir: Path = DEFAULT_OUTPUT_DIR
    start_date: str = "2008-01-01"
    end_date: str | None = None
    target_mode: str = "atr_touch"
    atr_window: int = 10
    atr_multipliers: tuple[float, ...] = (0.25, 0.50)
    min_train_rows: int = 750
    rolling_train_rows: int = 750
    test_rows: int = 50
    step_rows: int = 50
    context_ffill_limit: int = 2
    include_target_features: bool = True
    include_enhanced_features: bool = False
    include_atr_distribution_features: bool = False
    include_intraday_60m: bool = False
    allow_missing_intraday_symbols: bool = False
    intraday_60m_roots: tuple[Path, ...] = DEFAULT_INTRADAY_60M_ROOTS
    intraday_60m_file_map: dict[str, Path] = field(default_factory=dict)
    intraday_60m_start_time: str = "10:00"
    intraday_60m_end_time: str = "16:00"
    model_type: str = "ridge"
    ridge_l2: float = 10.0
    xgb_eta: float = 0.05
    xgb_max_depth: int = 3
    xgb_num_boost_round: int = 120
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_min_child_weight: float = 5.0
    xgb_reg_lambda: float = 5.0
    xgb_reg_alpha: float = 0.0
    xgb_device: str = "cpu"
    max_folds: int = 0

    def normalized_symbols(self) -> tuple[str, ...]:
        seen: set[str] = set()
        ordered: list[str] = []
        for symbol in (self.target_symbol, *self.symbols):
            upper = symbol.upper()
            if upper not in seen:
                seen.add(upper)
                ordered.append(upper)
        return tuple(ordered)

    def output_root(self) -> Path:
        return self.output_dir / self.run_name

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["data_roots"] = [str(path) for path in self.data_roots]
        payload["data_file_map"] = {symbol: str(path) for symbol, path in self.data_file_map.items()}
        payload["intraday_60m_roots"] = [str(path) for path in self.intraday_60m_roots]
        payload["intraday_60m_file_map"] = {
            symbol: str(path)
            for symbol, path in self.intraday_60m_file_map.items()
        }
        payload["output_dir"] = str(self.output_dir)
        return payload


@dataclass(frozen=True)
class DatasetBundle:
    frame: pd.DataFrame
    feature_columns: list[str]
    label_columns: list[str]
    probability_columns: dict[str, str]
    source_files: dict[str, str]


class RidgeClassifier:
    def __init__(self, *, l2: float) -> None:
        self.l2 = float(l2)
        self.weights_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.medians_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.stds_: np.ndarray | None = None
        self.train_positive_rate_: float = 0.0

    def fit(self, frame: pd.DataFrame, feature_columns: list[str], label_column: str) -> "RidgeClassifier":
        x_raw = frame.reindex(columns=feature_columns).apply(pd.to_numeric, errors="coerce")
        y = frame[label_column].astype(float).to_numpy()
        self.train_positive_rate_ = float(np.nanmean(y))

        medians = x_raw.median(axis=0).fillna(0.0).to_numpy(dtype=float)
        x = x_raw.to_numpy(dtype=float, copy=True)
        nan_mask = np.isnan(x)
        if nan_mask.any():
            x[nan_mask] = np.take(medians, np.where(nan_mask)[1])

        means = x.mean(axis=0)
        stds = x.std(axis=0)
        stds[stds == 0.0] = 1.0
        x_scaled = (x - means) / stds
        y_centered = y - self.train_positive_rate_

        identity = np.eye(x_scaled.shape[1], dtype=float)
        xtx = x_scaled.T @ x_scaled
        xty = x_scaled.T @ y_centered
        self.weights_ = np.linalg.solve(xtx + self.l2 * identity, xty)
        self.intercept_ = self.train_positive_rate_
        self.medians_ = medians
        self.means_ = means
        self.stds_ = stds
        return self

    def predict_proba(self, frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
        if self.weights_ is None or self.medians_ is None or self.means_ is None or self.stds_ is None:
            raise ValueError("RidgeClassifier has not been fit.")
        x_raw = frame.reindex(columns=feature_columns).apply(pd.to_numeric, errors="coerce").to_numpy(
            dtype=float,
            copy=True,
        )
        nan_mask = np.isnan(x_raw)
        if nan_mask.any():
            x_raw[nan_mask] = np.take(self.medians_, np.where(nan_mask)[1])
        x_scaled = (x_raw - self.means_) / self.stds_
        scores = self.intercept_ + x_scaled @ self.weights_
        return np.clip(scores, 0.001, 0.999)

    def to_artifact(self, *, label_column: str, feature_columns: list[str]) -> dict[str, Any]:
        if self.weights_ is None or self.medians_ is None or self.means_ is None or self.stds_ is None:
            raise ValueError("RidgeClassifier has not been fit.")
        return {
            "model_type": "standardized_ridge_classifier",
            "label_column": label_column,
            "feature_columns": feature_columns,
            "ridge_l2": self.l2,
            "train_positive_rate": self.train_positive_rate_,
            "intercept": self.intercept_,
            "medians": self.medians_.tolist(),
            "means": self.means_.tolist(),
            "stds": self.stds_.tolist(),
            "weights": self.weights_.tolist(),
        }


class XGBoostClassifier:
    def __init__(
        self,
        *,
        eta: float,
        max_depth: int,
        num_boost_round: int,
        subsample: float,
        colsample_bytree: float,
        min_child_weight: float,
        reg_lambda: float,
        reg_alpha: float,
        device: str,
    ) -> None:
        if xgb is None:
            raise ImportError("xgboost is not installed. Install xgboost or use --model-type ridge.")
        self.eta = float(eta)
        self.max_depth = int(max_depth)
        self.num_boost_round = int(num_boost_round)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.min_child_weight = float(min_child_weight)
        self.reg_lambda = float(reg_lambda)
        self.reg_alpha = float(reg_alpha)
        self.device = str(device)
        self.booster_: Any | None = None
        self.constant_probability_: float | None = None
        self.train_positive_rate_: float = 0.0

    def fit(self, frame: pd.DataFrame, feature_columns: list[str], label_column: str) -> "XGBoostClassifier":
        x_raw = frame.reindex(columns=feature_columns).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        y = frame[label_column].astype(float).to_numpy()
        self.train_positive_rate_ = float(np.nanmean(y))
        if np.unique(y).size < 2:
            self.constant_probability_ = float(np.clip(self.train_positive_rate_, 0.001, 0.999))
            self.booster_ = None
            return self

        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "eta": self.eta,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "lambda": self.reg_lambda,
            "alpha": self.reg_alpha,
            "tree_method": "hist",
            "device": self.device,
            "verbosity": 0,
        }
        dtrain = xgb.DMatrix(x_raw, label=y, feature_names=feature_columns, missing=np.nan)
        self.booster_ = xgb.train(
            params,
            dtrain,
            num_boost_round=self.num_boost_round,
            verbose_eval=False,
        )
        self.constant_probability_ = None
        return self

    def predict_proba(self, frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
        if self.constant_probability_ is not None:
            return np.full(len(frame), self.constant_probability_, dtype=float)
        if self.booster_ is None:
            raise ValueError("XGBoostClassifier has not been fit.")
        x_raw = frame.reindex(columns=feature_columns).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        dtest = xgb.DMatrix(x_raw, feature_names=feature_columns, missing=np.nan)
        return np.clip(self.booster_.predict(dtest), 0.001, 0.999)

    def _importance_by_feature(self, feature_columns: list[str]) -> dict[str, float]:
        if self.booster_ is None:
            return {feature: 0.0 for feature in feature_columns}
        raw_importance = self.booster_.get_score(importance_type="gain")
        return {feature: float(raw_importance.get(feature, 0.0)) for feature in feature_columns}

    def to_artifact(self, *, label_column: str, feature_columns: list[str]) -> dict[str, Any]:
        booster_json = ""
        if self.booster_ is not None:
            booster_json = bytes(self.booster_.save_raw(raw_format="json")).decode("utf-8")
        return {
            "model_type": "xgboost_classifier",
            "label_column": label_column,
            "feature_columns": feature_columns,
            "train_positive_rate": self.train_positive_rate_,
            "constant_probability": self.constant_probability_,
            "params": {
                "eta": self.eta,
                "max_depth": self.max_depth,
                "num_boost_round": self.num_boost_round,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "min_child_weight": self.min_child_weight,
                "reg_lambda": self.reg_lambda,
                "reg_alpha": self.reg_alpha,
                "device": self.device,
            },
            "feature_importance": self._importance_by_feature(feature_columns),
            "booster_json": booster_json,
        }


def build_model(config: WalkForwardConfig) -> RidgeClassifier | XGBoostClassifier:
    model_type = config.model_type.lower()
    if model_type in {"ridge", "standardized_ridge_classifier"}:
        return RidgeClassifier(l2=config.ridge_l2)
    if model_type in {"xgboost", "xgb", "xgboost_classifier"}:
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
    raise ValueError(f"Unsupported model_type={config.model_type!r}. Expected ridge or xgboost.")


def _format_multiplier(multiplier: float) -> str:
    return f"{multiplier:g}".replace(".", "_")


def true_range(price_frame: pd.DataFrame) -> pd.Series:
    high = price_frame["high"]
    low = price_frame["low"]
    close = price_frame["close"]
    prev_close = close.shift(1)
    return pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def average_true_range(price_frame: pd.DataFrame, window: int) -> pd.Series:
    return true_range(price_frame).rolling(window).mean()


def _time_to_minutes(value: str) -> int:
    parts = value.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid time value {value!r}. Expected HH:MM.")
    hour = int(parts[0])
    minute = int(parts[1])
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError(f"Invalid time value {value!r}. Expected HH:MM.")
    return hour * 60 + minute


def _intraday_minutes(index: pd.DatetimeIndex) -> np.ndarray:
    return index.hour.to_numpy(dtype=int) * 60 + index.minute.to_numpy(dtype=int)


def _rolling_percentile(series: pd.Series, *, window: int, min_periods: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.rolling(window, min_periods=min_periods).apply(
        lambda sample: float(pd.Series(sample).rank(pct=True).iloc[-1]),
        raw=False,
    )


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0.0, np.nan)


def _streak(values: pd.Series, *, positive: bool) -> pd.Series:
    signs = pd.to_numeric(values, errors="coerce").fillna(0.0)
    mask = signs.gt(0.0) if positive else signs.lt(0.0)
    groups = mask.ne(mask.shift(fill_value=False)).cumsum()
    streak = mask.groupby(groups).cumsum()
    return streak.where(mask, 0.0).astype(float)


def build_target_frame(target_prices: pd.DataFrame, config: WalkForwardConfig) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.DataFrame(index=target_prices.index)
    frame["target_close"] = target_prices["close"]
    frame["target_high"] = target_prices["high"]
    frame["target_low"] = target_prices["low"]
    frame["atr_points"] = average_true_range(target_prices, config.atr_window)
    frame["next_open"] = target_prices["open"].shift(-1)
    frame["next_high"] = target_prices["high"].shift(-1)
    frame["next_low"] = target_prices["low"].shift(-1)
    frame["next_close"] = target_prices["close"].shift(-1)
    frame["next_close_return_points"] = frame["next_close"] - frame["target_close"]
    frame["next_close_return_atr"] = frame["next_close_return_points"] / frame["atr_points"].replace(0.0, np.nan)
    frame["next_open_to_close_return_points"] = frame["next_close"] - frame["next_open"]
    frame["next_open_to_close_return_atr"] = (
        frame["next_open_to_close_return_points"] / frame["atr_points"].replace(0.0, np.nan)
    )
    frame["next_high_move_atr"] = (frame["next_high"] - frame["target_close"]) / frame["atr_points"].replace(
        0.0,
        np.nan,
    )
    frame["next_low_move_atr"] = (frame["target_close"] - frame["next_low"]) / frame["atr_points"].replace(
        0.0,
        np.nan,
    )

    target_mode = config.target_mode.lower()
    label_columns: list[str] = []
    if target_mode in {"open_close", "combined"}:
        valid_open_close = frame["next_open"].notna() & frame["next_close"].notna()
        frame["target_up_close"] = np.where(
            valid_open_close,
            frame["next_open_to_close_return_points"] > 0.0,
            np.nan,
        )
        frame["target_down_close"] = np.where(
            valid_open_close,
            frame["next_open_to_close_return_points"] < 0.0,
            np.nan,
        )
        label_columns.extend(["target_up_close", "target_down_close"])

    if target_mode in {"open_close_atr", "combined"}:
        valid_open_close_atr = (
            frame["atr_points"].notna()
            & frame["next_open"].notna()
            & frame["next_close"].notna()
        )
        for multiplier in config.atr_multipliers:
            suffix = _format_multiplier(multiplier)
            up_column = f"target_up_oc_{suffix}atr"
            down_column = f"target_down_oc_{suffix}atr"
            frame[up_column] = np.where(
                valid_open_close_atr,
                frame["next_open_to_close_return_atr"] >= multiplier,
                np.nan,
            )
            frame[down_column] = np.where(
                valid_open_close_atr,
                frame["next_open_to_close_return_atr"] <= -multiplier,
                np.nan,
            )
            label_columns.extend([up_column, down_column])

    if target_mode in {"atr_touch", "combined"}:
        valid_atr_touch = frame["atr_points"].notna() & frame["next_high"].notna() & frame["next_low"].notna()
        for multiplier in config.atr_multipliers:
            suffix = _format_multiplier(multiplier)
            up_column = f"target_up_{suffix}atr"
            down_column = f"target_down_{suffix}atr"
            frame[up_column] = np.where(valid_atr_touch, frame["next_high_move_atr"] >= multiplier, np.nan)
            frame[down_column] = np.where(valid_atr_touch, frame["next_low_move_atr"] >= multiplier, np.nan)
            label_columns.extend([up_column, down_column])

    if not label_columns:
        raise ValueError("target_mode must be one of: atr_touch, open_close, open_close_atr, combined")

    return frame, label_columns


def build_intraday_60m_feature_frame(
    price_frame: pd.DataFrame,
    *,
    symbol: str,
    target_index: pd.DatetimeIndex,
    config: WalkForwardConfig,
) -> pd.DataFrame:
    start_minutes = _time_to_minutes(config.intraday_60m_start_time)
    end_minutes = _time_to_minutes(config.intraday_60m_end_time)
    if end_minutes < start_minutes:
        raise ValueError("--intraday-rth-end cannot be earlier than --intraday-rth-start")

    sorted_frame = price_frame.sort_index()
    minutes = _intraday_minutes(sorted_frame.index)
    rth_frame = sorted_frame.loc[(minutes >= start_minutes) & (minutes <= end_minutes)].copy()
    feature_names = [
        *(f"{symbol}__rth60__{column}" for column in TES_INDICATOR_COLUMNS),
        f"{symbol}__rth60__bar_count",
        f"{symbol}__rth60__has_end_bar",
        f"{symbol}__rth60__return",
        f"{symbol}__rth60__range_pct",
        f"{symbol}__rth60__close_position",
        f"{symbol}__rth60__first_half_return",
        f"{symbol}__rth60__second_half_return",
        f"{symbol}__rth60__last_2h_return",
        f"{symbol}__rth60__last_hour_return",
        f"{symbol}__rth60__late_reversal",
        f"{symbol}__rth60__up_bar_ratio",
        f"{symbol}__rth60__runup_pct",
        f"{symbol}__rth60__drawdown_pct",
        f"{symbol}__rth60__trend_efficiency",
    ]
    if rth_frame.empty:
        return pd.DataFrame(0.0, index=target_index, columns=feature_names)

    feature_dates = pd.to_datetime(pd.Index(rth_frame.index.date), errors="coerce")
    indicators = build_indicator_frame(rth_frame)
    indicators["feature_date"] = feature_dates.to_numpy()
    daily = indicators.groupby("feature_date", sort=True).tail(1).set_index("feature_date")

    grouped = rth_frame.assign(feature_date=feature_dates.to_numpy()).groupby("feature_date", sort=True)
    session_high = grouped["high"].max()
    session_low = grouped["low"].min()
    first_open = grouped["open"].first()
    last_close = grouped["close"].last()
    session_range = (session_high - session_low).replace(0.0, np.nan)

    stats = pd.DataFrame(index=daily.index)
    stats["bar_count"] = grouped.size().reindex(daily.index).astype(float)
    stats["has_end_bar"] = grouped.apply(
        lambda frame: float((_intraday_minutes(frame.index) >= end_minutes).any()),
    ).reindex(daily.index).astype(float)
    stats["return"] = ((last_close / first_open) - 1.0).reindex(daily.index)
    stats["range_pct"] = ((session_high - session_low) / first_open.replace(0.0, np.nan)).reindex(daily.index)
    stats["close_position"] = (((last_close - session_low) / session_range) * 2.0 - 1.0).reindex(daily.index)
    midpoint_minutes = start_minutes + ((end_minutes - start_minutes) // 2)

    def _window_return(frame: pd.DataFrame, *, start: int | None = None, end: int | None = None) -> float:
        frame_minutes = _intraday_minutes(frame.index)
        mask = pd.Series(True, index=frame.index)
        if start is not None:
            mask &= frame_minutes >= start
        if end is not None:
            mask &= frame_minutes <= end
        window = frame.loc[mask.to_numpy()]
        if window.empty:
            return 0.0
        first = float(window["open"].iloc[0])
        if first == 0.0:
            return 0.0
        return float((window["close"].iloc[-1] / first) - 1.0)

    first_half_return = grouped.apply(lambda frame: _window_return(frame, end=midpoint_minutes))
    second_half_return = grouped.apply(lambda frame: _window_return(frame, start=midpoint_minutes + 1))
    last_2h_return = grouped.apply(lambda frame: _window_return(frame, start=max(start_minutes, end_minutes - 120)))
    last_hour_return = grouped.apply(lambda frame: _window_return(frame, start=max(start_minutes, end_minutes - 60)))
    up_bar_ratio = grouped.apply(lambda frame: float((frame["close"] > frame["open"]).mean()))

    stats["first_half_return"] = first_half_return.reindex(daily.index)
    stats["second_half_return"] = second_half_return.reindex(daily.index)
    stats["last_2h_return"] = last_2h_return.reindex(daily.index)
    stats["last_hour_return"] = last_hour_return.reindex(daily.index)
    stats["late_reversal"] = (stats["last_2h_return"] - stats["first_half_return"]).reindex(daily.index)
    stats["up_bar_ratio"] = up_bar_ratio.reindex(daily.index)
    stats["runup_pct"] = ((session_high - first_open) / first_open.replace(0.0, np.nan)).reindex(daily.index)
    stats["drawdown_pct"] = ((first_open - session_low) / first_open.replace(0.0, np.nan)).reindex(daily.index)
    stats["trend_efficiency"] = (
        (last_close - first_open).abs() / session_range
    ).reindex(daily.index)

    daily = pd.concat([daily.reindex(columns=TES_INDICATOR_COLUMNS), stats], axis=1)
    renamed = daily.rename(
        columns={column: f"{symbol}__rth60__{column}" for column in daily.columns},
    )
    return renamed.reindex(target_index).reindex(columns=feature_names).fillna(0.0)


def build_market_structure_feature_frame(
    price_frame: pd.DataFrame,
    *,
    symbol: str,
    target_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    frame = price_frame.sort_index().copy()
    open_ = pd.to_numeric(frame["open"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    prev_close = close.shift(1)
    daily_range = (high - low).replace(0.0, np.nan)

    features = pd.DataFrame(index=frame.index)
    features["gap_from_prior_close"] = _safe_ratio(open_ - prev_close, prev_close)
    features["close_to_close_return"] = close.pct_change()
    features["open_to_close_return"] = _safe_ratio(close - open_, open_)
    features["high_low_range_pct"] = _safe_ratio(high - low, open_)
    features["close_position"] = ((close - low) / daily_range) * 2.0 - 1.0
    features["gap_filled_prior_close"] = ((low <= prev_close) & (high >= prev_close)).astype(float)
    features["inside_day"] = ((high < high.shift(1)) & (low > low.shift(1))).astype(float)
    features["outside_day"] = ((high > high.shift(1)) & (low < low.shift(1))).astype(float)
    features["up_close_streak"] = _streak(close.diff(), positive=True)
    features["down_close_streak"] = _streak(close.diff(), positive=False)

    for window in (20, 50, 100, 200):
        rolling_high = high.shift(1).rolling(window).max()
        rolling_low = low.shift(1).rolling(window).min()
        rolling_mid = (rolling_high + rolling_low) / 2.0
        rolling_range = (rolling_high - rolling_low).replace(0.0, np.nan)
        features[f"distance_from_{window}d_high"] = _safe_ratio(close - rolling_high, rolling_high)
        features[f"distance_from_{window}d_low"] = _safe_ratio(close - rolling_low, rolling_low)
        features[f"{window}d_channel_position"] = ((close - rolling_low) / rolling_range) * 2.0 - 1.0
        features[f"drawdown_from_{window}d_high"] = _safe_ratio(close - rolling_high, rolling_high)
        features[f"distance_from_{window}d_mid"] = _safe_ratio(close - rolling_mid, rolling_mid)

    renamed = features.rename(columns={column: f"{symbol}__structure__{column}" for column in features.columns})
    return renamed.reindex(target_index).fillna(0.0)


def build_cross_market_feature_frame(
    frames: dict[str, pd.DataFrame],
    intraday_features: dict[str, pd.DataFrame],
    *,
    target_symbol: str,
    target_index: pd.DatetimeIndex,
    context_ffill_limit: int,
) -> pd.DataFrame:
    target_symbol = target_symbol.upper()
    target_close = pd.to_numeric(frames[target_symbol]["close"], errors="coerce").reindex(target_index)
    target_return_1d = target_close.pct_change()
    target_return_5d = target_close.pct_change(5)
    target_range_pct = _safe_ratio(
        pd.to_numeric(frames[target_symbol]["high"], errors="coerce").reindex(target_index)
        - pd.to_numeric(frames[target_symbol]["low"], errors="coerce").reindex(target_index),
        pd.to_numeric(frames[target_symbol]["open"], errors="coerce").reindex(target_index),
    )

    features = pd.DataFrame(index=target_index)
    for symbol, price_frame in frames.items():
        symbol = symbol.upper()
        if symbol == target_symbol:
            continue
        close = pd.to_numeric(price_frame["close"], errors="coerce").reindex(target_index).ffill(
            limit=context_ffill_limit,
        )
        open_ = pd.to_numeric(price_frame["open"], errors="coerce").reindex(target_index).ffill(
            limit=context_ffill_limit,
        )
        high = pd.to_numeric(price_frame["high"], errors="coerce").reindex(target_index).ffill(
            limit=context_ffill_limit,
        )
        low = pd.to_numeric(price_frame["low"], errors="coerce").reindex(target_index).ffill(
            limit=context_ffill_limit,
        )
        return_1d = close.pct_change()
        return_5d = close.pct_change(5)
        range_pct = _safe_ratio(high - low, open_)
        prefix = f"cross__{target_symbol}_{symbol}"
        features[f"{prefix}__return_1d_spread"] = target_return_1d - return_1d
        features[f"{prefix}__return_5d_spread"] = target_return_5d - return_5d
        features[f"{prefix}__return_1d_agreement"] = (np.sign(target_return_1d) == np.sign(return_1d)).astype(float)
        features[f"{prefix}__range_pct_ratio"] = _safe_ratio(target_range_pct, range_pct)

    for symbol, symbol_features in intraday_features.items():
        symbol = symbol.upper()
        if symbol == target_symbol:
            continue
        target_rth_return = intraday_features.get(target_symbol, pd.DataFrame(index=target_index)).get(
            f"{target_symbol}__rth60__return",
        )
        target_rth_range = intraday_features.get(target_symbol, pd.DataFrame(index=target_index)).get(
            f"{target_symbol}__rth60__range_pct",
        )
        context_rth_return = symbol_features.get(f"{symbol}__rth60__return")
        context_rth_range = symbol_features.get(f"{symbol}__rth60__range_pct")
        if target_rth_return is None or context_rth_return is None:
            continue
        prefix = f"cross__{target_symbol}_{symbol}"
        features[f"{prefix}__rth60_return_spread"] = target_rth_return - context_rth_return
        features[f"{prefix}__rth60_return_agreement"] = (
            np.sign(target_rth_return) == np.sign(context_rth_return)
        ).astype(float)
        if target_rth_range is not None and context_rth_range is not None:
            features[f"{prefix}__rth60_range_ratio"] = _safe_ratio(target_rth_range, context_rth_range)

    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_regime_feature_frame(frame: pd.DataFrame, *, symbols: tuple[str, ...]) -> pd.DataFrame:
    features = pd.DataFrame(index=frame.index)
    close = pd.to_numeric(frame["target_close"], errors="coerce")
    high = pd.to_numeric(frame["target_high"], errors="coerce")
    low = pd.to_numeric(frame["target_low"], errors="coerce")
    daily_range_pct = _safe_ratio(high - low, close)
    abs_return = close.pct_change().abs()
    atr_points = pd.to_numeric(frame["atr_points"], errors="coerce")

    for window, min_periods in ((250, 100), (750, 250)):
        features[f"regime__atr_percentile_{window}"] = _rolling_percentile(
            atr_points,
            window=window,
            min_periods=min_periods,
        )
        features[f"regime__daily_range_percentile_{window}"] = _rolling_percentile(
            daily_range_pct,
            window=window,
            min_periods=min_periods,
        )
        features[f"regime__abs_return_percentile_{window}"] = _rolling_percentile(
            abs_return,
            window=window,
            min_periods=min_periods,
        )
        es_rth_range = frame.get("ES__rth60__range_pct")
        es_rth_return = frame.get("ES__rth60__return")
        nq_rth_return = frame.get("NQ__rth60__return")
        if es_rth_range is not None:
            features[f"regime__es_rth_range_percentile_{window}"] = _rolling_percentile(
                pd.to_numeric(es_rth_range, errors="coerce"),
                window=window,
                min_periods=min_periods,
            )
        if es_rth_return is not None:
            features[f"regime__es_rth_abs_return_percentile_{window}"] = _rolling_percentile(
                pd.to_numeric(es_rth_return, errors="coerce").abs(),
                window=window,
                min_periods=min_periods,
            )
        if nq_rth_return is not None:
            features[f"regime__nq_rth_abs_return_percentile_{window}"] = _rolling_percentile(
                pd.to_numeric(nq_rth_return, errors="coerce").abs(),
                window=window,
                min_periods=min_periods,
            )

        range_percentiles = []
        for symbol in symbols:
            rth_range = frame.get(f"{symbol}__rth60__range_pct")
            if rth_range is None:
                continue
            range_percentiles.append(
                _rolling_percentile(
                    pd.to_numeric(rth_range, errors="coerce"),
                    window=window,
                    min_periods=min_periods,
                ),
            )
        if range_percentiles:
            features[f"regime__cross_asset_rth_range_percentile_mean_{window}"] = pd.concat(
                range_percentiles,
                axis=1,
            ).mean(axis=1)

    score_columns = [
        "regime__atr_percentile_750",
        "regime__es_rth_range_percentile_750",
        "regime__es_rth_abs_return_percentile_750",
        "regime__nq_rth_abs_return_percentile_750",
        "regime__cross_asset_rth_range_percentile_mean_750",
    ]
    gates = {
        "regime__atr_percentile_750": 0.90,
        "regime__es_rth_range_percentile_750": 0.90,
        "regime__es_rth_abs_return_percentile_750": 0.90,
        "regime__nq_rth_abs_return_percentile_750": 0.90,
        "regime__cross_asset_rth_range_percentile_mean_750": 0.75,
    }
    stress_score = pd.Series(0.0, index=frame.index)
    for column in score_columns:
        if column in features.columns:
            stress_score += features[column].ge(gates[column]).astype(float)
    features["regime__stress_score"] = stress_score
    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_atr_distribution_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=frame.index)
    atr_points = pd.to_numeric(frame["atr_points"], errors="coerce")
    close = pd.to_numeric(frame["target_close"], errors="coerce")

    atr_mean_5 = atr_points.rolling(5, min_periods=5).mean()
    atr_mean_20 = atr_points.rolling(20, min_periods=20).mean()
    atr_mean_100 = atr_points.rolling(100, min_periods=100).mean()
    atr_std_20 = atr_points.rolling(20, min_periods=20).std()
    atr_median_20 = atr_points.rolling(20, min_periods=20).median()

    features["atr_pctile_20_250"] = _rolling_percentile(
        atr_points,
        window=250,
        min_periods=20,
    )
    features["atr_z_20"] = _safe_ratio(atr_points - atr_mean_20, atr_std_20)
    features["atr_expansion_5_20"] = _safe_ratio(atr_mean_5, atr_mean_20)
    features["atr_expansion_20_100"] = _safe_ratio(atr_mean_20, atr_mean_100)
    features["atr_shock"] = _safe_ratio(atr_points, atr_median_20)
    features["atr_pct_close"] = _safe_ratio(atr_points, close)
    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_dataset(config: WalkForwardConfig) -> DatasetBundle:
    symbols = config.normalized_symbols()
    frames, source_files = load_symbol_frames(
        symbols,
        data_roots=config.data_roots,
        file_map=config.data_file_map,
    )
    target_symbol = config.target_symbol.upper()
    target_prices = frames[target_symbol]
    target_frame, label_columns = build_target_frame(target_prices, config)
    target_index = target_frame.index

    feature_frames: list[pd.DataFrame] = []
    feature_columns: list[str] = []
    intraday_features_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        if symbol == target_symbol and not config.include_target_features:
            continue
        indicators = build_indicator_frame(frames[symbol])
        aligned = indicators.reindex(target_index)
        if symbol != target_symbol:
            aligned = aligned.ffill(limit=config.context_ffill_limit)
        aligned = aligned.fillna(0.0)
        renamed = aligned.rename(columns={column: f"{symbol}__{column}" for column in TES_INDICATOR_COLUMNS})
        feature_frames.append(renamed)
        feature_columns.extend(renamed.columns.tolist())
        if config.include_enhanced_features:
            structure = build_market_structure_feature_frame(
                frames[symbol],
                symbol=symbol,
                target_index=target_index,
            )
            if symbol != target_symbol:
                structure = structure.ffill(limit=config.context_ffill_limit)
            structure = structure.fillna(0.0)
            feature_frames.append(structure)
            feature_columns.extend(structure.columns.tolist())

    if config.include_intraday_60m:
        if config.allow_missing_intraday_symbols:
            intraday_frames: dict[str, pd.DataFrame] = {}
            intraday_sources: dict[str, str] = {}
            for symbol in symbols:
                try:
                    symbol_frames, symbol_sources = load_intraday_symbol_frames(
                        (symbol,),
                        data_roots=config.intraday_60m_roots,
                        file_map=config.intraday_60m_file_map,
                    )
                except FileNotFoundError:
                    source_files[f"{symbol}__60m"] = "missing_skipped"
                    continue
                intraday_frames.update(symbol_frames)
                intraday_sources.update(symbol_sources)
        else:
            intraday_frames, intraday_sources = load_intraday_symbol_frames(
                symbols,
                data_roots=config.intraday_60m_roots,
                file_map=config.intraday_60m_file_map,
            )
            for symbol, source in intraday_sources.items():
                source_files[f"{symbol}__60m"] = source
        for symbol, source in intraday_sources.items():
            source_files[f"{symbol}__60m"] = source
        for symbol in symbols:
            if symbol == target_symbol and not config.include_target_features:
                continue
            if symbol not in intraday_frames:
                continue
            intraday_features = build_intraday_60m_feature_frame(
                intraday_frames[symbol],
                symbol=symbol,
                target_index=target_index,
                config=config,
            )
            intraday_features_by_symbol[symbol] = intraday_features
            feature_frames.append(intraday_features)
            feature_columns.extend(intraday_features.columns.tolist())
    if config.include_enhanced_features:
        cross_features = build_cross_market_feature_frame(
            frames,
            intraday_features_by_symbol,
            target_symbol=target_symbol,
            target_index=target_index,
            context_ffill_limit=config.context_ffill_limit,
        )
        feature_frames.append(cross_features)
        feature_columns.extend(cross_features.columns.tolist())

    frame = pd.concat([target_frame, *feature_frames], axis=1).sort_index().copy()
    if config.include_atr_distribution_features or config.include_enhanced_features:
        atr_distribution_features = build_atr_distribution_feature_frame(frame)
        frame = pd.concat([frame, atr_distribution_features], axis=1)
        feature_columns.extend(atr_distribution_features.columns.tolist())
    if config.include_enhanced_features:
        regime_features = build_regime_feature_frame(frame, symbols=symbols)
        frame = pd.concat([frame, regime_features], axis=1)
        feature_columns.extend(regime_features.columns.tolist())
    if config.start_date:
        frame = frame[frame.index >= pd.Timestamp(config.start_date)]
    if config.end_date:
        frame = frame[frame.index <= pd.Timestamp(config.end_date)]
    frame = frame.reset_index().rename(columns={"date": "signal_date"})
    frame["signal_date"] = pd.to_datetime(frame["signal_date"]).dt.date.astype(str)

    probability_columns = {
        label_column: f"prob_{label_column.removeprefix('target_')}"
        for label_column in label_columns
    }
    return DatasetBundle(
        frame=frame,
        feature_columns=feature_columns,
        label_columns=label_columns,
        probability_columns=probability_columns,
        source_files=source_files,
    )


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    y = np.asarray(y_true, dtype=float)
    scores = np.asarray(y_score, dtype=float)
    mask = ~np.isnan(y) & ~np.isnan(scores)
    y = y[mask]
    scores = scores[mask]
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    rank_sum = float(ranks[y == 1.0].sum())
    auc = (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=float)
    prob = np.asarray(y_prob, dtype=float)
    mask = ~np.isnan(y) & ~np.isnan(prob)
    y = y[mask]
    prob = np.clip(prob[mask], 0.001, 0.999)
    if len(y) == 0:
        return {
            "rows": 0,
            "positive_rate": None,
            "auc": None,
            "brier": None,
            "log_loss": None,
            "accuracy_50": None,
            "top_20pct_hit_rate": None,
        }
    top_n = max(1, int(np.ceil(len(y) * 0.20)))
    order = np.argsort(-prob)
    top_hit_rate = float(y[order[:top_n]].mean())
    predictions = (prob >= 0.5).astype(float)
    return {
        "rows": int(len(y)),
        "positive_rate": float(y.mean()),
        "auc": roc_auc(y, prob),
        "brier": float(np.mean((prob - y) ** 2)),
        "log_loss": float(-(y * np.log(prob) + (1.0 - y) * np.log(1.0 - prob)).mean()),
        "accuracy_50": float((predictions == y).mean()),
        "top_20pct_hit_rate": top_hit_rate,
    }


def _labeled_frame(dataset: DatasetBundle) -> pd.DataFrame:
    return dataset.frame.dropna(subset=dataset.label_columns).reset_index(drop=True)


def run_walk_forward(config: WalkForwardConfig, dataset: DatasetBundle) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled = _labeled_frame(dataset)
    predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    fold_number = 0

    for test_start in range(config.min_train_rows, len(labeled), config.step_rows):
        if config.max_folds and fold_number >= config.max_folds:
            break
        train_end = test_start
        train_start = 0
        if config.rolling_train_rows > 0:
            train_start = max(0, train_end - config.rolling_train_rows)
        if train_end - train_start < config.min_train_rows:
            continue
        test_end = min(test_start + config.test_rows, len(labeled))
        if test_end <= test_start:
            continue

        train_frame = labeled.iloc[train_start:train_end].copy()
        test_frame = labeled.iloc[test_start:test_end].copy()
        fold_pred = test_frame[
            [
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
                *dataset.label_columns,
            ]
        ].copy()
        fold_number += 1

        for label_column in dataset.label_columns:
            probability_column = dataset.probability_columns[label_column]
            model = build_model(config).fit(train_frame, dataset.feature_columns, label_column)
            fold_pred[probability_column] = model.predict_proba(test_frame, dataset.feature_columns)
            metrics = binary_metrics(
                test_frame[label_column].to_numpy(dtype=float),
                fold_pred[probability_column].to_numpy(dtype=float),
            )
            fold_rows.append(
                {
                    "fold": fold_number,
                    "label_column": label_column,
                    "probability_column": probability_column,
                    "train_start_date": train_frame["signal_date"].iloc[0],
                    "train_end_date": train_frame["signal_date"].iloc[-1],
                    "test_start_date": test_frame["signal_date"].iloc[0],
                    "test_end_date": test_frame["signal_date"].iloc[-1],
                    "train_rows": int(len(train_frame)),
                    "test_rows": int(len(test_frame)),
                    **metrics,
                },
            )
        predictions.append(fold_pred)

    if not predictions:
        raise RuntimeError(
            "Walk-forward produced no folds. Reduce --min-train-rows or verify there is enough labeled history.",
        )
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(fold_rows)


def fit_final_models(config: WalkForwardConfig, dataset: DatasetBundle) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    labeled = _labeled_frame(dataset)
    if labeled.empty:
        raise RuntimeError("No labeled rows are available for the final fit.")

    models: dict[str, Any] = {}
    feature_importance_rows: list[dict[str, Any]] = []
    final_predictions = labeled[
        [
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
            *dataset.label_columns,
        ]
    ].copy()

    latest_frame = dataset.frame.tail(1).copy()
    latest_signal = latest_frame[["signal_date", "target_close", "atr_points"]].copy()

    for label_column in dataset.label_columns:
        probability_column = dataset.probability_columns[label_column]
        model = build_model(config).fit(labeled, dataset.feature_columns, label_column)
        final_predictions[probability_column] = model.predict_proba(labeled, dataset.feature_columns)
        latest_signal[probability_column] = model.predict_proba(latest_frame, dataset.feature_columns)
        models[label_column] = model.to_artifact(
            label_column=label_column,
            feature_columns=dataset.feature_columns,
        )
        if models[label_column]["model_type"] == "xgboost_classifier":
            importance_map = models[label_column]["feature_importance"]
        else:
            importance_map = dict(
                zip(
                    dataset.feature_columns,
                    np.array(models[label_column]["weights"], dtype=float),
                    strict=True,
                ),
            )
        for feature_name in dataset.feature_columns:
            importance = float(importance_map.get(feature_name, 0.0))
            feature_importance_rows.append(
                {
                    "label_column": label_column,
                    "feature": feature_name,
                    "importance": importance,
                    "abs_importance": float(abs(importance)),
                },
            )

    artifact = {
        "config": config.to_json_dict(),
        "source_files": dataset.source_files,
        "feature_columns": dataset.feature_columns,
        "label_columns": dataset.label_columns,
        "probability_columns": dataset.probability_columns,
        "models": models,
    }
    feature_importance = pd.DataFrame(feature_importance_rows).sort_values(
        ["label_column", "abs_importance"],
        ascending=[True, False],
    )
    return artifact, latest_signal, final_predictions, feature_importance


def summarize_metrics(
    config: WalkForwardConfig,
    dataset: DatasetBundle,
    predictions: pd.DataFrame,
    fold_summary: pd.DataFrame,
    final_predictions: pd.DataFrame,
) -> dict[str, Any]:
    walk_forward: dict[str, Any] = {}
    final_fit: dict[str, Any] = {}
    for label_column in dataset.label_columns:
        probability_column = dataset.probability_columns[label_column]
        walk_forward[label_column] = binary_metrics(
            predictions[label_column].to_numpy(dtype=float),
            predictions[probability_column].to_numpy(dtype=float),
        )
        final_fit[label_column] = binary_metrics(
            final_predictions[label_column].to_numpy(dtype=float),
            final_predictions[probability_column].to_numpy(dtype=float),
        )
    return {
        "run_name": config.run_name,
        "target_symbol": config.target_symbol.upper(),
        "target_mode": config.target_mode,
        "model_type": config.model_type,
        "symbols": list(config.normalized_symbols()),
        "row_count": int(len(dataset.frame)),
        "labeled_row_count": int(len(_labeled_frame(dataset))),
        "feature_count": int(len(dataset.feature_columns)),
        "label_columns": dataset.label_columns,
        "source_files": dataset.source_files,
        "walk_forward_fold_count": int(fold_summary["fold"].nunique()) if not fold_summary.empty else 0,
        "walk_forward": walk_forward,
        "final_fit": final_fit,
    }


def train_pipeline(config: WalkForwardConfig) -> dict[str, Any]:
    output_root = config.output_root()
    output_root.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(config)
    predictions, fold_summary = run_walk_forward(config, dataset)
    model_artifact, latest_signal, final_predictions, feature_importance = fit_final_models(config, dataset)
    metrics = summarize_metrics(config, dataset, predictions, fold_summary, final_predictions)

    paths = {
        "training_dataset": output_root / "training_dataset.csv",
        "walk_forward_predictions": output_root / "walk_forward_predictions.csv",
        "walk_forward_folds": output_root / "walk_forward_folds.csv",
        "final_fit_predictions": output_root / "final_fit_predictions.csv",
        "latest_signal": output_root / "latest_signal.csv",
        "feature_importance": output_root / "feature_importance.csv",
        "model_artifact": output_root / "model_artifact.json",
        "metrics": output_root / "metrics.json",
    }
    dataset.frame.to_csv(paths["training_dataset"], index=False)
    predictions.to_csv(paths["walk_forward_predictions"], index=False)
    fold_summary.to_csv(paths["walk_forward_folds"], index=False)
    final_predictions.to_csv(paths["final_fit_predictions"], index=False)
    latest_signal.to_csv(paths["latest_signal"], index=False)
    feature_importance.to_csv(paths["feature_importance"], index=False)
    paths["model_artifact"].write_text(json.dumps(model_artifact, indent=2))
    paths["metrics"].write_text(json.dumps(metrics, indent=2))

    metrics["output_paths"] = {key: str(path) for key, path in paths.items()}
    return metrics
