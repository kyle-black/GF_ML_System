from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_PRICE_COLUMNS = ("open", "high", "low", "close")

TES_INDICATOR_COLUMNS = (
    "breakout",
    "ma_slope",
    "macd",
    "momentum",
    "adx",
    "adx_slope",
    "short_trend",
    "medium_trend",
    "micro_trend",
    "momentum_10",
    "momentum_20",
    "momentum_60",
    "momentum_100",
    "breakout_10",
    "breakout_20",
    "breakout_50",
    "breakout_100",
    "ma_cross_5_10",
    "ma_cross_10_20",
    "ma_cross_20_50",
    "ma_cross_50_100",
    "bollinger_pct_10",
    "bollinger_pct_20",
    "bollinger_pct_50",
    "bollinger_pct_100",
    "keltner_pct_10",
    "keltner_pct_20",
    "keltner_pct_50",
    "keltner_pct_100",
    "range_expansion_10",
    "range_expansion_20",
    "range_expansion_50",
    "range_expansion_100",
    "range_expansion_short",
    "range_expansion_medium",
    "range_expansion_long",
    "price_vs_ma_10",
    "price_vs_ma_20",
    "price_vs_ma_50",
    "price_vs_ma_100",
    "ma_slope_10",
    "ma_slope_20",
    "ma_slope_50",
    "ma_slope_100",
    "regression_10",
    "regression_short",
    "regression_medium",
    "regression_long",
    "adx_10",
    "adx_slope_10",
)

TES_BREAKOUT_RANGE_ONLY_COLUMNS = (
    "breakout_20",
    "breakout_50",
    "breakout_100",
    "range_expansion_short",
    "range_expansion_medium",
    "range_expansion_long",
)


def validate_price_frame(price_frame: pd.DataFrame) -> pd.DataFrame:
    frame = price_frame.copy()
    frame.columns = [str(column).lower() for column in frame.columns]
    missing = [column for column in REQUIRED_PRICE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"price_frame missing required columns: {missing}")
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date"]).sort_values("date").set_index("date", drop=True)
    return frame.sort_index()


def breakout_signal(close: pd.Series, lookback: int = 20) -> pd.Series:
    rolling_high = close.shift(1).rolling(lookback).max()
    rolling_low = close.shift(1).rolling(lookback).min()
    channel = (rolling_high - rolling_low).replace(0.0, np.nan)
    normalized = ((close - rolling_low) / channel) * 2.0 - 1.0
    return normalized.clip(-1.0, 1.0).fillna(0.0)


def moving_average_slope_signal(close: pd.Series, ma_window: int = 20, slope_window: int = 5) -> pd.Series:
    ma = close.rolling(ma_window).mean()
    slope = ma.diff(slope_window)
    scale = close.rolling(ma_window).std().replace(0.0, np.nan)
    return np.tanh((slope / scale).replace([np.inf, -np.inf], np.nan)).fillna(0.0)


def macd_signal(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    fast_ema = close.ewm(span=fast, adjust=False).mean()
    slow_ema = close.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    rolling_vol = close.pct_change().rolling(slow).std().replace(0.0, np.nan)
    scaled = histogram / (close * rolling_vol)
    return np.tanh((scaled * 3.0).replace([np.inf, -np.inf], np.nan)).fillna(0.0)


def momentum_signal(close: pd.Series, lookback: int = 20) -> pd.Series:
    momentum = close.pct_change(lookback)
    volatility = close.pct_change().rolling(lookback).std().replace(0.0, np.nan)
    standardized = momentum / volatility
    return np.tanh(standardized.replace([np.inf, -np.inf], np.nan)).fillna(0.0)


def trend_signal(close: pd.Series, short_window: int, long_window: int) -> pd.Series:
    short_ma = close.rolling(short_window).mean()
    long_ma = close.rolling(long_window).mean()
    spread = short_ma - long_ma
    scale = close.rolling(long_window).std().replace(0.0, np.nan)
    return np.tanh((spread / scale).replace([np.inf, -np.inf], np.nan)).fillna(0.0)


def bollinger_percent_signal(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    mean = close.rolling(window).mean()
    std = close.rolling(window).std().replace(0.0, np.nan)
    upper = mean + n_std * std
    lower = mean - n_std * std
    width = (upper - lower).replace(0.0, np.nan)
    normalized = ((close - lower) / width) * 2.0 - 1.0
    return normalized.clip(-1.0, 1.0).fillna(0.0)


def keltner_percent_signal(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
    atr_mult: float = 1.5,
) -> pd.Series:
    center = close.ewm(span=window, adjust=False).mean()
    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(window).mean().replace(0.0, np.nan)
    upper = center + atr_mult * atr
    lower = center - atr_mult * atr
    width = (upper - lower).replace(0.0, np.nan)
    normalized = ((close - lower) / width) * 2.0 - 1.0
    return normalized.clip(-1.0, 1.0).fillna(0.0)


def adx_signal(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0),
        index=high.index,
    )

    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean() / atr
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean() / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = dx.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean() / 100.0
    return adx.clip(0.0, 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def adx_slope_signal(adx: pd.Series, slope_window: int = 5) -> pd.Series:
    slope = adx.diff(slope_window)
    return np.tanh((slope * 5.0).replace([np.inf, -np.inf], np.nan)).fillna(0.0)


def range_expansion_signal(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    daily_range = (high - low).abs()
    avg_range = daily_range.shift(1).rolling(window).mean().replace(0.0, np.nan)
    ratio = daily_range / avg_range
    return np.tanh(((ratio - 1.0) * 2.0).replace([np.inf, -np.inf], np.nan)).fillna(0.0)


def price_vs_ma_signal(close: pd.Series, window: int = 20) -> pd.Series:
    ma = close.rolling(window).mean()
    scale = close.rolling(window).std().replace(0.0, np.nan)
    return np.tanh(((close - ma) / scale).replace([np.inf, -np.inf], np.nan)).fillna(0.0)


def regression_signal(close: pd.Series, window: int = 20) -> pd.Series:
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    denominator = float(np.dot(x_centered, x_centered))

    def slope(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        y_centered = values - values.mean()
        return float(np.dot(x_centered, y_centered) / denominator)

    raw_slope = close.rolling(window).apply(slope, raw=True)
    scale = close.rolling(window).std().replace(0.0, np.nan)
    standardized = raw_slope * window / scale
    return np.tanh(standardized.replace([np.inf, -np.inf], np.nan)).fillna(0.0)


def build_indicator_frame(price_frame: pd.DataFrame) -> pd.DataFrame:
    frame = validate_price_frame(price_frame)
    high = frame["high"]
    low = frame["low"]
    close = frame["close"]

    indicators = pd.DataFrame(index=frame.index)
    indicators["breakout"] = breakout_signal(close, lookback=20)
    indicators["ma_slope"] = moving_average_slope_signal(close, ma_window=20, slope_window=5)
    indicators["macd"] = macd_signal(close, fast=12, slow=26, signal=9)
    indicators["momentum"] = momentum_signal(close, lookback=20)
    indicators["adx"] = adx_signal(high, low, close, window=14)
    indicators["adx_slope"] = adx_slope_signal(indicators["adx"], slope_window=5)
    indicators["short_trend"] = trend_signal(close, short_window=10, long_window=30)
    indicators["medium_trend"] = trend_signal(close, short_window=50, long_window=150)
    indicators["micro_trend"] = trend_signal(close, short_window=5, long_window=10)

    indicators["momentum_10"] = momentum_signal(close, lookback=10)
    indicators["momentum_20"] = momentum_signal(close, lookback=20)
    indicators["momentum_60"] = momentum_signal(close, lookback=60)
    indicators["momentum_100"] = momentum_signal(close, lookback=100)

    indicators["breakout_10"] = breakout_signal(close, lookback=10)
    indicators["breakout_20"] = breakout_signal(close, lookback=20)
    indicators["breakout_50"] = breakout_signal(close, lookback=50)
    indicators["breakout_100"] = breakout_signal(close, lookback=100)

    indicators["ma_cross_5_10"] = trend_signal(close, short_window=5, long_window=10)
    indicators["ma_cross_10_20"] = trend_signal(close, short_window=10, long_window=20)
    indicators["ma_cross_20_50"] = trend_signal(close, short_window=20, long_window=50)
    indicators["ma_cross_50_100"] = trend_signal(close, short_window=50, long_window=100)

    indicators["bollinger_pct_10"] = bollinger_percent_signal(close, window=10)
    indicators["bollinger_pct_20"] = bollinger_percent_signal(close, window=20)
    indicators["bollinger_pct_50"] = bollinger_percent_signal(close, window=50)
    indicators["bollinger_pct_100"] = bollinger_percent_signal(close, window=100)

    indicators["keltner_pct_10"] = keltner_percent_signal(high, low, close, window=10)
    indicators["keltner_pct_20"] = keltner_percent_signal(high, low, close, window=20)
    indicators["keltner_pct_50"] = keltner_percent_signal(high, low, close, window=50)
    indicators["keltner_pct_100"] = keltner_percent_signal(high, low, close, window=100)

    indicators["range_expansion_10"] = range_expansion_signal(high, low, window=10)
    indicators["range_expansion_20"] = range_expansion_signal(high, low, window=20)
    indicators["range_expansion_50"] = range_expansion_signal(high, low, window=50)
    indicators["range_expansion_100"] = range_expansion_signal(high, low, window=100)
    indicators["range_expansion_short"] = indicators["range_expansion_20"]
    indicators["range_expansion_medium"] = indicators["range_expansion_50"]
    indicators["range_expansion_long"] = indicators["range_expansion_100"]

    indicators["price_vs_ma_10"] = price_vs_ma_signal(close, window=10)
    indicators["price_vs_ma_20"] = price_vs_ma_signal(close, window=20)
    indicators["price_vs_ma_50"] = price_vs_ma_signal(close, window=50)
    indicators["price_vs_ma_100"] = price_vs_ma_signal(close, window=100)

    indicators["ma_slope_10"] = moving_average_slope_signal(close, ma_window=10, slope_window=3)
    indicators["ma_slope_20"] = moving_average_slope_signal(close, ma_window=20, slope_window=5)
    indicators["ma_slope_50"] = moving_average_slope_signal(close, ma_window=50, slope_window=5)
    indicators["ma_slope_100"] = moving_average_slope_signal(close, ma_window=100, slope_window=5)

    indicators["regression_10"] = regression_signal(close, window=10)
    indicators["regression_short"] = regression_signal(close, window=20)
    indicators["regression_medium"] = regression_signal(close, window=50)
    indicators["regression_long"] = regression_signal(close, window=100)
    indicators["adx_10"] = adx_signal(high, low, close, window=10)
    indicators["adx_slope_10"] = adx_slope_signal(indicators["adx_10"], slope_window=3)

    return indicators.reindex(columns=TES_INDICATOR_COLUMNS).fillna(0.0)
