from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SYMBOLS = ("ES", "NQ", "TY", "US", "GC", "SI")
DEFAULT_TARGET_SYMBOL = "ES"
DEFAULT_DATA_ROOTS = (
    PROJECT_ROOT / "data" / "historical_data",
    Path("/home/kyle-black/GaleForce_TrendEngine/data/historical_data"),
)
DEFAULT_INTRADAY_60M_ROOTS = (
    PROJECT_ROOT / "data" / "historical_data_60min",
    Path("/home/kyle-black/GF_ALGO_ML/src/trade_station_connect/src/data/historical_data_60m"),
    Path("/home/kyle-black/GF_ML_Analysis/trade_station_connect/data/historical_data_60min"),
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "ml" / "tes_intermarket"
