from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd


DP_FILE_RE = re.compile(
    r"^DP-(?P<symbol>[A-Za-z]+)-(?P<tag>\d{6})-(?P<edition>\d+)\.csv$",
)
PRICE_COLUMNS = ("open", "high", "low", "close")


@dataclass(frozen=True)
class SymbolFile:
    symbol: str
    path: Path
    pull_date: date
    edition: int


def parse_data_roots(value: str | None, defaults: tuple[Path, ...]) -> tuple[Path, ...]:
    if not value:
        return defaults
    roots = tuple(Path(token).expanduser() for token in value.split(",") if token.strip())
    if not roots:
        raise ValueError("No data roots were provided.")
    return roots


def parse_symbol_list(value: str) -> tuple[str, ...]:
    symbols = tuple(token.strip().upper() for token in value.split(",") if token.strip())
    if not symbols:
        raise ValueError("No symbols were provided.")
    return symbols


def parse_file_map(value: str | None) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    if not value:
        return mapping
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Invalid data file map entry {token!r}. Expected SYMBOL=/path/file.csv")
        symbol, path = token.split("=", 1)
        symbol = symbol.strip().upper()
        if not symbol:
            raise ValueError(f"Invalid data file map entry {token!r}. Missing symbol.")
        mapping[symbol] = Path(path.strip()).expanduser()
    return mapping


def _parse_pull_date(tag: str) -> date:
    month = int(tag[0:2])
    day = int(tag[2:4])
    year = 2000 + int(tag[4:6])
    return date(year, month, day)


def find_latest_symbol_file(data_roots: tuple[Path, ...], symbol: str) -> SymbolFile:
    symbol = symbol.upper()
    candidates = find_symbol_files(data_roots, symbol)
    if not candidates:
        roots = ", ".join(str(root) for root in data_roots)
        raise FileNotFoundError(f"No DP CSV found for {symbol} under: {roots}")
    return max(candidates, key=lambda item: (item.pull_date, item.edition, item.path.stat().st_mtime))


def find_symbol_files(data_roots: tuple[Path, ...], symbol: str) -> list[SymbolFile]:
    symbol = symbol.upper()
    candidates: list[SymbolFile] = []
    for root in data_roots:
        symbol_dir = root / symbol
        if not symbol_dir.exists():
            continue
        for path in symbol_dir.glob(f"DP-{symbol}-*.csv"):
            match = DP_FILE_RE.match(path.name)
            if not match:
                continue
            candidates.append(
                SymbolFile(
                    symbol=symbol,
                    path=path,
                    pull_date=_parse_pull_date(match.group("tag")),
                    edition=int(match.group("edition")),
                ),
            )
    return sorted(candidates, key=lambda item: (item.pull_date, item.edition, item.path.stat().st_mtime))


def read_price_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    frame = raw.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]

    missing = [column for column in ("date", *PRICE_COLUMNS) if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in PRICE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "vol" in frame.columns:
        frame["vol"] = pd.to_numeric(frame["vol"], errors="coerce")
    if "oi" in frame.columns:
        frame["oi"] = pd.to_numeric(frame["oi"], errors="coerce")

    keep_columns = ["date", *PRICE_COLUMNS]
    for optional_column in ("vol", "oi"):
        if optional_column in frame.columns:
            keep_columns.append(optional_column)

    frame = frame[keep_columns].dropna(subset=["date", *PRICE_COLUMNS])
    frame = frame.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    frame = frame.set_index("date", drop=True)
    frame.index.name = "date"
    return frame


def read_intraday_price_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    frame = raw.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]

    missing = [column for column in ("date", "time", *PRICE_COLUMNS) if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required intraday columns: {missing}")

    timestamp_text = frame["date"].astype(str).str.strip() + " " + frame["time"].astype(str).str.strip()
    frame["timestamp"] = pd.to_datetime(timestamp_text, errors="coerce")
    for column in PRICE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "vol" in frame.columns:
        frame["vol"] = pd.to_numeric(frame["vol"], errors="coerce")
    if "oi" in frame.columns:
        frame["oi"] = pd.to_numeric(frame["oi"], errors="coerce")

    keep_columns = ["timestamp", *PRICE_COLUMNS]
    for optional_column in ("vol", "oi"):
        if optional_column in frame.columns:
            keep_columns.append(optional_column)

    frame = frame[keep_columns].dropna(subset=["timestamp", *PRICE_COLUMNS])
    frame = frame.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    frame = frame.set_index("timestamp", drop=True)
    frame.index.name = "timestamp"
    return frame


def load_symbol_frames(
    symbols: tuple[str, ...],
    *,
    data_roots: tuple[Path, ...],
    file_map: dict[str, Path] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    file_map = file_map or {}
    frames: dict[str, pd.DataFrame] = {}
    sources: dict[str, str] = {}
    for symbol in symbols:
        symbol = symbol.upper()
        path = file_map.get(symbol)
        if path is None:
            symbol_file = find_latest_symbol_file(data_roots, symbol)
            path = symbol_file.path
        if not path.exists():
            raise FileNotFoundError(f"{symbol} data file does not exist: {path}")
        frames[symbol] = read_price_csv(path)
        sources[symbol] = str(path)
    return frames, sources


def load_intraday_symbol_frames(
    symbols: tuple[str, ...],
    *,
    data_roots: tuple[Path, ...],
    file_map: dict[str, Path] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    file_map = file_map or {}
    frames: dict[str, pd.DataFrame] = {}
    sources: dict[str, str] = {}
    for symbol in symbols:
        symbol = symbol.upper()
        explicit_path = file_map.get(symbol)
        if explicit_path is not None:
            paths = [explicit_path]
        else:
            symbol_files = find_symbol_files(data_roots, symbol)
            paths = [symbol_file.path for symbol_file in symbol_files]
        if not paths:
            roots = ", ".join(str(root) for root in data_roots)
            raise FileNotFoundError(f"No intraday DP CSV found for {symbol} under: {roots}")

        symbol_frames = []
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"{symbol} intraday data file does not exist: {path}")
            symbol_frames.append(read_intraday_price_csv(path))
        merged = pd.concat(symbol_frames).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        frames[symbol] = merged
        sources[symbol] = ";".join(str(path) for path in paths)
    return frames, sources
