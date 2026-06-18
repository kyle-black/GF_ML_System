from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ChartConfig:
    backtest_dir: Path
    output_dir: Path
    title: str = ""
    rolling_window: int = 20


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_backtest(backtest_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    trades = _read_csv_if_exists(backtest_dir / "trades.csv")
    annual = _read_csv_if_exists(backtest_dir / "annual.csv")
    monthly = _read_csv_if_exists(backtest_dir / "monthly.csv")
    summary_path = backtest_dir / "summary.json"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())

    if not trades.empty:
        for column in ("signal_date", "entry_date", "exit_date"):
            if column in trades.columns:
                trades[column] = pd.to_datetime(trades[column], errors="coerce")
        numeric_columns = [
            "gross_points",
            "gross_pnl",
            "commission",
            "net_pnl",
            "cum_net_pnl",
            "cum_gross_points",
            "drawdown_net_pnl",
            "runup_points",
            "drawdown_points",
            "long_probability",
            "short_probability",
            "atr_points",
        ]
        for column in numeric_columns:
            if column in trades.columns:
                trades[column] = pd.to_numeric(trades[column], errors="coerce")
        trades = trades.sort_values(["exit_date", "trade_number"], na_position="last").reset_index(drop=True)

    return trades, annual, monthly, summary


def _format_currency(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"${float(value):,.0f}"


def _format_number(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{float(value):,.{digits}f}"


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#d9dee7", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#9aa6b2")
    ax.spines["bottom"].set_color("#9aa6b2")
    ax.tick_params(colors="#334155")
    ax.title.set_color("#111827")
    ax.xaxis.label.set_color("#334155")
    ax.yaxis.label.set_color("#334155")


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_equity(trades: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(trades["exit_date"], trades["cum_net_pnl"], color="#2563eb", linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel("Exit date")
    ax.set_ylabel("Cumulative net PnL")
    ax.yaxis.set_major_formatter(lambda x, _pos: f"${x/1_000_000:.1f}M")
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _style_axis(ax)
    _savefig(fig, path)


def _plot_drawdown(trades: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.fill_between(
        trades["exit_date"].to_numpy(),
        trades["drawdown_net_pnl"].to_numpy(dtype=float),
        0.0,
        color="#dc2626",
        alpha=0.22,
    )
    ax.plot(trades["exit_date"], trades["drawdown_net_pnl"], color="#b91c1c", linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel("Exit date")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(lambda x, _pos: f"${x/1_000:.0f}K")
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _style_axis(ax)
    _savefig(fig, path)


def _plot_annual(annual: pd.DataFrame, path: Path, title: str) -> None:
    frame = annual.copy()
    frame["period"] = frame["period"].astype(str)
    frame["net_pnl"] = pd.to_numeric(frame["net_pnl"], errors="coerce")
    colors = np.where(frame["net_pnl"] >= 0, "#16a34a", "#dc2626")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(frame["period"], frame["net_pnl"], color=colors)
    ax.axhline(0, color="#111827", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Net PnL")
    ax.yaxis.set_major_formatter(lambda x, _pos: f"${x/1_000:.0f}K")
    ax.tick_params(axis="x", rotation=45)
    _style_axis(ax)
    _savefig(fig, path)


def _plot_monthly_heatmap(monthly: pd.DataFrame, path: Path, title: str) -> None:
    frame = monthly.copy()
    frame["period_date"] = pd.to_datetime(frame["period"].astype(str), errors="coerce")
    frame["net_pnl"] = pd.to_numeric(frame["net_pnl"], errors="coerce")
    frame = frame.loc[frame["period_date"].notna()]
    frame["year"] = frame["period_date"].dt.year
    frame["month"] = frame["period_date"].dt.month
    pivot = frame.pivot_table(index="year", columns="month", values="net_pnl", aggfunc="sum").sort_index()
    pivot = pivot.reindex(columns=range(1, 13))

    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.45 * len(pivot) + 2.5)))
    values = pivot.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    limit = np.nanpercentile(np.abs(finite), 90) if finite.size else 1.0
    limit = max(limit, 1.0)
    im = ax.imshow(values, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(np.arange(12), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index.astype(str).tolist())
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            value = values[y, x]
            if np.isfinite(value):
                ax.text(x, y, f"{value/1_000:.0f}", ha="center", va="center", fontsize=7, color="#111827")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.ax.set_ylabel("Net PnL, $K")
    _savefig(fig, path)


def _plot_trade_distribution(trades: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    values = trades["net_pnl"].dropna()
    ax.hist(values, bins=45, color="#64748b", edgecolor="white", alpha=0.9)
    ax.axvline(0, color="#111827", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Net PnL per trade")
    ax.set_ylabel("Trade count")
    ax.xaxis.set_major_formatter(lambda x, _pos: f"${x/1_000:.0f}K")
    _style_axis(ax)
    _savefig(fig, path)


def _plot_rolling(trades: pd.DataFrame, path: Path, title: str, window: int) -> None:
    frame = trades.copy()
    frame["rolling_net"] = frame["net_pnl"].rolling(window, min_periods=max(3, window // 3)).sum()
    frame["rolling_win_rate"] = (
        (frame["net_pnl"] > 0).astype(float).rolling(window, min_periods=max(3, window // 3)).mean()
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(frame["exit_date"], frame["rolling_net"], color="#7c3aed", linewidth=1.8, label=f"{window}-trade net")
    ax.axhline(0, color="#111827", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Exit date")
    ax.set_ylabel(f"{window}-trade net PnL")
    ax.yaxis.set_major_formatter(lambda x, _pos: f"${x/1_000:.0f}K")
    ax2 = ax.twinx()
    ax2.plot(frame["exit_date"], frame["rolling_win_rate"], color="#0f766e", linewidth=1.3, alpha=0.8, label="Win rate")
    ax2.set_ylabel("Rolling win rate")
    ax2.set_ylim(0, 1)
    _style_axis(ax)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color("#9aa6b2")
    ax2.tick_params(colors="#334155")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left", frameon=False)
    _savefig(fig, path)


def _plot_side_summary(trades: pd.DataFrame, path: Path, title: str) -> None:
    grouped = trades.groupby("side", sort=True).agg(
        net_pnl=("net_pnl", "sum"),
        trades=("net_pnl", "size"),
        avg_trade=("net_pnl", "mean"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = ["#2563eb" if side == "long" else "#f97316" for side in grouped.index]
    axes[0].bar(grouped.index, grouped["net_pnl"], color=colors)
    axes[0].set_title("Net PnL By Side")
    axes[0].set_ylabel("Net PnL")
    axes[0].yaxis.set_major_formatter(lambda x, _pos: f"${x/1_000:.0f}K")
    axes[1].bar(grouped.index, grouped["trades"], color=colors)
    axes[1].set_title("Trade Count By Side")
    axes[1].set_ylabel("Trades")
    for ax in axes:
        _style_axis(ax)
    fig.suptitle(title, fontsize=14, color="#111827")
    _savefig(fig, path)


def _plot_exit_reasons(trades: pd.DataFrame, path: Path, title: str) -> None:
    grouped = trades.groupby("exit_reason", sort=True).agg(net_pnl=("net_pnl", "sum"), trades=("net_pnl", "size"))
    grouped = grouped.sort_values("net_pnl", ascending=False)
    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.45 * len(grouped) + 2)))
    colors = np.where(grouped["net_pnl"] >= 0, "#16a34a", "#dc2626")
    ax.barh(grouped.index, grouped["net_pnl"], color=colors)
    ax.axvline(0, color="#111827", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Net PnL")
    ax.xaxis.set_major_formatter(lambda x, _pos: f"${x/1_000:.0f}K")
    _style_axis(ax)
    _savefig(fig, path)


def _metric_cards(summary: dict[str, Any]) -> list[tuple[str, str]]:
    overall = summary.get("overall", {})
    return [
        ("Trades", f"{int(overall.get('trades', 0)):,}"),
        ("Net PnL", _format_currency(overall.get("net_pnl"))),
        ("Gross Points", _format_number(overall.get("gross_points"), 2)),
        ("Profit Factor", _format_number(overall.get("profit_factor"), 2)),
        ("Win Rate", f"{100 * float(overall.get('win_rate') or 0):.1f}%"),
        ("Max DD", _format_currency(overall.get("max_drawdown_net_pnl"))),
        ("Avg Trade", _format_currency(overall.get("avg_trade_net_pnl"))),
        ("Period", f"{overall.get('start_date', 'n/a')} to {overall.get('end_date', 'n/a')}"),
    ]


def _write_html_report(
    *,
    output_dir: Path,
    title: str,
    summary: dict[str, Any],
    chart_files: list[tuple[str, Path]],
) -> Path:
    cards = "\n".join(
        f"<div class='metric'><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>"
        for label, value in _metric_cards(summary)
    )
    figures = "\n".join(
        f"<section><h2>{html.escape(label)}</h2><img src='{html.escape(path.name)}' alt='{html.escape(label)}'></section>"
        for label, path in chart_files
    )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #111827; background: #f8fafc; }}
    header {{ padding: 24px 28px 12px; background: #ffffff; border-bottom: 1px solid #e5e7eb; }}
    h1 {{ margin: 0 0 16px; font-size: 24px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }}
    .metric {{ background: #f1f5f9; border: 1px solid #e2e8f0; padding: 12px; border-radius: 6px; }}
    .metric span {{ display: block; font-size: 12px; color: #475569; margin-bottom: 6px; }}
    .metric strong {{ font-size: 17px; }}
    main {{ padding: 18px 28px 32px; }}
    section {{ background: #ffffff; border: 1px solid #e5e7eb; border-radius: 6px; padding: 14px; margin: 0 0 16px; }}
    h2 {{ margin: 0 0 10px; font-size: 17px; }}
    img {{ width: 100%; height: auto; display: block; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <div class="metrics">{cards}</div>
  </header>
  <main>{figures}</main>
</body>
</html>
"""
    path = output_dir / "index.html"
    path.write_text(html_text)
    return path


def generate_charts(config: ChartConfig) -> dict[str, Any]:
    trades, annual, monthly, summary = _load_backtest(config.backtest_dir)
    if trades.empty:
        raise RuntimeError(f"No trades found in {config.backtest_dir / 'trades.csv'}")
    if annual.empty:
        annual = _derive_period_summary(trades, "Y")
    if monthly.empty:
        monthly = _derive_period_summary(trades, "M")

    title = config.title or config.backtest_dir.name.replace("_", " ").title()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    chart_specs = [
        ("Equity Curve", output_dir / "equity_curve.png", _plot_equity, (trades, "Equity Curve")),
        ("Drawdown", output_dir / "drawdown.png", _plot_drawdown, (trades, "Drawdown")),
        ("Annual Net PnL", output_dir / "annual_net_pnl.png", _plot_annual, (annual, "Annual Net PnL")),
        ("Monthly Heatmap", output_dir / "monthly_heatmap.png", _plot_monthly_heatmap, (monthly, "Monthly Net PnL, $K")),
        ("Trade Distribution", output_dir / "trade_distribution.png", _plot_trade_distribution, (trades, "Trade Distribution")),
        (
            "Rolling Performance",
            output_dir / "rolling_performance.png",
            _plot_rolling,
            (trades, f"Rolling {config.rolling_window}-Trade Performance", config.rolling_window),
        ),
        ("Side Summary", output_dir / "side_summary.png", _plot_side_summary, (trades, "Side Summary")),
        ("Exit Reasons", output_dir / "exit_reasons.png", _plot_exit_reasons, (trades, "Exit Reasons")),
    ]

    chart_files: list[tuple[str, Path]] = []
    for label, path, plotter, args in chart_specs:
        # Keep the call explicit so each plotter signature remains obvious.
        if plotter is _plot_rolling:
            plotter(args[0], path, args[1], args[2])
        else:
            plotter(args[0], path, args[1])
        chart_files.append((label, path))

    html_path = _write_html_report(output_dir=output_dir, title=title, summary=summary, chart_files=chart_files)
    manifest = {
        "backtest_dir": str(config.backtest_dir),
        "output_dir": str(output_dir),
        "title": title,
        "html_report": str(html_path),
        "charts": {label: str(path) for label, path in chart_files},
    }
    (output_dir / "chart_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _derive_period_summary(trades: pd.DataFrame, period: str) -> pd.DataFrame:
    frame = trades.copy()
    frame["period"] = frame["exit_date"].dt.to_period(period).astype(str)
    rows: list[dict[str, Any]] = []
    for value, group in frame.groupby("period", sort=True):
        rows.append(
            {
                "period": value,
                "trades": int(len(group)),
                "net_pnl": float(group["net_pnl"].sum()),
                "gross_points": float(group["gross_points"].sum()),
                "win_rate": float((group["net_pnl"] > 0).mean()),
            },
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate static charts for a local ML backtest folder.")
    parser.add_argument("--backtest-dir", required=True, help="Folder containing trades.csv and summary.json.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Chart output folder. Defaults to <backtest-dir>/charts.",
    )
    parser.add_argument("--title", default="", help="Report title.")
    parser.add_argument("--rolling-window", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backtest_dir = Path(args.backtest_dir)
    output_dir = Path(args.output_dir) if args.output_dir else backtest_dir / "charts"
    manifest = generate_charts(
        ChartConfig(
            backtest_dir=backtest_dir,
            output_dir=output_dir,
            title=args.title,
            rolling_window=args.rolling_window,
        ),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
