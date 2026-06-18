# GF ML System

First-pass TES intermarket ML pipeline for ES.

## What It Builds

- Target: ES next-day directional ATR touches.
- Labels: `target_up_0_25atr`, `target_down_0_25atr`, `target_up_0_5atr`, `target_down_0_5atr`.
- Features: TES indicators from `ES,NQ,TY,US,GC,SI`.
- Walk-forward: rolling train window with 50-row test blocks.
- Model: dependency-light standardized ridge classifier so the first pipeline runs in this repo env.

## Pull Fresh TradeStation Data

```bash
PYTHONPATH=src ./env/bin/python -m gf_ml_system.pull_tradestation_data \
  --futures ES,NQ,TY,US,GC,SI \
  --unit Daily \
  --interval 1 \
  --barsback 20000
```

The wrapper calls the existing GaleForce TradeStation connector and writes DP CSVs under:

```text
data/historical_data/<SYMBOL>/DP-<SYMBOL>-<MMDDYY>-<EDITION>.csv
```

## Train ES First

```bash
PYTHONPATH=src ./env/bin/python -m gf_ml_system.train_es_tes_intermarket \
  --target-symbol ES \
  --symbols ES,NQ,TY,US,GC,SI \
  --atr-multipliers 0.25,0.5 \
  --min-train-rows 750 \
  --rolling-train-rows 750 \
  --test-rows 50 \
  --step-rows 50
```

To run the same setup with XGBoost:

```bash
PYTHONPATH=src ./env/bin/python -m gf_ml_system.train_es_tes_intermarket \
  --run-name es_next_day_atr_tes_intermarket_xgboost \
  --data-roots data/historical_data \
  --target-symbol ES \
  --symbols ES,NQ,TY,US,GC,SI \
  --atr-multipliers 0.25,0.5 \
  --min-train-rows 750 \
  --rolling-train-rows 750 \
  --test-rows 50 \
  --step-rows 50 \
  --model-type xgboost
```

To train the open-to-close direction target for the TradeStation buy/sell-open and sell/buy-close test:

```bash
PYTHONPATH=src ./env/bin/python -m gf_ml_system.train_es_tes_intermarket \
  --run-name es_next_open_to_close_tes_intermarket_xgboost \
  --data-roots data/historical_data \
  --target-symbol ES \
  --symbols ES,NQ,TY,US,GC,SI \
  --target-mode open_close \
  --min-train-rows 750 \
  --rolling-train-rows 750 \
  --test-rows 50 \
  --step-rows 50 \
  --model-type xgboost
```

To train a thresholded open-to-close target, where the next session must move at least `0.25 ATR` from open to close:

```bash
PYTHONPATH=src ./env/bin/python -m gf_ml_system.train_es_tes_intermarket \
  --run-name es_next_open_to_close_025atr_tes_intermarket_xgboost \
  --data-roots data/historical_data \
  --target-symbol ES \
  --symbols ES,NQ,TY,US,GC,SI \
  --target-mode open_close_atr \
  --atr-multipliers 0.25 \
  --min-train-rows 750 \
  --rolling-train-rows 750 \
  --test-rows 50 \
  --step-rows 50 \
  --model-type xgboost
```

In the existing EasyLanguage strategy, this maps to selector `2` because the exporter places the thresholded open-to-close `0.25 ATR` probabilities into `prob_up_0_25_atr` and `prob_down_0_25_atr`.

To add 60-minute RTH context to that same target, use `--include-intraday-60m`.
The trainer merges all available 60-minute DP files per symbol, filters to bars stamped `10:00` through `16:00`, and only uses the last completed intraday bar for each signal date:

```bash
PYTHONPATH=src ./env/bin/python -m gf_ml_system.train_es_tes_intermarket \
  --run-name es_next_open_to_close_025atr_tes_daily_rth60_xgboost \
  --data-roots data/historical_data \
  --target-symbol ES \
  --symbols ES,NQ,TY,US,GC,SI \
  --target-mode open_close_atr \
  --atr-multipliers 0.25 \
  --min-train-rows 750 \
  --rolling-train-rows 750 \
  --test-rows 50 \
  --step-rows 50 \
  --model-type xgboost \
  --include-intraday-60m
```

If this repo has no fresh `data/historical_data` files yet, the trainer also checks the existing local fallback:

```text
/home/kyle-black/GaleForce_TrendEngine/data/historical_data
```

## Outputs

Default run folder:

```text
data/ml/tes_intermarket/es_next_day_atr_tes_intermarket/
```

Key artifacts:

- `training_dataset.csv`
- `walk_forward_predictions.csv`
- `walk_forward_folds.csv`
- `final_fit_predictions.csv`
- `latest_signal.csv`
- `feature_importance.csv`
- `model_artifact.json`
- `metrics.json`

## Export Testing Rows For TradeStation

The EasyLanguage backtest strategy expects the walk-forward testing predictions only, headerless, in this 15-column order:

```text
el_date,ts_date,date_iso,prob_up_close,prob_down_close,prob_up_0_25_atr,prob_down_0_25_atr,prob_up_0_5_atr,prob_down_0_5_atr,prob_up_0_75_atr,prob_down_0_75_atr,prob_up_1_atr,prob_down_1_atr,highest_up_atr_probability,highest_down_atr_probability
```

Export the XGBoost run:

```bash
PYTHONPATH=src ./env/bin/python -m gf_ml_system.export_tradestation_backtest \
  --run-name es_next_day_atr_tes_intermarket_xgboost
```

For the open-to-close run, use selector `1` in TradeStation because the exporter populates `prob_up_close` and `prob_down_close`:

```bash
PYTHONPATH=src ./env/bin/python -m gf_ml_system.export_tradestation_backtest \
  --run-name es_next_open_to_close_tes_intermarket_xgboost
```

For the thresholded `0.25 ATR` open-to-close run with 60-minute features:

```bash
PYTHONPATH=src ./env/bin/python -m gf_ml_system.export_tradestation_backtest \
  --run-name es_next_open_to_close_025atr_tes_daily_rth60_xgboost
```

To export the same run as long-only, with every short/down probability zeroed:

```bash
PYTHONPATH=src ./env/bin/python -m gf_ml_system.export_tradestation_backtest \
  --run-name es_next_open_to_close_025atr_tes_daily_rth60_xgboost \
  --side-policy long-only
```

This writes:

```text
data/ml/tes_intermarket/es_next_open_to_close_025atr_tes_daily_rth60_xgboost/tradestation_easylanguage_backtest_long_only.csv
```

To export a short-only version, with every long/up probability zeroed:

```bash
PYTHONPATH=src ./env/bin/python -m gf_ml_system.export_tradestation_backtest \
  --run-name es_next_open_to_close_025atr_tes_daily_rth60_xgboost_wf20_train500 \
  --side-policy short-only
```

This writes:

```text
data/ml/tes_intermarket/es_next_open_to_close_025atr_tes_daily_rth60_xgboost_wf20_train500/tradestation_easylanguage_backtest_short_only.csv
```

Default output:

```text
data/ml/tes_intermarket/es_next_day_atr_tes_intermarket_xgboost/tradestation_easylanguage_backtest.csv
```

## Live Champion Export

The frozen ES champion is `micro_rth_esret003_close70_nqpos`: true
multi-horizon ensemble, true meta filter, stress switch, then the micro RTH
long overlay. Run it with a fresh TradeStation pull:

```bash
PYTHONPATH=src ./env/bin/python -m gf_ml_system.live_micro_rth_champion \
  --pull-data
```

By default the live pull asks TradeStation for the current-day bar. If you only
want completed prior-session data, add `--exclude-current-day`. The live scorer
also has a 16:05 New York current-day guard: before that time, if today's row is
present, it is dropped and the prior row's next-session labels are cleared so
the rolling refit cannot learn from an incomplete current day. To override this
for an explicit provisional test, add `--allow-provisional-current-day`.

Parent/master output:

```text
data/ml/tes_intermarket/live_micro_rth_esret003_close70_nqpos/tradestation_master.csv
data/ml/tes_intermarket/live_micro_rth_esret003_close70_nqpos/tradestation_master_with_header.csv
data/ml/tes_intermarket/live_micro_rth_esret003_close70_nqpos/latest_prediction.csv
data/ml/tes_intermarket/live_micro_rth_esret003_close70_nqpos/tradestation_latest.csv
```

Each run also writes its own dated audit folder:

```text
data/ml/tes_intermarket/live_micro_rth_esret003_close70_nqpos/runs/YYYY-MM-DD_HHMMSS/
```

## Local Backtest Engine

Use the local engine to test the same prediction files without relying on
TradeStation CSV loading, chart sessions, or report formatting. For the ES
meta-filter balanced-DD export:

```bash
PYTHONPATH=src ./env/bin/python -m gf_ml_system.backtest \
  --input data/ml/tes_intermarket/es_next_open_to_close_025atr_tes_daily_rth60_xgboost_wf20_multihorizon/walk_forward_predictions_meta_filter_balanced_dd.csv \
  --output-dir data/ml/tes_intermarket/es_next_open_to_close_025atr_tes_daily_rth60_xgboost_wf20_multihorizon/backtests/meta_filter_balanced_dd_local \
  --long-threshold 0.375 \
  --short-threshold 0.35 \
  --contracts 25 \
  --point-value 50 \
  --commission-per-contract-side 2.12 \
  --hold-bars 1 \
  --use-atr-stop \
  --atr-stop-multiple 1 \
  --same-bar-priority stop
```

Key outputs:

```text
summary.json
trades.csv
annual.csv
monthly.csv
skipped_rows.csv
```

The engine uses the signal row to enter the next available daily bar at the
open and exits after `--hold-bars` daily bars at the close unless an ATR
stop/target is hit. When a stop and target are both touched on the same daily
bar, `--same-bar-priority stop` is the conservative default.

## Chart Backtest Results

After running a local backtest, generate a static chart report:

```bash
PYTHONPATH=src ./env/bin/python -m gf_ml_system.charting \
  --backtest-dir data/ml/tes_intermarket/es_next_open_to_close_025atr_tes_daily_rth60_xgboost_wf20_multihorizon/backtests/meta_filter_balanced_dd_local \
  --title "ES Meta Filter Balanced DD"
```

This writes:

```text
charts/index.html
charts/equity_curve.png
charts/drawdown.png
charts/annual_net_pnl.png
charts/monthly_heatmap.png
charts/trade_distribution.png
charts/rolling_performance.png
charts/side_summary.png
charts/exit_reasons.png
```
