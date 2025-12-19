# PROJECT_STATE (canonical context)

## Project
dissertation-options-beta-neutral

## Data (S3 Parquet)
- options: s3://opt-data-staging-project/parquet/options/ (814,129,572 rows)
- stocks: s3://opt-data-staging-project/parquet/stocks/ (9,227,708 rows)
- optionstats: s3://opt-data-staging-project/parquet/optionstats/ (9,378,896 rows)
- spx_index: s3://opt-data-staging-project/parquet/index/spx_data.parquet (3,772 rows)

## Available data months
- 2002: Feb-Dec (11 months)
- 2003-2012: All months (12 months each)
- 2013: Jan-Oct (10 months)
- **Total: 141 months of continuous data**

## Backtest modules (src/backtest/)
- config.py   - Pilot universe and strategy params
- strategy.py - Strangle/IC position builder
- costs.py    - Transaction cost model
- risk.py     - Beta, Greeks, hedging
- engine.py   - Backtest harness

## Current gap / priority
1. Implement REAL SPY beta hedging in engine (daily EOD hedge + threshold re-hedge)
2. Add iron condor structure support
3. Add crash overlays (constant convexity + regime-trigger)
4. VRP analysis by period/regime
5. Run full backtest on complete 2002-2013 dataset

## Baseline results (10-stock pilot strangle)
- Win rate: 88.9%
- Sharpe: 3.69
- Return: 0.65%

## How to run
```bash
# Generate status report and push to GitHub
make status-push          # Unix/Mac
scripts\status.bat push   # Windows

# Run backtest (placeholder - update when implemented)
python -m src.backtest.engine
```

## Session start prompt (for Claude Code)
```
Read docs/PROJECT_STATE.md and reports/status.json first.
Do not scan the whole repo.
Then open only the files needed for the next task.
Keep changes in small commits.
```
