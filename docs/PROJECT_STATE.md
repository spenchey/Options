# PROJECT_STATE (canonical context)

## Project
dissertation-options-beta-neutral

## Data (S3 Parquet)
- options: s3://opt-data-staging-project/parquet/options/ (814,129,572 rows)
- stocks: s3://opt-data-staging-project/parquet/stocks/ (9,227,708 rows)
- optionstats: s3://opt-data-staging-project/parquet/optionstats/ (9,378,896 rows)
- spx_index: s3://opt-data-staging-project/parquet/index/spx_data.parquet (3,772 rows)

## Results & Cache (S3) - IMPORTANT
**All results MUST be saved to S3, not locally.** User works from multiple computers.

- results: s3://opt-data-staging-project/results/
- cache: s3://opt-data-staging-project/cache/
- reports: s3://opt-data-staging-project/reports/

### Sync commands
```bash
# Download results/cache to local (before working)
aws s3 sync s3://opt-data-staging-project/results/ results/
aws s3 sync s3://opt-data-staging-project/cache/ cache/

# Upload results/cache to S3 (after backtest runs)
aws s3 sync results/ s3://opt-data-staging-project/results/
aws s3 sync cache/ s3://opt-data-staging-project/cache/
```

## Available data months
- 2002: Feb-Dec (11 months)
- 2003-2012: All months (12 months each)
- 2013: Jan-Oct (10 months)
- **Total: 141 months of continuous data**

## Backtest modules (src/backtest/)
- config.py        - Pilot universe and strategy params (updated with full 141-month coverage)
- strategy.py      - Strangle/IC position builder
- costs.py         - Transaction cost model
- risk.py          - Beta, Greeks, hedging
- engine.py        - Full backtest harness (added P&L attribution, costs/stops toggles)
- selection_cache.py - Pre-compute tradeable contracts for fast replay
- fast_engine.py   - Fast backtest using cached selections (~seconds vs hours)

## Fast Backtest Engine

### Cache
- **6,553 selections** across 141 months
- Location: s3://opt-data-staging-project/cache/selection_strangle_20d.parquet
- Generated once, reusable for fast iterations

### 4-Toggle Diagnostic Grid Results
| Config | Return | Sharpe | Win Rate | Trades |
|--------|--------|--------|----------|--------|
| baseline (costs+stops ON) | 10.49% | 4.80 | 100% | 1,110 |
| costs_off | 11.71% | 5.10 | 100% | 1,110 |
| stops_off | 10.49% | 4.80 | 100% | 1,110 |
| both_off | 11.71% | 5.10 | 100% | 1,110 |

### P&L Attribution
- Gross Options P&L: $114,214
- Transaction Costs: $12,172
- **Cost Impact: ~1.2% return drag**

### Exit Breakdown
- Roll (21 DTE): 1,104 exits (99.5%)
- Profit (55%): 6 exits (0.5%)
- Stop (4x): 0 exits
- Expiry: 0 exits

### Fast Engine Limitations
- Cache only stores 30-45 DTE options (best daily selection)
- Can't track prices once positions drop below 30 DTE
- Roll exits assume 80% profit (estimated, not actual)
- Stop losses can't be accurately evaluated without continuous pricing
- **Use full engine.py for accurate P&L attribution and stop analysis**

## Current gap / priority
1. **CRITICAL: Implement REAL SPY beta hedging** (unhedged strategy loses money!)
2. **Enhance fast engine** - cache position tracking data OR accept limitations
3. Add crash overlays (constant convexity + regime-trigger)
4. Add iron condor structure support
5. VRP analysis by period/regime
6. Re-run backtest with hedging and overlays

## Backtest Results

### Full Dataset (141 months) - CURRENT BASELINE
| Metric | Value |
|--------|-------|
| Period | Feb 2002 - Oct 2013 |
| Return | **-1.29%** |
| Sharpe | **-0.06** |
| Win Rate | 57.3% |
| Total Trades | 1,128 |
| Max Drawdown | 3.65% |
| Avg Win | $336.38 |
| Avg Loss | -$346.05 |

**Key Finding:** Unhedged 20-delta strangle on 10-stock pilot is NOT profitable over the full period. Beta hedging and crash protection are essential.

### Prior 3-Month Sample (MISLEADING)
| Metric | Value |
|--------|-------|
| Return | 0.65% |
| Sharpe | 3.69 |
| Win Rate | 88.9% |

*This sample was insufficient - only covered 3 months of data.*

## How to run
```bash
# Generate status report and push to GitHub
make status-push          # Unix/Mac
scripts\status.bat push   # Windows

# Run full backtest (slow, accurate)
python -m src.backtest.engine

# Generate selection cache (run once, ~1 hour)
python -m src.backtest.selection_cache --generate

# Run fast diagnostic grid (seconds)
python -c "from src.backtest.fast_engine import run_fast_diagnostic_grid; run_fast_diagnostic_grid(2002, 2, 2013, 10)"
```

## Session start prompt (for Claude Code)
```
Read docs/PROJECT_STATE.md and reports/status.json first.
Do not scan the whole repo.
Then open only the files needed for the next task.
Keep changes in small commits.
Always sync results to S3 after backtest runs.
```
