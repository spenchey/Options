# Project Status

**Project:** dissertation-options-beta-neutral
**Last update (UTC):** 2025-12-22T15:49:19Z
**Data coverage:** 141 months (Feb 2002 - Oct 2013)

## Data Pipeline

| Dataset | Rows | Location |
|---------|------|----------|
| options | 814,129,572 | `s3://opt-data-staging-project/parquet/options/` |
| stocks | 9,227,708 | `s3://opt-data-staging-project/parquet/stocks/` |
| optionstats | 9,378,896 | `s3://opt-data-staging-project/parquet/optionstats/` |
| spx_index | 3,772 | `s3://opt-data-staging-project/parquet/index/spx_data.parquet` |

## Backtest Framework

| Module | Status |
|--------|--------|
| config | Ready |
| strategy | Ready |
| costs | Ready |
| risk | Ready |
| engine | Ready |

## Latest Backtest Results (FULL 141-MONTH RUN)

| Metric | Value |
|--------|-------|
| Period | Feb 2002 - Oct 2013 |
| Return | **-1.29%** |
| Sharpe | **-0.058** |
| Max Drawdown | 3.65% |
| Win Rate | 57.3% |
| Total Trades | 1,128 |
| Avg Win | $336.38 |
| Avg Loss | -$346.05 |

### Key Finding

> **Unhedged 20-delta strangle on 10-stock pilot is NOT profitable over the full period.**
> Beta hedging and crash protection are essential.

## Priority Next Steps

1. **CRITICAL:** Implement real SPY beta hedging
2. Add P&L attribution (options vs hedge vs costs)
3. Run 4-toggle diagnostic grid (Costs ON/OFF x Stops ON/OFF)
4. Add crash protection overlays (constant convexity + regime-trigger)
5. Add iron condor structure support

---
*Auto-generated at 2025-12-22T15:49:19Z*
