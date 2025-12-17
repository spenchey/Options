#!/usr/bin/env python
"""
Run the pilot backtest with vega targeting and beta hedging.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.options_db import OptionsDB
from src.market_data import MarketData
from src.backtest.engine import BacktestEngine, print_results, save_results
from src.backtest.config import get_pilot_tickers

def main():
    print("=" * 60)
    print("BETA-NEUTRAL SHORT STRANGLE BACKTEST")
    print("Vega-targeted sizing + Real SPY hedging")
    print("=" * 60)

    # Initialize data connections
    print("\nInitializing data connections...")
    db = OptionsDB()
    md = MarketData()

    # Get pilot tickers
    tickers = get_pilot_tickers()
    print(f"Pilot universe: {tickers}")

    # Create engine
    engine = BacktestEngine(db, md, capital=1_000_000, verbose=True)

    # Run backtest over available periods (2011-2013)
    print("\nRunning backtest...")
    results = engine.run(
        tickers=tickers,
        start_year=2011,
        start_month=1,
        end_year=2013,
        end_month=12
    )

    # Print results
    print_results(results)

    # Save results
    print("\nSaving results...")
    output_dir = save_results(results, "results/baseline/strangle")

    print(f"\nResults saved to: {output_dir}")
    print("\nDone!")

    return results

if __name__ == "__main__":
    main()
