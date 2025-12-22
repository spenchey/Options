"""
Selection Cache - Pre-compute tradeable contracts to speed up backtests.

The expensive part of backtesting is scanning S3 to find 20-delta options each day.
This module pre-computes the selection ONCE and caches it locally.

Usage:
    # Generate cache (run once, takes ~1 hour for full dataset)
    python -m src.backtest.selection_cache --generate

    # Then backtest runs use the cache (seconds instead of hours)
    from src.backtest.selection_cache import load_selection_cache
    cache = load_selection_cache()
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

from ..options_db import OptionsDB
from .config import STRATEGY_PARAMS, PILOT_UNIVERSE, DATA_PERIODS, get_pilot_tickers

# Cache location
CACHE_DIR = Path(__file__).resolve().parents[2] / "cache"
SELECTION_CACHE_FILE = CACHE_DIR / "selection_strangle_20d.parquet"
CACHE_METADATA_FILE = CACHE_DIR / "selection_metadata.json"


def find_strangle_candidates(options_df: pd.DataFrame, ticker: str, date: str) -> list:
    """
    Find strangle candidates for a ticker on a given date.
    Returns list of dicts with selected put/call info.
    """
    # Filter for this ticker and date
    df = options_df[
        (options_df['UnderlyingSymbol'] == ticker) &
        (options_df['DataDate'] == pd.to_datetime(date))
    ].copy()

    if len(df) == 0:
        return []

    # Get underlying price
    underlying_price = df['UnderlyingPrice'].iloc[0]

    # Filter by DTE (30-45 days)
    df['Expiration'] = pd.to_datetime(df['Expiration'])
    df['DTE'] = (df['Expiration'] - pd.to_datetime(date)).dt.days
    df = df[(df['DTE'] >= STRATEGY_PARAMS['dte_min']) &
            (df['DTE'] <= STRATEGY_PARAMS['dte_max'])]

    if len(df) == 0:
        return []

    # Find 20-delta options
    delta_target = STRATEGY_PARAMS['delta_target']
    delta_tol = STRATEGY_PARAMS['delta_tolerance']

    # Puts: delta around -0.20
    puts = df[(df['Type'].str.lower() == 'put') &
              (df['Delta'].between(-delta_target - delta_tol, -delta_target + delta_tol))]

    # Calls: delta around +0.20
    calls = df[(df['Type'].str.lower() == 'call') &
               (df['Delta'].between(delta_target - delta_tol, delta_target + delta_tol))]

    if len(puts) == 0 or len(calls) == 0:
        return []

    candidates = []

    # For each expiration, find best put/call pair
    for exp in puts['Expiration'].unique():
        exp_puts = puts[puts['Expiration'] == exp]
        exp_calls = calls[calls['Expiration'] == exp]

        if len(exp_puts) == 0 or len(exp_calls) == 0:
            continue

        # Select put closest to target delta
        best_put = exp_puts.iloc[(exp_puts['Delta'] + delta_target).abs().argmin()]
        best_call = exp_calls.iloc[(exp_calls['Delta'] - delta_target).abs().argmin()]

        # Calculate credit
        put_mid = (best_put['Bid'] + best_put['Ask']) / 2
        call_mid = (best_call['Bid'] + best_call['Ask']) / 2
        credit = put_mid + call_mid

        if credit <= 0:
            continue

        candidates.append({
            'ticker': ticker,
            'date': date,
            'expiration': str(exp.date()),
            'dte': int(best_put['DTE']),
            'underlying_price': underlying_price,
            # Put leg
            'put_strike': best_put['Strike'],
            'put_bid': best_put['Bid'],
            'put_ask': best_put['Ask'],
            'put_delta': best_put['Delta'],
            'put_gamma': best_put['Gamma'],
            'put_theta': best_put['Theta'],
            'put_vega': best_put['Vega'],
            'put_iv': best_put['IV'],
            # Call leg
            'call_strike': best_call['Strike'],
            'call_bid': best_call['Bid'],
            'call_ask': best_call['Ask'],
            'call_delta': best_call['Delta'],
            'call_gamma': best_call['Gamma'],
            'call_theta': best_call['Theta'],
            'call_vega': best_call['Vega'],
            'call_iv': best_call['IV'],
            # Combined
            'credit': credit,
        })

    return candidates


def generate_selection_cache(tickers: list = None, verbose: bool = True):
    """
    Generate the selection cache for all available months.
    This is the slow step - run once and cache the results.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    tickers = tickers or get_pilot_tickers()
    db = OptionsDB()

    all_selections = []
    months_processed = 0

    for year, months in DATA_PERIODS.items():
        for month in months:
            if verbose:
                print(f"Processing {year}-{month:02d}...", end=" ", flush=True)

            try:
                # Load month data (filtered by tickers)
                options_df = db.query_month(year, month, tickers=tickers, minimal=True)

                if len(options_df) == 0:
                    if verbose:
                        print("no data")
                    continue

                # Get trading dates
                options_df['DataDate'] = pd.to_datetime(options_df['DataDate'])
                trading_dates = sorted(options_df['DataDate'].unique())

                month_selections = 0
                for date in trading_dates:
                    date_str = str(date.date())
                    for ticker in tickers:
                        candidates = find_strangle_candidates(options_df, ticker, date_str)
                        if candidates:
                            # Take the best candidate (highest credit)
                            best = max(candidates, key=lambda x: x['credit'])
                            all_selections.append(best)
                            month_selections += 1

                months_processed += 1
                if verbose:
                    print(f"{month_selections} selections")

            except Exception as e:
                if verbose:
                    print(f"error: {e}")

    # Save to parquet
    if all_selections:
        df = pd.DataFrame(all_selections)
        df.to_parquet(SELECTION_CACHE_FILE, index=False)

        # Save metadata
        metadata = {
            'generated_utc': datetime.utcnow().isoformat(),
            'tickers': tickers,
            'months_processed': months_processed,
            'total_selections': len(all_selections),
            'delta_target': STRATEGY_PARAMS['delta_target'],
            'dte_range': [STRATEGY_PARAMS['dte_min'], STRATEGY_PARAMS['dte_max']],
        }
        with open(CACHE_METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)

        if verbose:
            print(f"\nCache saved: {SELECTION_CACHE_FILE}")
            print(f"Total selections: {len(all_selections)}")
            print(f"Months processed: {months_processed}")

    return df if all_selections else pd.DataFrame()


def load_selection_cache() -> pd.DataFrame:
    """Load the pre-computed selection cache."""
    if not SELECTION_CACHE_FILE.exists():
        raise FileNotFoundError(
            f"Selection cache not found at {SELECTION_CACHE_FILE}\n"
            "Run: python -m src.backtest.selection_cache --generate"
        )
    return pd.read_parquet(SELECTION_CACHE_FILE)


def get_cache_metadata() -> dict:
    """Get metadata about the cache."""
    if not CACHE_METADATA_FILE.exists():
        return {}
    with open(CACHE_METADATA_FILE) as f:
        return json.load(f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Selection cache generator')
    parser.add_argument('--generate', action='store_true', help='Generate the cache')
    parser.add_argument('--info', action='store_true', help='Show cache info')
    args = parser.parse_args()

    if args.generate:
        print("Generating selection cache (this takes ~1 hour for full dataset)...")
        print("="*60)
        generate_selection_cache()
    elif args.info:
        meta = get_cache_metadata()
        if meta:
            print("Selection Cache Info:")
            print(f"  Generated: {meta.get('generated_utc')}")
            print(f"  Tickers: {meta.get('tickers')}")
            print(f"  Months: {meta.get('months_processed')}")
            print(f"  Selections: {meta.get('total_selections')}")
        else:
            print("No cache found. Run with --generate first.")
    else:
        parser.print_help()
