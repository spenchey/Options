"""
Fast Backtest Engine - Uses cached selection table for instant replays.

This engine reads from the pre-computed selection cache instead of S3.
Typical speedup: 100-1000x faster than the full engine.

Usage:
    from src.backtest.fast_engine import FastBacktestEngine, run_fast_diagnostic_grid

    # Run single backtest
    engine = FastBacktestEngine(costs_enabled=True, stops_enabled=True)
    result = engine.run(2002, 2, 2013, 10)

    # Run diagnostic grid
    results = run_fast_diagnostic_grid(2002, 2, 2013, 10)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import json

from .selection_cache import load_selection_cache, get_cache_metadata
from .config import STRATEGY_PARAMS, get_pilot_tickers
from .costs import CostModel


@dataclass
class FastPosition:
    """Position state for fast replay."""
    ticker: str
    date: str
    expiration: str
    entry_credit: float
    put_strike: float
    call_strike: float
    put_delta: float
    call_delta: float
    put_vega: float
    call_vega: float
    underlying_price: float
    contracts: int = 1
    status: str = 'open'
    pnl: float = 0.0


@dataclass
class FastResult:
    """Fast backtest results."""
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    total_return: float
    total_pnl: float
    num_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    sharpe_ratio: float
    # P&L attribution
    gross_options_pnl: float = 0.0
    total_costs: float = 0.0
    total_stop_losses: float = 0.0


class FastBacktestEngine:
    """
    Fast backtest engine using cached selection table.

    Instead of scanning S3 to find trades, reads from pre-computed cache.
    Focuses on replay logic: costs, stops, sizing.
    """

    def __init__(
        self,
        capital: float = STRATEGY_PARAMS['capital'],
        costs_enabled: bool = True,
        stops_enabled: bool = True,
        verbose: bool = False
    ):
        self.initial_capital = capital
        self.costs_enabled = costs_enabled
        self.stops_enabled = stops_enabled
        self.verbose = verbose

        self.cost_model = CostModel()

        # Load cache
        self.cache = load_selection_cache()
        self.cache['date'] = pd.to_datetime(self.cache['date'])

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    def run(
        self,
        start_year: int,
        start_month: int,
        end_year: int,
        end_month: int,
        tickers: List[str] = None
    ) -> FastResult:
        """Run fast backtest using cached selections."""
        tickers = tickers or get_pilot_tickers()

        # Filter cache to date range and tickers
        start_date = f"{start_year}-{start_month:02d}-01"
        end_date = f"{end_year}-{end_month:02d}-28"

        df = self.cache[
            (self.cache['date'] >= start_date) &
            (self.cache['date'] <= end_date) &
            (self.cache['ticker'].isin(tickers))
        ].copy()

        if len(df) == 0:
            return self._empty_result()

        # Sort by date
        df = df.sort_values('date')

        # Simulate
        return self._simulate(df)

    def _simulate(self, selections: pd.DataFrame) -> FastResult:
        """Run the simulation on cached selections."""
        cash = self.initial_capital
        positions: Dict[str, FastPosition] = {}
        trades = []
        daily_values = []

        # P&L attribution
        gross_options_pnl = 0.0
        total_costs = 0.0
        total_stop_losses = 0.0

        # Group by date
        for date, day_selections in selections.groupby('date'):
            date_str = str(date.date())

            # 1. Check for exits on existing positions
            tickers_to_close = []
            for ticker, pos in positions.items():
                # Check if we have new data for this position
                pos_data = day_selections[
                    (day_selections['ticker'] == ticker) &
                    (day_selections['expiration'] == pos.expiration)
                ]

                if len(pos_data) == 0:
                    # No price data for this expiration - check roll and expiry
                    days_to_exp = (pd.to_datetime(pos.expiration) - date).days

                    # Roll threshold (21 DTE) - estimate close at 80% profit
                    if days_to_exp <= STRATEGY_PARAMS['roll_dte']:
                        tickers_to_close.append((ticker, 'roll', pos.entry_credit * 0.2))
                        continue

                    # Check expiration
                    if date_str >= pos.expiration:
                        tickers_to_close.append((ticker, 'expiry', pos.entry_credit * 0.1))  # Assume 10% left
                    continue

                row = pos_data.iloc[0]
                current_mid = row['credit'] * pos.contracts * 100  # Convert to dollars

                # Profit target (55% of max profit)
                profit_pct = (pos.entry_credit - current_mid) / pos.entry_credit
                if profit_pct >= STRATEGY_PARAMS['profit_take_pct']:
                    tickers_to_close.append((ticker, 'profit', current_mid))
                    continue

                # Stop loss (4x credit)
                if self.stops_enabled:
                    if current_mid >= pos.entry_credit * (1 + STRATEGY_PARAMS['loss_limit_mult']):
                        tickers_to_close.append((ticker, 'stop', current_mid))
                        continue

                # Roll threshold (21 DTE)
                days_to_exp = (pd.to_datetime(pos.expiration) - date).days
                if days_to_exp <= STRATEGY_PARAMS['roll_dte']:
                    tickers_to_close.append((ticker, 'roll', current_mid))

            # Execute closes
            for ticker, reason, close_price in tickers_to_close:
                pos = positions[ticker]
                # close_price is already in dollars (entry_credit * factor or current_mid in dollars)
                close_cost = close_price

                # Transaction costs
                if self.costs_enabled:
                    # Convert back to per-share for cost model
                    close_price_per_share = close_cost / (pos.contracts * 100)
                    tx_cost = self.cost_model.strangle_trade_cost(
                        pos.contracts, close_price_per_share * 0.49, close_price_per_share * 0.51,
                        close_price_per_share * 0.49, close_price_per_share * 0.51, is_opening=False
                    ).total
                else:
                    tx_cost = 0.0

                # P&L
                options_pnl = pos.entry_credit - close_cost
                realized_pnl = options_pnl - tx_cost

                cash -= close_cost
                cash -= tx_cost

                gross_options_pnl += options_pnl
                total_costs += tx_cost
                if reason == 'stop':
                    total_stop_losses += abs(realized_pnl) if realized_pnl < 0 else 0

                trades.append({
                    'date': date_str,
                    'ticker': ticker,
                    'action': reason,
                    'pnl': realized_pnl
                })

                del positions[ticker]

            # 2. Open new positions
            for _, row in day_selections.iterrows():
                ticker = row['ticker']
                if ticker in positions:
                    continue

                # Size position (simplified: 1 contract per name)
                contracts = 1
                entry_credit = row['credit'] * contracts * 100

                # Transaction costs
                if self.costs_enabled:
                    tx_cost = self.cost_model.strangle_trade_cost(
                        -contracts, row['put_bid'], row['put_ask'],
                        row['call_bid'], row['call_ask'], is_opening=True
                    ).total
                else:
                    tx_cost = 0.0

                cash += entry_credit
                cash -= tx_cost
                total_costs += tx_cost

                positions[ticker] = FastPosition(
                    ticker=ticker,
                    date=date_str,
                    expiration=row['expiration'],
                    entry_credit=entry_credit,
                    put_strike=row['put_strike'],
                    call_strike=row['call_strike'],
                    put_delta=row['put_delta'],
                    call_delta=row['call_delta'],
                    put_vega=row['put_vega'],
                    call_vega=row['call_vega'],
                    underlying_price=row['underlying_price'],
                    contracts=contracts
                )

                trades.append({
                    'date': date_str,
                    'ticker': ticker,
                    'action': 'open',
                    'pnl': 0.0
                })

            # Track daily value
            positions_value = sum(p.entry_credit for p in positions.values())
            daily_values.append(cash + positions_value)

        # Calculate metrics
        final_value = daily_values[-1] if daily_values else self.initial_capital
        total_return = (final_value - self.initial_capital) / self.initial_capital
        total_pnl = final_value - self.initial_capital

        # Trade stats
        closed_trades = [t for t in trades if t['action'] != 'open']
        wins = [t for t in closed_trades if t['pnl'] > 0]
        losses = [t for t in closed_trades if t['pnl'] <= 0]

        win_rate = len(wins) / len(closed_trades) if closed_trades else 0.0
        avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0.0
        avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0.0

        # Drawdown
        peak = daily_values[0] if daily_values else self.initial_capital
        max_dd = 0.0
        for v in daily_values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            max_dd = max(max_dd, dd)

        # Sharpe
        if len(daily_values) > 1:
            returns = np.diff(daily_values) / self.initial_capital
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0.0
        else:
            sharpe = 0.0

        return FastResult(
            start_date=str(selections['date'].min().date()),
            end_date=str(selections['date'].max().date()),
            initial_capital=self.initial_capital,
            final_value=final_value,
            total_return=total_return,
            total_pnl=total_pnl,
            num_trades=len(closed_trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            gross_options_pnl=gross_options_pnl,
            total_costs=total_costs,
            total_stop_losses=total_stop_losses
        )

    def _empty_result(self) -> FastResult:
        return FastResult(
            start_date='', end_date='',
            initial_capital=self.initial_capital,
            final_value=self.initial_capital,
            total_return=0.0, total_pnl=0.0,
            num_trades=0, win_rate=0.0, avg_win=0.0, avg_loss=0.0,
            max_drawdown=0.0, sharpe_ratio=0.0
        )


def run_fast_diagnostic_grid(
    start_year: int, start_month: int,
    end_year: int, end_month: int,
    tickers: List[str] = None
) -> dict:
    """
    Run the 4-toggle diagnostic grid using fast engine.
    Typical runtime: seconds instead of hours.
    """
    configs = [
        {'costs_enabled': True, 'stops_enabled': True, 'name': 'baseline'},
        {'costs_enabled': False, 'stops_enabled': True, 'name': 'costs_off'},
        {'costs_enabled': True, 'stops_enabled': False, 'name': 'stops_off'},
        {'costs_enabled': False, 'stops_enabled': False, 'name': 'both_off'},
    ]

    results = {}

    for cfg in configs:
        engine = FastBacktestEngine(
            costs_enabled=cfg['costs_enabled'],
            stops_enabled=cfg['stops_enabled']
        )
        result = engine.run(start_year, start_month, end_year, end_month, tickers)
        results[cfg['name']] = result

        print(f"{cfg['name']:<12}: Return={result.total_return:>8.2%}, "
              f"Sharpe={result.sharpe_ratio:>6.2f}, Win={result.win_rate:>5.1%}, "
              f"Trades={result.num_trades:>4}")

    # Summary
    print("\n" + "="*60)
    print("P&L ATTRIBUTION")
    print("="*60)
    baseline = results['baseline']
    costs_off = results['costs_off']
    stops_off = results['stops_off']

    print(f"Gross Options P&L:  ${baseline.gross_options_pnl:>12,.2f}")
    print(f"Transaction Costs:  ${baseline.total_costs:>12,.2f}")
    print(f"Stop Loss Impact:   ${baseline.total_stop_losses:>12,.2f}")
    print(f"Net P&L:            ${baseline.total_pnl:>12,.2f}")
    print()
    print(f"Cost Impact:        {costs_off.total_return - baseline.total_return:>+.2%}")
    print(f"Stop Impact:        {stops_off.total_return - baseline.total_return:>+.2%}")

    return results


if __name__ == '__main__':
    import time
    print("Fast Backtest Engine Test")
    print("="*60)

    # Check cache
    try:
        meta = get_cache_metadata()
        print(f"Cache found: {meta.get('total_selections', 0)} selections")
    except:
        print("No cache found. Run: python -m src.backtest.selection_cache --generate")
        exit(1)

    print("\nRunning diagnostic grid...")
    start = time.time()
    results = run_fast_diagnostic_grid(2002, 2, 2013, 10)
    elapsed = time.time() - start

    print(f"\nTotal time: {elapsed:.1f} seconds")
