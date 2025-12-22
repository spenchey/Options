"""
Backtest Engine

Main backtesting harness that simulates the strategy over historical data.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime, timedelta
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from .config import STRATEGY_PARAMS, PILOT_UNIVERSE, DATA_PERIODS, get_pilot_tickers, get_expected_months
from .costs import CostModel, TradeCost
from .strategy import (
    StrategyBuilder, Strangle, OptionLeg,
    pick_strikes_by_target_delta, contracts_for_target_vega, vega_budget_allocator
)
from .risk import RiskManager, BetaCalculator, PortfolioRisk, PositionGreeks, hedge_shares

# Engine constants (from ChatGPT recommendations)
PORTFOLIO_VEGA_TARGET = STRATEGY_PARAMS.get('portfolio_vega_target', 10_000)
PER_NAME_CAP = STRATEGY_PARAMS.get('per_name_vega_cap', 0.08) * PORTFOLIO_VEGA_TARGET
TARGET_DTE_RANGE = (STRATEGY_PARAMS['dte_min'], STRATEGY_PARAMS['dte_max'])
TARGET_DELTA = STRATEGY_PARAMS['delta_target']
LADDERS_PER_NAME = STRATEGY_PARAMS.get('ladders_per_name', 2)
TP_PCT = STRATEGY_PARAMS['profit_take_pct']
SL_MULT = STRATEGY_PARAMS['loss_limit_mult']
REHEDGE_THRESHOLD_BP = STRATEGY_PARAMS.get('hedge_threshold_bp', 20)


@dataclass
class Position:
    """Active position in the portfolio."""
    strangle: Strangle
    entry_date: str
    entry_credit: float       # Total credit received
    contracts: int            # Number of strangles (negative for short)
    entry_cost: float         # Transaction costs on entry
    current_value: float      # Current mark-to-market value
    pnl: float                # Unrealized P&L
    status: str = 'open'      # 'open', 'closed_profit', 'closed_loss', 'closed_expiry'


@dataclass
class DailySnapshot:
    """Daily portfolio state."""
    date: str
    portfolio_value: float
    cash: float
    positions_value: float
    total_pnl: float
    daily_pnl: float
    beta_weighted_delta: float
    spy_hedge_shares: int
    hedge_value: float
    num_positions: int
    total_theta: float
    total_vega: float
    total_gamma: float = 0.0
    E_mkt_before_hedge: float = 0.0
    E_mkt_after_hedge: float = 0.0
    # P&L Attribution (for diagnostics)
    options_pnl: float = 0.0           # Daily options MTM P&L
    hedge_pnl: float = 0.0             # Daily SPY hedge P&L
    transaction_costs: float = 0.0     # Daily transaction costs
    gross_premium_outstanding: float = 0.0  # Total credit outstanding
    avg_dte: float = 0.0               # Average DTE of open positions


@dataclass
class TradeRecord:
    """Record of a single trade."""
    date: str
    ticker: str
    action: str              # 'open', 'close_profit', 'close_loss', 'close_expiry'
    contracts: int
    price: float             # Per-contract price
    cost: float              # Transaction cost
    pnl: float               # Realized P&L (for closes)


@dataclass
class BacktestResult:
    """Complete backtest results."""
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
    daily_snapshots: List[DailySnapshot]
    trades: List[TradeRecord]
    positions_summary: pd.DataFrame
    exposure_log: List[Dict] = field(default_factory=list)
    months_processed: int = 0
    months_expected: int = 0


class BacktestEngine:
    """
    Main backtesting engine for the short strangle strategy.

    Simulates:
    - Position entry based on strategy criteria
    - Daily mark-to-market
    - Position management (profit targets, stop losses)
    - Beta hedging with SPY
    - Transaction costs
    """

    def __init__(
        self,
        options_db,
        market_data,
        capital: float = STRATEGY_PARAMS['capital'],
        verbose: bool = True,
        costs_enabled: bool = True,
        stops_enabled: bool = True
    ):
        """
        Args:
            options_db: OptionsDB instance for querying options data
            market_data: MarketData instance for stock/SPX prices
            capital: Starting capital
            verbose: Print progress updates
            costs_enabled: If False, zero out all transaction costs (diagnostic toggle)
            stops_enabled: If False, disable stop-loss exits (diagnostic toggle)
        """
        self.db = options_db
        self.md = market_data
        self.initial_capital = capital
        self.verbose = verbose
        self.costs_enabled = costs_enabled
        self.stops_enabled = stops_enabled

        # Initialize components
        self.cost_model = CostModel()
        self.strategy = StrategyBuilder()
        self.beta_calc = BetaCalculator(market_data)
        self.risk_mgr = RiskManager(market_data, self.beta_calc)

        # Portfolio state
        self.cash = capital
        self.positions: Dict[str, Position] = {}
        self.spy_hedge_shares = 0
        self.spy_hedge_cost_basis = 0.0
        self.spy_prev_close = 0.0  # For daily hedge P&L

        # Exposure tracking
        self.exposure_log: List[Dict] = []

        # Daily P&L attribution tracking
        self._daily_options_pnl = 0.0
        self._daily_hedge_pnl = 0.0
        self._daily_tx_costs = 0.0

        # Results tracking
        self.daily_snapshots: List[DailySnapshot] = []
        self.trades: List[TradeRecord] = []
        self.closed_positions: List[Position] = []

    def log(self, msg: str):
        """Print if verbose mode enabled."""
        if self.verbose:
            print(msg)

    def run(
        self,
        tickers: List[str],
        start_year: int,
        start_month: int,
        end_year: int,
        end_month: int
    ) -> BacktestResult:
        """
        Run backtest over specified period.

        Note: Due to data availability, this will only process months
        where we have data (2011-01, 2012-01, 2013-11).

        Args:
            tickers: List of tickers to trade
            start_year, start_month: Start of backtest period
            end_year, end_month: End of backtest period

        Returns:
            BacktestResult with complete results
        """
        self.log(f"\n{'='*60}")
        self.log(f"Starting Backtest: Beta-Neutral Short Strangle Strategy")
        self.log(f"{'='*60}")
        self.log(f"Capital: ${self.initial_capital:,.0f}")
        self.log(f"Universe: {', '.join(tickers)}")
        self.log(f"Period: {start_year}-{start_month:02d} to {end_year}-{end_month:02d}")
        self.log(f"{'='*60}\n")

        # Get available months from DATA_PERIODS
        available_months = []
        for year, months in DATA_PERIODS.items():
            for month in months:
                if (year > start_year or (year == start_year and month >= start_month)):
                    if (year < end_year or (year == end_year and month <= end_month)):
                        available_months.append((year, month))

        self.log(f"Available data months: {available_months}\n")

        # Process each available month
        for year, month in available_months:
            self._process_month(year, month, tickers)

        # Generate results
        return self._generate_results()

    def _process_month(self, year: int, month: int, tickers: List[str]):
        """Process a single month of data."""
        self.log(f"\n--- Processing {year}-{month:02d} ---")

        # Load options data for this month (fast: filtered by tickers, minimal columns)
        try:
            options_df = self.db.query_month(year, month, tickers=tickers, minimal=True)
            self.log(f"Loaded {len(options_df):,} option records (filtered)")
        except Exception as e:
            self.log(f"Error loading data: {e}")
            return

        if len(options_df) == 0:
            self.log("No data for this month")
            return

        # Get unique trading dates
        options_df['DataDate'] = pd.to_datetime(options_df['DataDate'])
        trading_dates = sorted(options_df['DataDate'].unique())
        self.log(f"Trading dates: {len(trading_dates)}")

        # Process each trading day
        for date in trading_dates:
            date_str = str(date.date())
            day_options = options_df[options_df['DataDate'] == date]

            self._process_day(date_str, day_options, tickers)

    def _process_day(
        self,
        date: str,
        options_df: pd.DataFrame,
        tickers: List[str]
    ):
        """Process a single trading day."""
        # Reset daily P&L trackers
        self._daily_options_pnl = 0.0
        self._daily_hedge_pnl = 0.0
        self._daily_tx_costs = 0.0
        
        # 1. Mark existing positions to market
        self._mark_to_market(date, options_df)

        # 2. Check for exits (profit target, stop loss, expiry)
        self._check_exits(date, options_df)

        # 3. Look for new entry opportunities
        self._check_entries(date, options_df, tickers)

        # 4. Adjust beta hedge
        self._adjust_hedge(date)

        # 5. Record daily snapshot
        self._record_snapshot(date)

    def _mark_to_market(self, date: str, options_df: pd.DataFrame):
        """Update position values based on current prices."""
        for ticker, position in self.positions.items():
            if position.status != 'open':
                continue

            # Find current prices for the position's strikes
            put_price = self._get_option_price(
                options_df, ticker, 'put',
                position.strangle.put.strike,
                position.strangle.put.expiration
            )
            call_price = self._get_option_price(
                options_df, ticker, 'call',
                position.strangle.call.strike,
                position.strangle.call.expiration
            )

            if put_price is not None and call_price is not None:
                # Current value to close (what we'd pay to buy back)
                current_value = (put_price + call_price) * abs(position.contracts) * 100

                # P&L = credit received - current value to close
                position.current_value = current_value
                position.pnl = position.entry_credit - current_value

    def _get_option_price(
        self,
        options_df: pd.DataFrame,
        ticker: str,
        opt_type: str,
        strike: float,
        expiration: str
    ) -> Optional[float]:
        """Get mid price for a specific option."""
        # Convert expiration to datetime for comparison
        options_df = options_df.copy()
        if 'Expiration' in options_df.columns:
            options_df['Expiration'] = pd.to_datetime(options_df['Expiration'])

        exp_date = pd.to_datetime(expiration)

        mask = (
            (options_df['UnderlyingSymbol'] == ticker) &
            (options_df['Type'].str.lower() == opt_type) &
            (options_df['Strike'] == strike) &
            (options_df['Expiration'] == exp_date)
        )

        matches = options_df[mask]
        if len(matches) == 0:
            return None

        # Use first match, return mid price
        row = matches.iloc[0]
        mid = (row['Bid'] + row['Ask']) / 2

        # Sanity check: option price should not be absurdly high
        # Max reasonable per-contract price is about $50 (deeply ITM)
        if mid > 100:
            return None

        return mid

    def _check_exits(self, date: str, options_df: pd.DataFrame):
        """Check for position exits."""
        tickers_to_close = []

        for ticker, position in self.positions.items():
            if position.status != 'open':
                continue

            # Get current prices
            put_price = self._get_option_price(
                options_df, ticker, 'put',
                position.strangle.put.strike,
                position.strangle.put.expiration
            )
            call_price = self._get_option_price(
                options_df, ticker, 'call',
                position.strangle.call.strike,
                position.strangle.call.expiration
            )

            if put_price is None or call_price is None:
                continue

            current_premium = put_price + call_price
            original_credit = position.strangle.credit

            # Check profit target (50% of max profit)
            profit_pct = (original_credit - current_premium) / original_credit
            if profit_pct >= STRATEGY_PARAMS['profit_take_pct']:
                tickers_to_close.append((ticker, 'close_profit'))
                continue

            # Check stop loss (4x credit) - only if stops enabled
            if self.stops_enabled:
                if current_premium >= original_credit * (1 + STRATEGY_PARAMS['loss_limit_mult']):
                    tickers_to_close.append((ticker, 'close_loss'))
                    continue

            # Check approaching expiry (roll threshold)
            try:
                exp_date = datetime.strptime(position.strangle.put.expiration, '%Y-%m-%d')
                current_date = datetime.strptime(date, '%Y-%m-%d')
                dte = (exp_date - current_date).days
                if dte <= STRATEGY_PARAMS['roll_dte']:
                    tickers_to_close.append((ticker, 'close_expiry'))
            except:
                pass

        # Execute closes
        for ticker, reason in tickers_to_close:
            self._close_position(ticker, date, reason, options_df)

    def _close_position(
        self,
        ticker: str,
        date: str,
        reason: str,
        options_df: pd.DataFrame
    ):
        """Close a position."""
        position = self.positions.get(ticker)
        if not position:
            return

        # Get closing prices
        put_price = self._get_option_price(
            options_df, ticker, 'put',
            position.strangle.put.strike,
            position.strangle.put.expiration
        )
        call_price = self._get_option_price(
            options_df, ticker, 'call',
            position.strangle.call.strike,
            position.strangle.call.expiration
        )

        if put_price is None or call_price is None:
            # Can't get current prices - skip this close attempt
            # The position will be tried again on the next day
            self.log(f"  SKIP CLOSE {ticker}: prices not available")
            return

        close_cost = (put_price + call_price) * abs(position.contracts) * 100

        # CRITICAL FIX: Do NOT skip closes due to high cost
        # Per ChatGPT: "close at 4× and log it. Skipping hides the left tail."
        # We ALWAYS close regardless of cost - this is essential for accurate backtesting

        # Transaction costs (zero if costs disabled)
        if self.costs_enabled:
            tx_cost = self.cost_model.strangle_trade_cost(
                abs(position.contracts),  # Buying to close
                put_price * 0.98, put_price * 1.02,
                call_price * 0.98, call_price * 1.02,
                is_opening=False
            ).total
        else:
            tx_cost = 0.0

        # Track daily transaction costs for attribution
        self._daily_tx_costs += tx_cost

        # Options P&L (before costs) for attribution
        options_pnl = position.entry_credit - close_cost
        self._daily_options_pnl += options_pnl

        # Realized P&L
        realized_pnl = position.entry_credit - close_cost - tx_cost

        # Update cash
        self.cash -= close_cost  # Pay to close
        self.cash -= tx_cost

        # Record trade
        self.trades.append(TradeRecord(
            date=date,
            ticker=ticker,
            action=reason,
            contracts=abs(position.contracts),
            price=put_price + call_price,
            cost=tx_cost,
            pnl=realized_pnl
        ))

        # Update position status
        position.status = reason
        position.pnl = realized_pnl
        self.closed_positions.append(position)

        # Remove from active positions
        del self.positions[ticker]

        self.log(f"  CLOSE {ticker}: {reason}, P&L: ${realized_pnl:,.2f}")

    def _check_entries(
        self,
        date: str,
        options_df: pd.DataFrame,
        tickers: List[str]
    ):
        """Check for new entry opportunities."""
        # Only enter if we have capacity
        max_positions = len(tickers)
        if len(self.positions) >= max_positions * 0.8:
            return

        for ticker in tickers:
            # Skip if already have position
            if ticker in self.positions:
                continue

            # Find strangle candidates
            candidates = self.strategy.find_strangle_candidates(
                options_df, ticker, date
            )

            if not candidates:
                continue

            # Select best candidate
            best = self.strategy.select_best_strangle(candidates)
            if not best:
                continue

            # Size the position
            contracts = self.strategy.size_position(
                best,
                self.cash,
                max_position_pct=STRATEGY_PARAMS['max_position_pct']
            )

            if contracts < 1:
                continue

            # Calculate entry credit and costs
            entry_credit = best.credit * contracts * 100
            if self.costs_enabled:
                tx_cost = self.cost_model.strangle_trade_cost(
                    -contracts,
                    best.put.bid, best.put.ask,
                    best.call.bid, best.call.ask,
                    is_opening=True
                ).total
            else:
                tx_cost = 0.0
            
            # Track daily transaction costs
            self._daily_tx_costs += tx_cost

            # Update cash (receive credit, pay costs)
            self.cash += entry_credit
            self.cash -= tx_cost

            # Create position
            position = Position(
                strangle=best,
                entry_date=date,
                entry_credit=entry_credit,
                contracts=-contracts,  # Short
                entry_cost=tx_cost,
                current_value=entry_credit,
                pnl=0.0
            )

            self.positions[ticker] = position

            # Record trade
            self.trades.append(TradeRecord(
                date=date,
                ticker=ticker,
                action='open',
                contracts=contracts,
                price=best.credit,
                cost=tx_cost,
                pnl=0.0
            ))

            self.log(f"  OPEN {ticker}: {contracts} strangles, credit ${entry_credit:,.2f}")

    def _adjust_hedge(self, date: str):
        """Adjust SPY hedge to maintain beta neutrality."""
        # Get SPY price for this date
        try:
            spy_price = self.md.get_spy_price(date)
        except:
            spy_price = 0

        if spy_price <= 0:
            return 0.0  # Can't hedge without SPY price

        # Calculate total market exposure from positions
        E_mkt = 0.0
        for ticker, position in self.positions.items():
            if position.status != 'open':
                continue

            # Get beta for this ticker
            beta = self.beta_calc.get_beta(ticker, date)

            # Position delta (strangle has 2 legs)
            put_delta = position.strangle.put.delta
            call_delta = position.strangle.call.delta
            net_delta = (put_delta + call_delta) * abs(position.contracts) * 100

            # Dollar delta
            underlying_price = position.strangle.put.underlying_price
            dollar_delta = net_delta * underlying_price

            # Beta-weighted exposure
            E_mkt += dollar_delta * beta

        # Store pre-hedge exposure for logging
        self._current_E_mkt_before = E_mkt

        # Check if re-hedge needed (threshold = 0.20% of equity)
        portfolio_value = self.cash + sum(
            p.entry_credit - p.current_value for p in self.positions.values()
        )
        threshold = REHEDGE_THRESHOLD_BP * 1e-4 * portfolio_value

        # Account for existing hedge
        hedge_exposure = self.spy_hedge_shares * spy_price
        net_exposure = E_mkt + hedge_exposure

        self._current_E_mkt_after = net_exposure

        hedge_pnl = 0.0

        # First, mark existing hedge to market
        if self.spy_hedge_shares != 0 and self.spy_prev_close > 0:
            spy_move = spy_price - self.spy_prev_close
            hedge_pnl = self.spy_hedge_shares * spy_move
            # Track daily hedge P&L for attribution
            self._daily_hedge_pnl += hedge_pnl

        # Re-hedge if exposure exceeds threshold
        if abs(net_exposure) > threshold:
            target_shares = hedge_shares(E_mkt, spy_price)
            shares_to_trade = target_shares - self.spy_hedge_shares

            if shares_to_trade != 0:
                # Trade SPY
                if self.costs_enabled:
                    trade_cost = self.cost_model.spy_trade_cost(shares_to_trade)
                    tx_cost = trade_cost.total
                else:
                    tx_cost = 0.0
                trade_value = shares_to_trade * spy_price
                
                # Track daily transaction costs
                self._daily_tx_costs += tx_cost

                self.cash -= trade_value  # Buy costs money, sell returns money
                self.cash -= tx_cost

                # Update hedge position
                self.spy_hedge_shares = target_shares
                self.spy_hedge_cost_basis = target_shares * spy_price

                # Record trade
                self.trades.append(TradeRecord(
                    date=date,
                    ticker='SPY',
                    action='hedge',
                    contracts=shares_to_trade,
                    price=spy_price,
                    cost=trade_cost.total,
                    pnl=0.0
                ))

                self.log(f"  HEDGE: {shares_to_trade:+d} SPY @ ${spy_price:.2f}")

        # Update prev close for next day
        self.spy_prev_close = spy_price

        return hedge_pnl

    def _record_snapshot(self, date: str):
        """Record daily portfolio snapshot."""
        # Calculate total positions value
        positions_value = sum(p.current_value for p in self.positions.values())

        # Total P&L
        unrealized_pnl = sum(p.pnl for p in self.positions.values())
        realized_pnl = sum(t.pnl for t in self.trades if t.action not in ('open', 'hedge'))
        total_pnl = unrealized_pnl + realized_pnl

        # Portfolio value (including SPY hedge value)
        positions_pnl = sum(p.entry_credit - p.current_value for p in self.positions.values())
        spy_value = self.spy_hedge_shares * self.spy_prev_close if self.spy_prev_close > 0 else 0
        portfolio_value = self.cash + positions_pnl + spy_value

        # Daily P&L
        if self.daily_snapshots:
            daily_pnl = portfolio_value - self.daily_snapshots[-1].portfolio_value
        else:
            daily_pnl = portfolio_value - self.initial_capital

        # Greeks
        total_vega = sum(
            abs(p.strangle.put.vega) + abs(p.strangle.call.vega)
            for p in self.positions.values()
        ) * 100  # Per position vega
        total_theta = sum(
            p.strangle.put.theta + p.strangle.call.theta
            for p in self.positions.values()
        )
        total_gamma = sum(
            p.strangle.put.gamma + p.strangle.call.gamma
            for p in self.positions.values()
        )

        # Get exposure values (set by _adjust_hedge)
        E_mkt_before = getattr(self, '_current_E_mkt_before', 0.0)
        E_mkt_after = getattr(self, '_current_E_mkt_after', 0.0)

        # Calculate gross premium outstanding (total credit of open positions)
        gross_premium = sum(p.entry_credit for p in self.positions.values())
        
        # Calculate average DTE of open positions
        avg_dte = 0.0
        if self.positions:
            from datetime import datetime as dt
            dte_list = []
            for p in self.positions.values():
                try:
                    exp_date = dt.strptime(p.strangle.put.expiration, '%Y-%m-%d')
                    curr_date = dt.strptime(date, '%Y-%m-%d')
                    dte_list.append((exp_date - curr_date).days)
                except:
                    pass
            if dte_list:
                avg_dte = sum(dte_list) / len(dte_list)

        snapshot = DailySnapshot(
            date=date,
            portfolio_value=portfolio_value,
            cash=self.cash,
            positions_value=positions_value,
            total_pnl=total_pnl,
            daily_pnl=daily_pnl,
            beta_weighted_delta=E_mkt_before,
            spy_hedge_shares=self.spy_hedge_shares,
            hedge_value=spy_value,
            num_positions=len(self.positions),
            total_theta=total_theta,
            total_vega=total_vega,
            total_gamma=total_gamma,
            E_mkt_before_hedge=E_mkt_before,
            E_mkt_after_hedge=E_mkt_after,
            # P&L Attribution
            options_pnl=self._daily_options_pnl,
            hedge_pnl=self._daily_hedge_pnl,
            transaction_costs=self._daily_tx_costs,
            gross_premium_outstanding=gross_premium,
            avg_dte=avg_dte
        )

        self.daily_snapshots.append(snapshot)

        # Log exposure for export (includes P&L attribution)
        self.exposure_log.append({
            'date': date,
            'num_positions': len(self.positions),
            'total_vega': total_vega,
            'total_gamma': total_gamma,
            'total_theta': total_theta,
            'E_mkt_before': E_mkt_before,
            'E_mkt_after': E_mkt_after,
            'spy_hedge_shares': self.spy_hedge_shares,
            'portfolio_value': portfolio_value,
            # P&L Attribution
            'options_pnl': self._daily_options_pnl,
            'hedge_pnl': self._daily_hedge_pnl,
            'transaction_costs': self._daily_tx_costs,
            'gross_premium_outstanding': gross_premium,
            'avg_dte': avg_dte
        })

    def _generate_results(self) -> BacktestResult:
        """Generate final backtest results."""
        if not self.daily_snapshots:
            return BacktestResult(
                start_date='',
                end_date='',
                initial_capital=self.initial_capital,
                final_value=self.initial_capital,
                total_return=0.0,
                total_pnl=0.0,
                num_trades=0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                daily_snapshots=[],
                trades=[],
                positions_summary=pd.DataFrame(),
                months_processed=getattr(self, '_months_processed', 0),
                months_expected=getattr(self, '_months_expected', 0)
            )

        # Basic metrics
        start_date = self.daily_snapshots[0].date
        end_date = self.daily_snapshots[-1].date
        final_value = self.daily_snapshots[-1].portfolio_value
        total_return = (final_value - self.initial_capital) / self.initial_capital
        total_pnl = final_value - self.initial_capital

        # Trade statistics
        closed_trades = [t for t in self.trades if t.action != 'open']
        num_trades = len(closed_trades)

        if num_trades > 0:
            wins = [t for t in closed_trades if t.pnl > 0]
            losses = [t for t in closed_trades if t.pnl <= 0]
            win_rate = len(wins) / num_trades
            avg_win = np.mean([t.pnl for t in wins]) if wins else 0.0
            avg_loss = np.mean([t.pnl for t in losses]) if losses else 0.0
        else:
            win_rate = avg_win = avg_loss = 0.0

        # Drawdown
        values = [s.portfolio_value for s in self.daily_snapshots]
        peak = values[0]
        max_dd = 0.0
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            max_dd = max(max_dd, dd)

        # Sharpe (simplified - assumes daily returns)
        if len(self.daily_snapshots) > 1:
            daily_returns = [s.daily_pnl / self.initial_capital
                           for s in self.daily_snapshots]
            if np.std(daily_returns) > 0:
                sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        # Positions summary
        all_positions = self.closed_positions + list(self.positions.values())
        if all_positions:
            pos_data = [{
                'ticker': p.strangle.ticker,
                'entry_date': p.entry_date,
                'status': p.status,
                'contracts': abs(p.contracts),
                'entry_credit': p.entry_credit,
                'pnl': p.pnl
            } for p in all_positions]
            positions_df = pd.DataFrame(pos_data)
        else:
            positions_df = pd.DataFrame()

        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_value=final_value,
            total_return=total_return,
            total_pnl=total_pnl,
            num_trades=num_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            daily_snapshots=self.daily_snapshots,
            trades=self.trades,
            positions_summary=positions_df,
            exposure_log=self.exposure_log,
            months_processed=getattr(self, '_months_processed', 0),
            months_expected=getattr(self, '_months_expected', 0)
        )


def print_results(results: BacktestResult):
    """Print formatted backtest results."""
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    print(f"\nPeriod: {results.start_date} to {results.end_date}")
    print(f"Initial Capital: ${results.initial_capital:,.0f}")
    print(f"Final Value: ${results.final_value:,.0f}")

    print(f"\n--- Performance ---")
    print(f"Total Return: {results.total_return:.2%}")
    print(f"Total P&L: ${results.total_pnl:,.2f}")
    print(f"Max Drawdown: {results.max_drawdown:.2%}")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")

    print(f"\n--- Trading ---")
    print(f"Total Trades: {results.num_trades}")
    print(f"Win Rate: {results.win_rate:.1%}")
    print(f"Avg Win: ${results.avg_win:,.2f}")
    print(f"Avg Loss: ${results.avg_loss:,.2f}")

    if len(results.positions_summary) > 0:
        print(f"\n--- Positions ---")
        print(results.positions_summary.to_string(index=False))


def save_results(results: BacktestResult, output_dir: str = "results/baseline/strangle"):
    """
    Save backtest results to files.

    Creates:
        - daily_pnl.parquet: Daily P&L time series
        - summary.json: Summary metrics
        - exposure.csv: Daily exposure log

    Args:
        results: BacktestResult from backtest run
        output_dir: Directory to save files
    """
    from pathlib import Path

    # Create output directory
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Daily P&L to Parquet (includes P&L attribution)
    if results.daily_snapshots:
        daily_data = [{
            'date': s.date,
            'portfolio_value': s.portfolio_value,
            'daily_pnl': s.daily_pnl,
            'total_pnl': s.total_pnl,
            'num_positions': s.num_positions,
            'total_vega': s.total_vega,
            'total_theta': s.total_theta,
            'total_gamma': s.total_gamma,
            'spy_hedge_shares': s.spy_hedge_shares,
            'E_mkt_before_hedge': s.E_mkt_before_hedge,
            'E_mkt_after_hedge': s.E_mkt_after_hedge,
            # P&L Attribution
            'options_pnl': s.options_pnl,
            'hedge_pnl': s.hedge_pnl,
            'transaction_costs': s.transaction_costs,
            'gross_premium_outstanding': s.gross_premium_outstanding,
            'avg_dte': s.avg_dte
        } for s in results.daily_snapshots]
        daily_df = pd.DataFrame(daily_data)
        daily_df.to_parquet(out_path / "daily_pnl.parquet", index=False)
        print(f"Saved: {out_path / 'daily_pnl.parquet'}")

    # 2. Summary JSON
    summary = {
        'run_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'structure': 'strangle',
        'start_date': results.start_date,
        'end_date': results.end_date,
        'months_processed': results.months_processed,
        'months_expected': results.months_expected,
        'initial_capital': results.initial_capital,
        'final_value': results.final_value,
        'total_return': results.total_return,
        'total_pnl': results.total_pnl,
        'max_drawdown': results.max_drawdown,
        'sharpe_ratio': results.sharpe_ratio,
        'win_rate': results.win_rate,
        'total_trades': results.num_trades,
        'avg_win': results.avg_win,
        'avg_loss': results.avg_loss,
        'config': {
            'delta_target': TARGET_DELTA,
            'dte_range': list(TARGET_DTE_RANGE),
            'portfolio_vega_target': PORTFOLIO_VEGA_TARGET,
            'per_name_cap': PER_NAME_CAP,
            'ladders_per_name': LADDERS_PER_NAME,
            'profit_take': TP_PCT,
            'stop_loss_mult': SL_MULT,
            'rehedge_threshold_bp': REHEDGE_THRESHOLD_BP
        }
    }
    with open(out_path / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_path / 'summary.json'}")

    # 3. Exposure log to CSV
    if results.exposure_log:
        exposure_df = pd.DataFrame(results.exposure_log)
        exposure_df.to_csv(out_path / "exposure.csv", index=False)
        print(f"Saved: {out_path / 'exposure.csv'}")

    # 4. Trades to CSV
    if results.trades:
        trades_data = [{
            'date': t.date,
            'ticker': t.ticker,
            'action': t.action,
            'contracts': t.contracts,
            'price': t.price,
            'cost': t.cost,
            'pnl': t.pnl
        } for t in results.trades]
        trades_df = pd.DataFrame(trades_data)
        trades_df.to_csv(out_path / "trades.csv", index=False)
        print(f"Saved: {out_path / 'trades.csv'}")

    return out_path




def run_diagnostic_grid(options_db, market_data, tickers, start_year, start_month, end_year, end_month):
    """
    Run the 4-toggle diagnostic grid: Costs ON/OFF x Stops ON/OFF.
    
    This helps diagnose whether losses are due to:
    - Transaction costs / spreads
    - Stop-loss design / tail events
    - Both
    
    Returns:
        dict: Results for each configuration
    """
    configs = [
        {'costs_enabled': True, 'stops_enabled': True, 'name': 'baseline'},
        {'costs_enabled': False, 'stops_enabled': True, 'name': 'costs_off'},
        {'costs_enabled': True, 'stops_enabled': False, 'name': 'stops_off'},
        {'costs_enabled': False, 'stops_enabled': False, 'name': 'both_off'},
    ]
    
    results = {}
    
    for config in configs:
        print("")
        print("=" * 60)
        print(f"Running: {config['name']} (costs={config['costs_enabled']}, stops={config['stops_enabled']})")
        print("=" * 60)
        
        engine = BacktestEngine(
            options_db,
            market_data,
            costs_enabled=config['costs_enabled'],
            stops_enabled=config['stops_enabled'],
            verbose=False
        )
        
        result = engine.run(tickers, start_year, start_month, end_year, end_month)
        results[config['name']] = result
        
        print(f"Return: {result.total_return:.2%}, Sharpe: {result.sharpe_ratio:.2f}, Win Rate: {result.win_rate:.1%}")
    
    # Print summary table
    print("")
    print("=" * 60)
    print("DIAGNOSTIC GRID SUMMARY")
    print("=" * 60)
    print(f"{'Config':<15} {'Return':>10} {'Sharpe':>10} {'Win Rate':>10} {'Max DD':>10}")
    print("-" * 55)
    for name, r in results.items():
        print(f"{name:<15} {r.total_return:>10.2%} {r.sharpe_ratio:>10.2f} {r.win_rate:>10.1%} {r.max_drawdown:>10.2%}")
    
    return results


def analyze_worst_days(results, n=10):
    """
    Analyze the worst N days from a backtest result.
    
    Returns:
        DataFrame: Top N worst days with P&L attribution
    """
    if not results.daily_snapshots:
        return pd.DataFrame()
    
    data = [{
        'date': s.date,
        'daily_pnl': s.daily_pnl,
        'options_pnl': s.options_pnl,
        'hedge_pnl': s.hedge_pnl,
        'transaction_costs': s.transaction_costs,
        'num_positions': s.num_positions,
        'total_vega': s.total_vega
    } for s in results.daily_snapshots]
    
    df = pd.DataFrame(data)
    df['year_month'] = df['date'].str[:7]
    
    # Get worst days
    worst = df.nsmallest(n, 'daily_pnl')
    return worst


def analyze_worst_trades(results, n=10):
    """
    Analyze the worst N trades from a backtest result.
    
    Returns:
        DataFrame: Top N worst trades
    """
    if not results.trades:
        return pd.DataFrame()
    
    trades = [{
        'date': t.date,
        'ticker': t.ticker,
        'action': t.action,
        'pnl': t.pnl,
        'contracts': t.contracts,
        'price': t.price
    } for t in results.trades if t.action not in ('open', 'hedge')]
    
    df = pd.DataFrame(trades)
    df['year_month'] = df['date'].str[:7]
    
    # Get worst trades
    worst = df.nsmallest(n, 'pnl')
    return worst


if __name__ == '__main__':
    print("Backtest Engine Module")
    print("=" * 50)
    print("\nTo run a backtest:")
    print("  from src.options_db import OptionsDB")
    print("  from src.market_data import MarketData")
    print("  from src.backtest.engine import BacktestEngine, print_results")
    print("  from src.backtest.config import get_pilot_tickers")
    print("")
    print("  db = OptionsDB()")
    print("  md = MarketData()")
    print("  engine = BacktestEngine(db, md)")
    print("  results = engine.run(get_pilot_tickers(), 2011, 1, 2013, 12)")
    print("  print_results(results)")
