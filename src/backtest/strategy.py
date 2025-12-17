"""
Strategy Module

Strangle and iron condor position construction from options data.
"""

import pandas as pd
import numpy as np
import math
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from datetime import datetime, timedelta
from .config import STRATEGY_PARAMS


@dataclass
class OptionLeg:
    """Single option leg."""
    ticker: str
    option_type: str        # 'put' or 'call'
    strike: float
    expiration: str
    bid: float
    ask: float
    delta: float
    gamma: float
    vega: float
    theta: float
    iv: float
    underlying_price: float
    dte: int


@dataclass
class Strangle:
    """Short strangle position."""
    ticker: str
    put: OptionLeg
    call: OptionLeg
    data_date: str
    contracts: int          # Negative for short
    credit: float           # Per-contract credit at mid
    max_loss: float         # Theoretical max loss (unlimited for strangle)
    probability_profit: float  # Estimated POP


class StrategyBuilder:
    """
    Build strangle/iron condor positions from options data.
    """

    def __init__(
        self,
        dte_min: int = STRATEGY_PARAMS['dte_min'],
        dte_max: int = STRATEGY_PARAMS['dte_max'],
        delta_target: float = STRATEGY_PARAMS['delta_target'],
        delta_tolerance: float = STRATEGY_PARAMS['delta_tolerance']
    ):
        """
        Args:
            dte_min: Minimum days to expiration
            dte_max: Maximum days to expiration
            delta_target: Target delta for wings
            delta_tolerance: Acceptable delta range
        """
        self.dte_min = dte_min
        self.dte_max = dte_max
        self.delta_target = delta_target
        self.delta_tolerance = delta_tolerance

    def find_strangle_candidates(
        self,
        options_df: pd.DataFrame,
        ticker: str,
        data_date: str
    ) -> List[Strangle]:
        """
        Find all valid strangle candidates for a ticker on a given date.

        Args:
            options_df: DataFrame with options data
            ticker: Stock symbol to filter
            data_date: Date string (YYYY-MM-DD)

        Returns:
            List of Strangle objects meeting criteria
        """
        # Filter for ticker
        df = options_df[options_df['UnderlyingSymbol'] == ticker].copy()
        if len(df) == 0:
            return []

        # Parse dates and calculate DTE
        df['DataDate'] = pd.to_datetime(df['DataDate'])
        df['Expiration'] = pd.to_datetime(df['Expiration'])
        df['DTE'] = (df['Expiration'] - df['DataDate']).dt.days

        # Filter by DTE
        df = df[(df['DTE'] >= self.dte_min) & (df['DTE'] <= self.dte_max)]
        if len(df) == 0:
            return []

        # Separate puts and calls
        puts = df[df['Type'].str.lower() == 'put'].copy()
        calls = df[df['Type'].str.lower() == 'call'].copy()

        # Filter by delta (absolute value near target)
        delta_min = self.delta_target - self.delta_tolerance
        delta_max = self.delta_target + self.delta_tolerance

        puts = puts[(puts['Delta'].abs() >= delta_min) &
                    (puts['Delta'].abs() <= delta_max)]
        calls = calls[(calls['Delta'].abs() >= delta_min) &
                      (calls['Delta'].abs() <= delta_max)]

        if len(puts) == 0 or len(calls) == 0:
            return []

        # Find matching pairs (same expiration)
        strangles = []
        for exp in puts['Expiration'].unique():
            exp_puts = puts[puts['Expiration'] == exp]
            exp_calls = calls[calls['Expiration'] == exp]

            if len(exp_puts) == 0 or len(exp_calls) == 0:
                continue

            # Find best put (closest to target delta)
            exp_puts = exp_puts.copy()
            exp_puts['delta_diff'] = (exp_puts['Delta'].abs() - self.delta_target).abs()
            best_put = exp_puts.loc[exp_puts['delta_diff'].idxmin()]

            # Find best call (closest to target delta)
            exp_calls = exp_calls.copy()
            exp_calls['delta_diff'] = (exp_calls['Delta'].abs() - self.delta_target).abs()
            best_call = exp_calls.loc[exp_calls['delta_diff'].idxmin()]

            # Build option legs
            put_leg = OptionLeg(
                ticker=ticker,
                option_type='put',
                strike=best_put['Strike'],
                expiration=str(exp.date()),
                bid=best_put['Bid'],
                ask=best_put['Ask'],
                delta=best_put['Delta'],
                gamma=best_put.get('Gamma', 0),
                vega=best_put.get('Vega', 0),
                theta=best_put.get('Theta', 0),
                iv=best_put['IV'],
                underlying_price=best_put['UnderlyingPrice'],
                dte=best_put['DTE']
            )

            call_leg = OptionLeg(
                ticker=ticker,
                option_type='call',
                strike=best_call['Strike'],
                expiration=str(exp.date()),
                bid=best_call['Bid'],
                ask=best_call['Ask'],
                delta=best_call['Delta'],
                gamma=best_call.get('Gamma', 0),
                vega=best_call.get('Vega', 0),
                theta=best_call.get('Theta', 0),
                iv=best_call['IV'],
                underlying_price=best_call['UnderlyingPrice'],
                dte=best_call['DTE']
            )

            # Calculate credit at mid
            put_mid = (best_put['Bid'] + best_put['Ask']) / 2
            call_mid = (best_call['Bid'] + best_call['Ask']) / 2
            credit = put_mid + call_mid

            # Estimate probability of profit (simplified)
            # For a strangle, POP ~ 1 - (2 * delta) when balanced
            avg_delta = (abs(best_put['Delta']) + abs(best_call['Delta'])) / 2
            pop = 1 - 2 * avg_delta

            strangle = Strangle(
                ticker=ticker,
                put=put_leg,
                call=call_leg,
                data_date=data_date,
                contracts=-1,  # Short by default
                credit=credit,
                max_loss=float('inf'),  # Unlimited for naked strangle
                probability_profit=pop
            )

            strangles.append(strangle)

        return strangles

    def select_best_strangle(
        self,
        candidates: List[Strangle],
        prefer_dte: Optional[int] = None
    ) -> Optional[Strangle]:
        """
        Select the best strangle from candidates.

        Selection criteria:
        1. Prefer DTE closest to target (if specified)
        2. Higher credit per unit of delta exposure
        3. Better liquidity (tighter spreads)

        Args:
            candidates: List of Strangle candidates
            prefer_dte: Preferred DTE (defaults to STRATEGY_PARAMS['dte_target'])

        Returns:
            Best Strangle or None if no candidates
        """
        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        target_dte = prefer_dte or STRATEGY_PARAMS['dte_target']

        # Score each candidate
        scored = []
        for s in candidates:
            # DTE score (higher is better, closer to target)
            dte_diff = abs(s.put.dte - target_dte)
            dte_score = 1 / (1 + dte_diff)

            # Credit score (higher credit is better)
            credit_score = s.credit

            # Liquidity score (tighter spread is better)
            put_spread = s.put.ask - s.put.bid
            call_spread = s.call.ask - s.call.bid
            avg_spread = (put_spread + call_spread) / 2
            spread_score = 1 / (1 + avg_spread * 10)

            # Combined score
            total_score = dte_score * 0.4 + credit_score * 0.4 + spread_score * 0.2

            scored.append((total_score, s))

        # Return highest scored
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def size_position(
        self,
        strangle: Strangle,
        portfolio_capital: float,
        max_position_pct: float = STRATEGY_PARAMS['max_position_pct'],
        max_risk_per_position: Optional[float] = None
    ) -> int:
        """
        Determine position size in contracts.

        Args:
            strangle: Strangle to size
            portfolio_capital: Total portfolio capital
            max_position_pct: Max % of capital in one position
            max_risk_per_position: Max $ risk per position

        Returns:
            Number of contracts to trade
        """
        # Max capital allocation
        max_allocation = portfolio_capital * max_position_pct

        # Notional per contract (using underlying price)
        notional_per_contract = strangle.put.underlying_price * 100

        # Max contracts by capital
        max_by_capital = int(max_allocation / notional_per_contract)

        # If max risk specified, also limit by that
        if max_risk_per_position:
            # For strangle, risk is theoretically unlimited
            # Use a proxy: 2x the credit received as max loss assumption
            assumed_max_loss = strangle.credit * 100 * 2
            max_by_risk = int(max_risk_per_position / assumed_max_loss)
            return min(max_by_capital, max_by_risk, 10)  # Cap at 10 for now

        return min(max_by_capital, 10)  # Cap at 10 contracts per position

    def check_profit_target(
        self,
        strangle: Strangle,
        current_put_mid: float,
        current_call_mid: float,
        profit_target: float = STRATEGY_PARAMS['profit_take_pct']
    ) -> bool:
        """
        Check if position has reached profit target.

        Args:
            strangle: Original strangle position
            current_put_mid: Current put mid price
            current_call_mid: Current call mid price
            profit_target: Fraction of max profit to target

        Returns:
            True if profit target reached
        """
        original_credit = strangle.credit
        current_value = current_put_mid + current_call_mid
        profit_pct = (original_credit - current_value) / original_credit

        return profit_pct >= profit_target

    def check_loss_limit(
        self,
        strangle: Strangle,
        current_put_mid: float,
        current_call_mid: float,
        loss_limit: float = STRATEGY_PARAMS['loss_limit_mult']
    ) -> bool:
        """
        Check if position has breached loss limit.

        Args:
            strangle: Original strangle position
            current_put_mid: Current put mid price
            current_call_mid: Current call mid price
            loss_limit: Multiple of credit for stop loss

        Returns:
            True if loss limit breached
        """
        original_credit = strangle.credit
        current_value = current_put_mid + current_call_mid
        loss = current_value - original_credit

        return loss >= original_credit * loss_limit


# =============================================================================
# Vega-targeted sizing functions (ChatGPT implementation)
# =============================================================================

def pick_strikes_by_target_delta(
    chain_df: pd.DataFrame,
    target_delta: float = 0.20,
    right_put: str = 'put',
    right_call: str = 'call'
) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    """
    Select put and call strikes closest to target delta.

    Args:
        chain_df: DataFrame with options for one underlying & expiry.
                  Must have columns: 'Type', 'Delta', 'Strike', etc.
        target_delta: Target absolute delta (e.g., 0.20 for 20-delta)
        right_put: Value for put type in 'Type' column
        right_call: Value for call type in 'Type' column

    Returns:
        (put_row, call_row) - Series for each leg, or None if not found
    """
    # Normalize type column
    chain_df = chain_df.copy()
    chain_df['_type_lower'] = chain_df['Type'].str.lower()

    puts = chain_df[chain_df['_type_lower'] == right_put.lower()].copy()
    calls = chain_df[chain_df['_type_lower'] == right_call.lower()].copy()

    if len(puts) == 0 or len(calls) == 0:
        return None, None

    # Put delta is negative, so find closest to -target_delta
    puts['_delta_diff'] = (puts['Delta'] + target_delta).abs()
    best_put_idx = puts['_delta_diff'].idxmin()
    put_row = puts.loc[best_put_idx]

    # Call delta is positive, so find closest to +target_delta
    calls['_delta_diff'] = (calls['Delta'] - target_delta).abs()
    best_call_idx = calls['_delta_diff'].idxmin()
    call_row = calls.loc[best_call_idx]

    return put_row, call_row


def contracts_for_target_vega(
    contract_vega: float,
    name_target_vega: float,
    min_contracts: int = 1
) -> int:
    """
    Calculate number of contracts to reach target vega exposure.

    Args:
        contract_vega: Vega per contract (absolute value)
        name_target_vega: Target vega for this name
        min_contracts: Minimum contracts to open

    Returns:
        Number of contracts
    """
    if pd.isna(contract_vega) or contract_vega == 0:
        return 0
    n = math.floor(abs(name_target_vega) / abs(contract_vega))
    return max(n, min_contracts)


def vega_budget_allocator(
    eligible_names: List[str],
    per_name_cap: float,
    portfolio_vega_target: float
) -> Dict[str, float]:
    """
    Allocate vega budget across names, capped so total <= portfolio target.

    Args:
        eligible_names: List of ticker symbols
        per_name_cap: Max vega per name
        portfolio_vega_target: Total portfolio vega budget

    Returns:
        Dict {ticker: allocated_vega}
    """
    names = list(eligible_names)
    if not names:
        return {}

    # If equal allocation would exceed target, scale down
    raw_total = per_name_cap * len(names)
    scale = min(1.0, portfolio_vega_target / max(1, raw_total))

    return {n: per_name_cap * scale for n in names}


def iv_scaled_vega_target(
    base_vega: float,
    ivp: float
) -> float:
    """
    Scale vega target based on IV percentile (v2 enhancement).

    Args:
        base_vega: Base vega allocation for the name
        ivp: IV percentile (0-100)

    Returns:
        Scaled vega target
    """
    if ivp < 20:
        return base_vega * 0.5   # Low IV: reduce exposure
    elif ivp > 60:
        return base_vega * 1.5   # High IV: increase exposure
    else:
        return base_vega         # Normal: no scaling


def analyze_strangle_opportunities(
    options_df: pd.DataFrame,
    tickers: List[str]
) -> pd.DataFrame:
    """
    Analyze strangle opportunities across multiple tickers.

    Args:
        options_df: Full options DataFrame
        tickers: List of tickers to analyze

    Returns:
        DataFrame with opportunity summary
    """
    builder = StrategyBuilder()
    results = []

    for ticker in tickers:
        # Get unique dates for this ticker
        ticker_df = options_df[options_df['UnderlyingSymbol'] == ticker]
        if len(ticker_df) == 0:
            continue

        dates = ticker_df['DataDate'].unique()

        for date in dates:
            candidates = builder.find_strangle_candidates(
                options_df,
                ticker,
                str(date)
            )

            if candidates:
                best = builder.select_best_strangle(candidates)
                if best:
                    results.append({
                        'ticker': ticker,
                        'date': str(date),
                        'put_strike': best.put.strike,
                        'call_strike': best.call.strike,
                        'expiration': best.put.expiration,
                        'dte': best.put.dte,
                        'credit': best.credit,
                        'put_delta': best.put.delta,
                        'call_delta': best.call.delta,
                        'avg_iv': (best.put.iv + best.call.iv) / 2,
                        'underlying_price': best.put.underlying_price,
                        'credit_pct': best.credit / best.put.underlying_price * 100,
                        'pop': best.probability_profit
                    })

    return pd.DataFrame(results)


if __name__ == '__main__':
    print("Strategy Module Test")
    print("=" * 50)

    # Test with sample data
    builder = StrategyBuilder()

    print(f"\nStrategy Parameters:")
    print(f"  DTE range: {builder.dte_min}-{builder.dte_max} days")
    print(f"  Delta target: {builder.delta_target} +/- {builder.delta_tolerance}")
    print(f"  Profit target: {STRATEGY_PARAMS['profit_take_pct']:.0%}")
    print(f"  Loss limit: {STRATEGY_PARAMS['loss_limit_mult']}x credit")

    print("\nTo test with real data:")
    print("  from src.options_db import OptionsDB")
    print("  from src.backtest.strategy import StrategyBuilder, analyze_strangle_opportunities")
    print("  db = OptionsDB()")
    print("  df = db.query_month(2012, 1)")
    print("  results = analyze_strangle_opportunities(df, ['AAPL', 'MSFT'])")
