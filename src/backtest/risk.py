"""
Risk Management Module

Beta calculations, position sizing, and delta hedging logic.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .config import STRATEGY_PARAMS, RISK_LIMITS


@dataclass
class PositionGreeks:
    """Greeks for a single position."""
    ticker: str
    delta: float           # Position delta in shares
    gamma: float           # Position gamma
    vega: float            # Position vega ($ per 1% IV move)
    theta: float           # Position theta ($ per day)
    contracts: int         # Number of option contracts
    notional: float        # Notional exposure


@dataclass
class PortfolioRisk:
    """Aggregate portfolio risk metrics."""
    total_delta: float             # Sum of position deltas ($ terms)
    beta_weighted_delta: float     # BWD in SPY-equivalent $
    total_vega: float              # Portfolio vega
    total_theta: float             # Portfolio theta
    total_gamma: float             # Portfolio gamma
    spy_hedge_shares: int          # SPY shares needed to hedge
    positions: Dict[str, PositionGreeks]


class BetaCalculator:
    """Calculate rolling betas with shrinkage."""

    def __init__(self, market_data, lookback: int = 60, shrinkage: float = 0.3):
        """
        Args:
            market_data: MarketData instance for stock/SPX prices
            lookback: Days for rolling beta calculation
            shrinkage: Fraction to shrink toward 1.0 (0 = no shrinkage)
        """
        self.md = market_data
        self.lookback = lookback
        self.shrinkage = shrinkage
        self._beta_cache = {}

    def get_beta(self, ticker: str, date: str) -> float:
        """
        Get beta for ticker as of date, with shrinkage toward 1.0.

        Args:
            ticker: Stock ticker
            date: Date string (YYYY-MM-DD)

        Returns:
            Shrunk beta estimate
        """
        cache_key = (ticker, date, self.lookback)
        if cache_key in self._beta_cache:
            return self._beta_cache[cache_key]

        # Get raw OLS beta from market_data module
        raw_beta = self.md.get_beta(ticker, date, lookback_days=self.lookback)

        # Apply shrinkage toward 1.0
        # shrunk_beta = (1 - shrinkage) * raw_beta + shrinkage * 1.0
        shrunk_beta = (1 - self.shrinkage) * raw_beta + self.shrinkage * 1.0

        self._beta_cache[cache_key] = shrunk_beta
        return shrunk_beta

    def get_betas(self, tickers: List[str], date: str) -> Dict[str, float]:
        """Get betas for multiple tickers."""
        return {ticker: self.get_beta(ticker, date) for ticker in tickers}


class RiskManager:
    """Portfolio risk calculation and hedging."""

    def __init__(self, market_data, beta_calculator: Optional[BetaCalculator] = None):
        """
        Args:
            market_data: MarketData instance
            beta_calculator: Optional BetaCalculator (creates one if not provided)
        """
        self.md = market_data
        self.beta_calc = beta_calculator or BetaCalculator(
            market_data,
            lookback=STRATEGY_PARAMS['beta_lookback'],
            shrinkage=STRATEGY_PARAMS['beta_shrinkage']
        )

    def calculate_position_greeks(
        self,
        ticker: str,
        contracts: int,
        option_delta: float,
        option_gamma: float,
        option_vega: float,
        option_theta: float,
        underlying_price: float,
        multiplier: int = 100
    ) -> PositionGreeks:
        """
        Calculate position-level greeks.

        Args:
            ticker: Stock symbol
            contracts: Number of contracts (negative for short)
            option_delta: Per-contract delta
            option_gamma: Per-contract gamma
            option_vega: Per-contract vega
            option_theta: Per-contract theta
            underlying_price: Current stock price
            multiplier: Contract multiplier (usually 100)

        Returns:
            PositionGreeks dataclass
        """
        position_delta = contracts * option_delta * multiplier
        position_gamma = contracts * option_gamma * multiplier
        position_vega = contracts * option_vega * multiplier
        position_theta = contracts * option_theta * multiplier
        notional = abs(contracts) * underlying_price * multiplier

        return PositionGreeks(
            ticker=ticker,
            delta=position_delta,
            gamma=position_gamma,
            vega=position_vega,
            theta=position_theta,
            contracts=contracts,
            notional=notional
        )

    def calculate_strangle_greeks(
        self,
        ticker: str,
        contracts: int,
        put_greeks: dict,
        call_greeks: dict,
        underlying_price: float
    ) -> PositionGreeks:
        """
        Calculate combined greeks for a short strangle.

        Args:
            ticker: Stock symbol
            contracts: Number of strangles (negative for short)
            put_greeks: Dict with 'delta', 'gamma', 'vega', 'theta'
            call_greeks: Dict with 'delta', 'gamma', 'vega', 'theta'
            underlying_price: Current stock price

        Returns:
            PositionGreeks for the combined strangle
        """
        multiplier = 100

        # Combine put + call greeks (multiply by contracts)
        # Note: short strangle has negative contracts
        combined_delta = contracts * (put_greeks['delta'] + call_greeks['delta']) * multiplier
        combined_gamma = contracts * (put_greeks['gamma'] + call_greeks['gamma']) * multiplier
        combined_vega = contracts * (put_greeks['vega'] + call_greeks['vega']) * multiplier
        combined_theta = contracts * (put_greeks['theta'] + call_greeks['theta']) * multiplier

        # Notional is 2x contracts since strangle has 2 legs
        notional = abs(contracts) * 2 * underlying_price * multiplier

        return PositionGreeks(
            ticker=ticker,
            delta=combined_delta,
            gamma=combined_gamma,
            vega=combined_vega,
            theta=combined_theta,
            contracts=contracts * 2,  # 2 legs per strangle
            notional=notional
        )

    def calculate_portfolio_risk(
        self,
        positions: Dict[str, PositionGreeks],
        date: str,
        spy_price: float
    ) -> PortfolioRisk:
        """
        Calculate aggregate portfolio risk and hedge requirements.

        Args:
            positions: Dict of ticker -> PositionGreeks
            date: Current date for beta lookup
            spy_price: Current SPY price

        Returns:
            PortfolioRisk with aggregate metrics and hedge size
        """
        total_delta = 0.0
        beta_weighted_delta = 0.0
        total_vega = 0.0
        total_theta = 0.0
        total_gamma = 0.0

        for ticker, greeks in positions.items():
            # Get stock price for $ delta
            # Note: Delta is already in shares, convert to $ delta
            stock_price = greeks.notional / (abs(greeks.contracts) * 50) if greeks.contracts != 0 else 0

            # Dollar delta = share delta * stock price
            dollar_delta = greeks.delta * stock_price if stock_price > 0 else greeks.delta

            # Get beta for this ticker
            beta = self.beta_calc.get_beta(ticker, date)

            # Beta-weighted delta in SPY terms
            bwd = dollar_delta * beta

            total_delta += dollar_delta
            beta_weighted_delta += bwd
            total_vega += greeks.vega
            total_theta += greeks.theta
            total_gamma += greeks.gamma

        # Calculate SPY shares needed to hedge BWD
        # Negative BWD means we're short delta, need to buy SPY
        spy_hedge_shares = -int(beta_weighted_delta / spy_price)

        return PortfolioRisk(
            total_delta=total_delta,
            beta_weighted_delta=beta_weighted_delta,
            total_vega=total_vega,
            total_theta=total_theta,
            total_gamma=total_gamma,
            spy_hedge_shares=spy_hedge_shares,
            positions=positions
        )

    def check_hedge_needed(
        self,
        current_bwd: float,
        target_bwd: float,
        portfolio_notional: float
    ) -> Tuple[bool, float]:
        """
        Check if rehedging is needed based on BWD drift.

        Args:
            current_bwd: Current beta-weighted delta ($)
            target_bwd: Target BWD (usually 0 for delta-neutral)
            portfolio_notional: Total portfolio notional for threshold calc

        Returns:
            (needs_hedge, adjustment_amount)
        """
        threshold = portfolio_notional * STRATEGY_PARAMS['hedge_threshold']
        drift = current_bwd - target_bwd

        if abs(drift) > threshold:
            return True, -drift  # Return amount to adjust
        return False, 0.0

    def check_risk_limits(
        self,
        portfolio_risk: PortfolioRisk,
        portfolio_capital: float
    ) -> Dict[str, bool]:
        """
        Check if portfolio is within risk limits.

        Args:
            portfolio_risk: Current portfolio risk metrics
            portfolio_capital: Total portfolio capital

        Returns:
            Dict of limit_name -> is_within_limit
        """
        checks = {}

        # Delta limit (as % of capital)
        delta_pct = abs(portfolio_risk.beta_weighted_delta) / portfolio_capital
        checks['delta_limit'] = delta_pct <= RISK_LIMITS['max_portfolio_delta']

        # Vega limit (as % of capital)
        vega_pct = abs(portfolio_risk.total_vega) / portfolio_capital
        checks['vega_limit'] = vega_pct <= RISK_LIMITS['max_portfolio_vega']

        # Concentration limit (any single position)
        max_concentration = 0.0
        total_notional = sum(p.notional for p in portfolio_risk.positions.values())
        if total_notional > 0:
            for pos in portfolio_risk.positions.values():
                concentration = pos.notional / total_notional
                max_concentration = max(max_concentration, concentration)
        checks['concentration_limit'] = max_concentration <= RISK_LIMITS['concentration_limit']

        return checks


def calculate_spy_equivalent_shares(
    ticker_delta: float,
    ticker_beta: float,
    ticker_price: float,
    spy_price: float
) -> int:
    """
    Calculate SPY-equivalent shares for a position's delta.

    Args:
        ticker_delta: Position delta in shares
        ticker_beta: Stock's beta to SPY
        ticker_price: Current stock price
        spy_price: Current SPY price

    Returns:
        Number of SPY shares equivalent to this delta exposure
    """
    # Dollar delta of position
    dollar_delta = ticker_delta * ticker_price

    # Beta-weighted dollar delta
    bwd = dollar_delta * ticker_beta

    # SPY shares equivalent
    return int(bwd / spy_price)


if __name__ == '__main__':
    # Test the risk module
    print("Risk Module Test")
    print("=" * 50)

    # Create mock greeks for a short strangle
    put_greeks = {'delta': -0.16, 'gamma': 0.02, 'vega': 0.25, 'theta': -0.05}
    call_greeks = {'delta': 0.16, 'gamma': 0.02, 'vega': 0.25, 'theta': -0.05}

    # Simulate position (10 short strangles on AAPL at $500)
    print("\nExample: 10 short strangles on AAPL @ $500")
    print(f"  Put delta: {put_greeks['delta']}")
    print(f"  Call delta: {call_greeks['delta']}")
    print(f"  Combined delta per strangle: {put_greeks['delta'] + call_greeks['delta']}")

    # Net delta of strangle
    net_delta_per_strangle = (put_greeks['delta'] + call_greeks['delta']) * 100
    total_net_delta = -10 * net_delta_per_strangle  # 10 short strangles

    print(f"\n  Position delta (10 short): {total_net_delta} shares")

    # With beta = 1.2
    beta = 1.2
    stock_price = 500
    spy_price = 140
    dollar_delta = total_net_delta * stock_price
    bwd = dollar_delta * beta
    spy_equiv = int(bwd / spy_price)

    print(f"  Dollar delta: ${dollar_delta:,.0f}")
    print(f"  Beta: {beta}")
    print(f"  Beta-weighted delta: ${bwd:,.0f}")
    print(f"  SPY equivalent shares: {spy_equiv}")
