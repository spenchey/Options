"""
Backtest Configuration

Pilot universe and strategy parameters for the baseline backtest engine.
Adapted from ChatGPT Step 1 plan for 2011-2013 available data.
"""

# Pilot Universe - 10 liquid single-name equities across sectors
# Selected based on options volume analysis from 2011-2013 data
PILOT_UNIVERSE = {
    # Tech (3 names)
    'AAPL': {'sector': 'Technology', 'avg_price': 435.38},
    'MSFT': {'sector': 'Technology', 'avg_price': 31.44},
    'INTC': {'sector': 'Technology', 'avg_price': 24.12},

    # Financials (2 names)
    'BAC': {'sector': 'Financials', 'avg_price': 11.46},
    'JPM': {'sector': 'Financials', 'avg_price': 43.43},

    # Telecom (1 name)
    'T': {'sector': 'Telecom', 'avg_price': 31.60},

    # Industrial (1 name)
    'GE': {'sector': 'Industrials', 'avg_price': 21.36},

    # Healthcare (1 name)
    'PFE': {'sector': 'Healthcare', 'avg_price': 24.31},

    # Auto/Consumer Discretionary (1 name)
    'F': {'sector': 'Consumer Discretionary', 'avg_price': 15.46},

    # Consumer Staples (1 name)
    'PG': {'sector': 'Consumer Staples', 'avg_price': 72.29},
}

# Data availability constraints
DATA_PERIODS = {
    2011: [1],   # January only
    2012: [1],   # January only
    2013: [11],  # November only
}

# Strategy Parameters (from ChatGPT plan)
STRATEGY_PARAMS = {
    # Structure selection
    'structure': 'strangle',      # 'strangle' or 'iron_condor'

    # DTE targeting
    'dte_target': 45,             # Target days to expiration
    'dte_min': 30,                # Minimum DTE to open position
    'dte_max': 60,                # Maximum DTE to open position

    # Delta targeting for short strikes
    'delta_target': 0.16,         # ~16 delta wings (1 std dev)
    'delta_tolerance': 0.05,      # +/- 5 delta acceptable range

    # Position management
    'profit_take_pct': 0.50,      # Close at 50% of max profit
    'loss_limit_mult': 2.0,       # Close if loss > 2x credit received
    'roll_dte': 21,               # Roll when DTE drops below this

    # Beta hedging
    'hedge_instrument': 'SPY',    # Use SPY for hedging (SPX/10)
    'beta_lookback': 60,          # Days for rolling beta calculation
    'beta_shrinkage': 0.3,        # Shrink beta toward 1.0 (0.3 = 30% shrinkage)
    'hedge_threshold': 0.02,      # Re-hedge when BWD moves > 2% of notional

    # Position sizing
    'capital': 1_000_000,         # $1M notional portfolio
    'max_position_pct': 0.10,     # Max 10% in any single name
    'sector_cap': 0.30,           # Max 30% in any sector
    'max_vega': 0.08,             # Max 8% of portfolio vega per name
}

# Transaction costs (conservative estimates for 2011-2013)
COST_PARAMS = {
    'commission_per_contract': 1.00,      # $1.00 per contract
    'fee_per_contract': 0.25,             # Exchange fees
    'spread_capture': 0.50,               # Assume we capture 50% of bid-ask
    'spy_commission_per_share': 0.005,    # SPY hedge cost
}

# Risk limits
RISK_LIMITS = {
    'max_portfolio_delta': 0.05,          # Max 5% directional exposure
    'max_portfolio_vega': 0.15,           # Max 15% vega exposure
    'max_drawdown_pct': 0.20,             # Stop trading if DD > 20%
    'concentration_limit': 0.25,          # Max 25% in single position
}


def get_pilot_tickers():
    """Return list of pilot universe tickers."""
    return list(PILOT_UNIVERSE.keys())


def get_sector_weights():
    """Calculate sector weights in the pilot universe."""
    sectors = {}
    for ticker, info in PILOT_UNIVERSE.items():
        sector = info['sector']
        sectors[sector] = sectors.get(sector, 0) + 1

    total = len(PILOT_UNIVERSE)
    return {s: count/total for s, count in sectors.items()}


if __name__ == '__main__':
    print("Pilot Universe:")
    print("-" * 50)
    for ticker, info in PILOT_UNIVERSE.items():
        print(f"  {ticker:5} - {info['sector']:25} (${info['avg_price']:,.2f})")

    print("\nSector Weights:")
    for sector, weight in get_sector_weights().items():
        print(f"  {sector:25}: {weight:.1%}")

    print(f"\nData Periods: {DATA_PERIODS}")
