"""
Market Data Module
==================
Unified access to options, stocks, and index data with beta calculations.

Usage:
    from src.market_data import MarketData
    md = MarketData()

    # Get stock prices
    aapl = md.get_stock('AAPL', year=2012, month=1)

    # Get SPX/market returns
    spx = md.get_spx(year=2012, month=1)

    # Calculate rolling betas
    betas = md.calculate_betas(year=2012, month=1, window=60)

    # Get beta-weighted delta for a position
    bwd = md.beta_weighted_delta(ticker='AAPL', delta=0.5, shares=100, beta=1.2)
"""

import os
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


class MarketData:
    """Unified market data access with beta calculations."""

    def __init__(self):
        """Initialize connection to S3 data."""
        self.bucket = "opt-data-staging-project"
        self.region = os.getenv('AWS_REGION', 'us-east-2')

        # Initialize DuckDB
        self.con = duckdb.connect()
        self._setup_s3()

        print(f"MarketData connected to s3://{self.bucket}/")

    def _setup_s3(self):
        """Configure DuckDB for S3 access."""
        self.con.execute("INSTALL httpfs; LOAD httpfs;")

        # Try environment variables first, then fall back to credential chain
        access_key = os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

        if access_key and secret_key:
            self.con.execute(f"""
                CREATE SECRET (
                    TYPE S3,
                    KEY_ID '{access_key}',
                    SECRET '{secret_key}',
                    REGION '{self.region}'
                );
            """)
        else:
            # Use AWS credential chain (shared credentials file, IAM role, etc.)
            self.con.execute(f"""
                CREATE SECRET (
                    TYPE S3,
                    PROVIDER CREDENTIAL_CHAIN,
                    REGION '{self.region}'
                );
            """)

    def sql(self, query: str) -> pd.DataFrame:
        """Execute SQL query."""
        return self.con.execute(query).df()

    # ========== STOCK DATA ==========

    def get_stock(self, ticker: str, year: int = None, month: int = None) -> pd.DataFrame:
        """Get stock OHLCV data for a ticker."""
        if year and month:
            path = f"s3://{self.bucket}/parquet/stocks/year={year}/month={month:02d}/*.parquet"
        elif year:
            path = f"s3://{self.bucket}/parquet/stocks/year={year}/*/*.parquet"
        else:
            path = f"s3://{self.bucket}/parquet/stocks/*/*/*"

        return self.sql(f"""
            SELECT * FROM read_parquet('{path}')
            WHERE UPPER(symbol) = UPPER('{ticker}')
            ORDER BY quotedate
        """)

    def get_all_stocks(self, year: int, month: int) -> pd.DataFrame:
        """Get all stock data for a month."""
        path = f"s3://{self.bucket}/parquet/stocks/year={year}/month={month:02d}/*.parquet"
        return self.sql(f"SELECT * FROM read_parquet('{path}') ORDER BY symbol, quotedate")

    # ========== INDEX DATA ==========

    def get_spx(self, year: int = None, month: int = None) -> pd.DataFrame:
        """Get SPX index data."""
        path = f"s3://{self.bucket}/parquet/index/spx_data.parquet"
        query = f"SELECT * FROM read_parquet('{path}')"

        if year and month:
            query += f" WHERE year = {year} AND month = {month}"
        elif year:
            query += f" WHERE year = {year}"

        query += " ORDER BY date"
        return self.sql(query)

    def get_spy_price(self, date: str) -> float:
        """
        Get SPY price for a specific date.

        Uses SPX data with spy_equiv column, or derives from SPX/10.

        Args:
            date: Date string (YYYY-MM-DD)

        Returns:
            SPY price (float), or 0 if not found
        """
        path = f"s3://{self.bucket}/parquet/index/spx_data.parquet"
        date = pd.to_datetime(date).strftime('%Y-%m-%d')

        try:
            result = self.sql(f"""
                SELECT spy_equiv, close
                FROM read_parquet('{path}')
                WHERE date = '{date}'
                LIMIT 1
            """)

            if len(result) == 0:
                # Try to find nearest date
                result = self.sql(f"""
                    SELECT spy_equiv, close, date
                    FROM read_parquet('{path}')
                    WHERE date <= '{date}'
                    ORDER BY date DESC
                    LIMIT 1
                """)

            if len(result) > 0:
                # Use spy_equiv if available, otherwise SPX/10
                spy = result['spy_equiv'].iloc[0]
                if pd.isna(spy) or spy == 0:
                    spy = result['close'].iloc[0] / 10
                return float(spy)

        except Exception as e:
            print(f"Warning: Could not get SPY price for {date}: {e}")

        return 0.0

    # ========== BETA CALCULATIONS ==========

    def calculate_stock_returns(self, year: int, month: int) -> pd.DataFrame:
        """Calculate daily returns for all stocks in a month."""
        stocks = self.get_all_stocks(year, month)

        # Calculate returns by ticker
        stocks = stocks.sort_values(['symbol', 'quotedate'])
        stocks['return'] = stocks.groupby('symbol')['close'].pct_change()

        return stocks[['symbol', 'quotedate', 'close', 'return']]

    def calculate_betas(self, year: int, month: int, lookback_days: int = 60) -> pd.DataFrame:
        """
        Calculate rolling betas for all stocks vs SPX.

        Uses OLS regression: stock_return = alpha + beta * market_return + epsilon

        Args:
            year: Year to calculate betas for
            month: Month to calculate betas for
            lookback_days: Number of trading days for rolling window (default 60)

        Returns:
            DataFrame with columns: symbol, date, beta, r_squared
        """
        # Get SPX returns
        spx = self.get_spx()
        spx = spx[['date', 'return']].rename(columns={'return': 'mkt_return', 'date': 'quotedate'})
        spx['quotedate'] = pd.to_datetime(spx['quotedate'])

        # Get stock returns for this period + lookback
        # Need to get more data for the lookback period
        stocks = self.sql(f"""
            SELECT symbol, quotedate, close
            FROM read_parquet('s3://{self.bucket}/parquet/stocks/*/*/*')
            WHERE year >= {year - 1}  -- Get extra year for lookback
            ORDER BY symbol, quotedate
        """)

        stocks['quotedate'] = pd.to_datetime(stocks['quotedate'])
        stocks = stocks.sort_values(['symbol', 'quotedate'])
        stocks['return'] = stocks.groupby('symbol')['close'].pct_change()

        # Merge with market returns
        merged = stocks.merge(spx, on='quotedate', how='inner')
        merged = merged.dropna(subset=['return', 'mkt_return'])

        # Calculate rolling betas
        def calc_rolling_beta(group):
            if len(group) < lookback_days:
                return pd.DataFrame()

            results = []
            for i in range(lookback_days, len(group)):
                window = group.iloc[i-lookback_days:i]
                x = window['mkt_return'].values
                y = window['return'].values

                # OLS: beta = cov(x,y) / var(x)
                cov = np.cov(x, y)[0, 1]
                var = np.var(x)
                beta = cov / var if var > 0 else 1.0

                # R-squared
                corr = np.corrcoef(x, y)[0, 1]
                r2 = corr ** 2 if not np.isnan(corr) else 0

                results.append({
                    'symbol': group['symbol'].iloc[0],
                    'quotedate': group['quotedate'].iloc[i],
                    'beta': beta,
                    'r_squared': r2
                })

            return pd.DataFrame(results)

        # Apply to each stock
        print(f"Calculating {lookback_days}-day rolling betas...")
        betas = merged.groupby('symbol').apply(calc_rolling_beta, include_groups=False)

        if len(betas) > 0:
            betas = betas.reset_index(drop=True)
            # Filter to requested year/month
            betas['quotedate'] = pd.to_datetime(betas['quotedate'])
            betas = betas[
                (betas['quotedate'].dt.year == year) &
                (betas['quotedate'].dt.month == month)
            ]

        return betas

    def get_beta(self, ticker: str, date: str, lookback_days: int = 60) -> float:
        """
        Get beta for a single ticker on a specific date.

        Args:
            ticker: Stock symbol
            date: Date string (YYYY-MM-DD)
            lookback_days: Days of history for calculation

        Returns:
            Beta value (float), or 1.0 if insufficient data
        """
        date = pd.to_datetime(date)

        # Get stock returns
        stock = self.sql(f"""
            SELECT quotedate, close
            FROM read_parquet('s3://{self.bucket}/parquet/stocks/*/*/*')
            WHERE UPPER(symbol) = UPPER('{ticker}')
            ORDER BY quotedate
        """)

        if len(stock) < lookback_days:
            return 1.0  # Default beta

        stock['quotedate'] = pd.to_datetime(stock['quotedate'])
        stock = stock[stock['quotedate'] <= date].tail(lookback_days + 1)
        stock['return'] = stock['close'].pct_change()

        # Get SPX returns
        spx = self.get_spx()
        spx['date'] = pd.to_datetime(spx['date'])
        spx = spx[spx['date'] <= date].tail(lookback_days + 1)

        # Merge and calculate
        merged = stock.merge(spx[['date', 'return']], left_on='quotedate', right_on='date')
        merged = merged.rename(columns={'return_x': 'stock_ret', 'return_y': 'mkt_ret'})
        merged = merged.dropna()

        if len(merged) < 20:  # Need minimum data
            return 1.0

        x = merged['mkt_ret'].values
        y = merged['stock_ret'].values

        cov = np.cov(x, y)[0, 1]
        var = np.var(x)

        return cov / var if var > 0 else 1.0

    # ========== BETA-WEIGHTED DELTA ==========

    def beta_weighted_delta(self, ticker: str, delta: float, shares: float,
                            stock_price: float, beta: float = None,
                            date: str = None) -> dict:
        """
        Calculate beta-weighted delta for a position.

        Beta-weighted delta converts position delta to SPX-equivalent exposure.

        Args:
            ticker: Stock symbol
            delta: Option delta (e.g., 0.5 for ATM call)
            shares: Number of shares the options control (contracts * 100)
            stock_price: Current stock price
            beta: Stock beta (if None, will calculate from date)
            date: Date for beta calculation (required if beta not provided)

        Returns:
            dict with position_delta, beta_weighted_delta, spx_equivalent
        """
        if beta is None:
            if date is None:
                raise ValueError("Either beta or date must be provided")
            beta = self.get_beta(ticker, date)

        # Position delta in dollar terms
        position_delta = delta * shares * stock_price

        # Beta-weighted delta (SPX-equivalent exposure)
        beta_weighted = position_delta * beta

        # SPX equivalent shares (assuming SPX ~ 10 * SPY)
        spx = self.get_spx()
        spx_price = spx['close'].iloc[-1]
        spx_equiv_shares = beta_weighted / spx_price

        return {
            'ticker': ticker,
            'delta': delta,
            'shares': shares,
            'stock_price': stock_price,
            'beta': beta,
            'position_delta_$': position_delta,
            'beta_weighted_delta_$': beta_weighted,
            'spx_equivalent_shares': spx_equiv_shares
        }


# Quick test
if __name__ == "__main__":
    print("Testing MarketData...")
    md = MarketData()

    print("\nSPX data sample:")
    spx = md.get_spx(year=2002, month=4)
    print(spx[['date', 'close', 'spy_equiv', 'return']].head())

    print("\nAAPL stock data (April 2002):")
    aapl = md.get_stock('AAPL', year=2002, month=4)
    print(aapl.head())

    print("\nCalculating beta for AAPL on 2002-04-15...")
    beta = md.get_beta('AAPL', '2002-04-15', lookback_days=60)
    print(f"AAPL beta: {beta:.3f}")
