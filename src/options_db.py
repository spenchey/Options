"""
Options Database Query Interface
================================
Connects to S3 Parquet data and provides easy query methods.

Usage:
    from src.options_db import OptionsDB
    db = OptionsDB()

    # Get summary of all data
    db.summary()

    # Query specific month (fast - uses partition pruning)
    df = db.query_month(2002, 4)

    # Query a ticker
    df = db.query_ticker('AAPL', year=2012)

    # Custom SQL
    df = db.sql("SELECT * FROM options WHERE Delta BETWEEN 0.4 AND 0.6 LIMIT 100")
"""

import os
import yaml
import duckdb
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


class OptionsDB:
    """Query interface for options data stored in S3 Parquet format."""

    def __init__(self, config_path: str = None):
        """Initialize connection to S3 using config.yml settings."""
        # Load config
        config_path = config_path or (ROOT / "config.yml")
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Extract settings
        self.bucket = self.config['s3']['bucket']
        self.parquet_prefix = self.config['s3']['prefixes']['parquet']
        self.region = self.config['aws']['region']

        # Get credentials from environment
        self.access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')

        # Initialize DuckDB
        self.con = duckdb.connect()
        self._setup_s3()

        print(f"Connected to: s3://{self.bucket}/{self.parquet_prefix}/")

    def _setup_s3(self):
        """Configure DuckDB for S3 access."""
        self.con.execute("INSTALL httpfs; LOAD httpfs;")

        # Try environment variables first, then fall back to credential chain
        if self.access_key and self.secret_key:
            self.con.execute(f"""
                CREATE SECRET (
                    TYPE S3,
                    KEY_ID '{self.access_key}',
                    SECRET '{self.secret_key}',
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

        # Create view for easy querying
        parquet_path = f"s3://{self.bucket}/{self.parquet_prefix}/*/*/*"
        self.con.execute(f"""
            CREATE OR REPLACE VIEW options AS
            SELECT * FROM read_parquet('{parquet_path}')
        """)

    def sql(self, query: str) -> pd.DataFrame:
        """Execute custom SQL query and return DataFrame."""
        return self.con.execute(query).df()

    def summary(self) -> pd.DataFrame:
        """Get summary statistics by year/month."""
        return self.sql("""
            SELECT
                year,
                month,
                COUNT(*) as total_rows,
                COUNT(DISTINCT UnderlyingSymbol) as unique_tickers,
                ROUND(AVG(IV), 4) as avg_iv,
                SUM(Volume) as total_volume,
                SUM(OpenInterest) as total_oi
            FROM options
            GROUP BY year, month
            ORDER BY year, month
        """)

    # Columns needed for backtest (minimal set for speed)
    BACKTEST_COLUMNS = [
        'datadate', 'underlyingsymbol', 'underlyingprice', 'expiration',
        'type', 'strike', 'bid', 'ask', 'delta', 'gamma', 'theta', 'vega', 'iv'
    ]

    def query_month(self, year: int, month: int, limit: int = None,
                    tickers: list = None, minimal: bool = False) -> pd.DataFrame:
        """
        Query data for a specific month (uses partition pruning).
        
        Args:
            year, month: Period to query
            limit: Optional row limit
            tickers: Optional list of tickers to filter (huge speedup!)
            minimal: If True, only return columns needed for backtest
        """
        path = f"s3://{self.bucket}/{self.parquet_prefix}/year={year}/month={month:02d}/*.parquet"
        
        # Column selection
        if minimal:
            cols = ', '.join(self.BACKTEST_COLUMNS)
        else:
            cols = '*'
        
        query = f"SELECT {cols} FROM read_parquet('{path}')"
        
        # Ticker filter (pushdown to parquet - much faster)
        if tickers:
            ticker_list = "', '".join(tickers)
            query += f" WHERE underlyingsymbol IN ('{ticker_list}')"
        
        if limit:
            query += f" LIMIT {limit}"
        
        df = self.con.execute(query).df()
        df = self._normalize_columns(df)
        return df
    
    def query_month_fast(self, year: int, month: int, tickers: list) -> pd.DataFrame:
        """
        Fast query for backtest - minimal columns, filtered tickers.
        This is 10-50x faster than query_month with SELECT *.
        """
        return self.query_month(year, month, tickers=tickers, minimal=True)

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names from lowercase to expected CamelCase."""
        column_map = {
            'datadate': 'DataDate',
            'underlyingsymbol': 'UnderlyingSymbol',
            'underlyingprice': 'UnderlyingPrice',
            'expiration': 'Expiration',
            'type': 'Type',
            'strike': 'Strike',
            'bid': 'Bid',
            'ask': 'Ask',
            'iv': 'IV',
            'delta': 'Delta',
            'gamma': 'Gamma',
            'theta': 'Theta',
            'vega': 'Vega',
            'volume': 'Volume',
            'openinterest': 'OpenInterest',
            'year': 'year',
            'month': 'month'
        }
        df.columns = [column_map.get(c.lower(), c) for c in df.columns]
        return df

    def query_ticker(self, ticker: str, year: int = None, month: int = None,
                     option_type: str = None) -> pd.DataFrame:
        """Query all options for a specific underlying ticker."""
        if year and month:
            path = f"s3://{self.bucket}/{self.parquet_prefix}/year={year}/month={month:02d}/*.parquet"
        elif year:
            path = f"s3://{self.bucket}/{self.parquet_prefix}/year={year}/*/*.parquet"
        else:
            path = f"s3://{self.bucket}/{self.parquet_prefix}/*/*/*"

        query = f"""
            SELECT * FROM read_parquet('{path}')
            WHERE UnderlyingSymbol = '{ticker}'
        """
        if option_type:
            query += f" AND Type = '{option_type}'"
        query += " ORDER BY DataDate, Expiration, Strike"

        return self.con.execute(query).df()

    def get_underlyings(self, year: int = None, month: int = None) -> pd.DataFrame:
        """Get list of all underlying symbols with statistics."""
        if year and month:
            path = f"s3://{self.bucket}/{self.parquet_prefix}/year={year}/month={month:02d}/*.parquet"
        else:
            path = f"s3://{self.bucket}/{self.parquet_prefix}/*/*/*"

        return self.con.execute(f"""
            SELECT
                UnderlyingSymbol,
                COUNT(*) as option_rows,
                AVG(UnderlyingPrice) as avg_price,
                MIN(DataDate) as first_date,
                MAX(DataDate) as last_date
            FROM read_parquet('{path}')
            GROUP BY UnderlyingSymbol
            ORDER BY option_rows DESC
        """).df()

    def daily_stats(self, year: int, month: int) -> pd.DataFrame:
        """Get daily statistics for a month."""
        path = f"s3://{self.bucket}/{self.parquet_prefix}/year={year}/month={month:02d}/*.parquet"
        return self.con.execute(f"""
            SELECT
                DataDate,
                COUNT(*) as contracts,
                COUNT(DISTINCT UnderlyingSymbol) as tickers,
                SUM(Volume) as total_volume,
                SUM(OpenInterest) as total_oi,
                ROUND(AVG(IV), 4) as avg_iv,
                ROUND(AVG(CASE WHEN Type = 'call' THEN Delta END), 4) as avg_call_delta,
                ROUND(AVG(CASE WHEN Type = 'put' THEN Delta END), 4) as avg_put_delta
            FROM read_parquet('{path}')
            GROUP BY DataDate
            ORDER BY DataDate
        """).df()

    def available_data(self) -> pd.DataFrame:
        """Show what year/month combinations are available."""
        return self.sql("""
            SELECT DISTINCT year, month
            FROM options
            ORDER BY year, month
        """)

    def query_by_delta(self, delta_min: float, delta_max: float,
                       year: int = None, month: int = None,
                       option_type: str = None) -> pd.DataFrame:
        """Query options by delta range (useful for strategy research)."""
        if year and month:
            path = f"s3://{self.bucket}/{self.parquet_prefix}/year={year}/month={month:02d}/*.parquet"
        else:
            path = f"s3://{self.bucket}/{self.parquet_prefix}/*/*/*"

        query = f"""
            SELECT * FROM read_parquet('{path}')
            WHERE ABS(Delta) BETWEEN {delta_min} AND {delta_max}
        """
        if option_type:
            query += f" AND Type = '{option_type}'"

        return self.con.execute(query).df()


# Quick test if run directly
if __name__ == "__main__":
    print("Testing OptionsDB connection...")
    db = OptionsDB()

    print("\nAvailable data periods:")
    print(db.available_data())

    print("\nQuick sample from April 2002:")
    df = db.query_month(2002, 4, limit=5)
    print(df[['UnderlyingSymbol', 'Type', 'Strike', 'Bid', 'Ask', 'IV', 'Delta']])
