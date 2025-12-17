"""
Download SPX (S&P 500 Index) historical data and upload to S3.
Creates a SPY-equivalent by dividing SPX by 10.
"""

import os
import yfinance as yf
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import boto3
from io import BytesIO
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

BUCKET = "opt-data-staging-project"
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-2')

s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)


def download_spx_data():
    """Download SPX data from Yahoo Finance."""
    print("Downloading SPX (^GSPC) data from Yahoo Finance...")

    # Download SPX index data (^GSPC is Yahoo's symbol for S&P 500)
    spx = yf.download("^GSPC", start="2000-01-01", end="2014-12-31", progress=False, auto_adjust=False)

    # Reset index to make date a column
    spx = spx.reset_index()

    # Flatten multi-level columns if present
    if isinstance(spx.columns, pd.MultiIndex):
        spx.columns = [col[0].lower() if col[1] == '' else col[0].lower() for col in spx.columns]
    else:
        spx.columns = [c.lower() for c in spx.columns]

    # Rename columns to standard names
    spx = spx.rename(columns={'adj close': 'adj_close'})

    # Add SPY-equivalent (SPX / 10)
    spx['spy_equiv'] = spx['close'] / 10

    # Add symbol column
    spx['symbol'] = 'SPX'

    # Calculate daily returns
    spx['return'] = spx['close'].pct_change()

    # Add year/month for partitioning
    spx['year'] = spx['date'].dt.year
    spx['month'] = spx['date'].dt.month

    print(f"Downloaded {len(spx)} rows from {spx['date'].min()} to {spx['date'].max()}")
    return spx


def upload_to_s3(df):
    """Upload SPX data to S3 as Parquet."""
    print("\nUploading to S3...")

    # Convert to Parquet
    table = pa.Table.from_pandas(df, preserve_index=False)
    buffer = BytesIO()
    pq.write_table(table, buffer, compression='snappy')
    buffer.seek(0)

    # Upload
    key = f"parquet/index/spx_data.parquet"
    s3.put_object(Bucket=BUCKET, Key=key, Body=buffer.getvalue())

    print(f"Uploaded to s3://{BUCKET}/{key}")
    return key


def main():
    print("=" * 60)
    print("SPX INDEX DATA DOWNLOAD")
    print("=" * 60)

    # Download
    df = download_spx_data()

    # Show sample
    print("\nSample data:")
    print(df[['date', 'close', 'spy_equiv', 'return']].head(10))

    # Show coverage for your data periods
    print("\nCoverage for your options data periods:")
    for year in [2002, 2004, 2005, 2011, 2012, 2013]:
        year_data = df[df['year'] == year]
        if len(year_data) > 0:
            print(f"  {year}: {len(year_data)} trading days, SPX range {year_data['close'].min():.0f} - {year_data['close'].max():.0f}")
        else:
            print(f"  {year}: No data")

    # Upload
    upload_to_s3(df)

    # Also save locally for reference
    local_path = ROOT / "data"
    local_path.mkdir(exist_ok=True)
    df.to_csv(local_path / "spx_data.csv", index=False)
    print(f"\nAlso saved locally to {local_path / 'spx_data.csv'}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
