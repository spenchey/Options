"""
Convert stock quotes and optionstats CSV files to Parquet format.
"""

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from io import BytesIO
import re
import os
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Configuration
BUCKET = "opt-data-staging-project"
SOURCE_PREFIX = "General Backup/Raw Org Data by Year/"

AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-2')

s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)


def list_files(pattern):
    """List files matching pattern in S3."""
    files = []
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=BUCKET, Prefix=SOURCE_PREFIX):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if pattern in key and key.endswith('.csv'):
                files.append(key)
    return sorted(files)


def extract_date(filename):
    """Extract date from filename like stockquotes_20020401.csv"""
    match = re.search(r'_(\d{8})\.csv', filename)
    if match:
        d = match.group(1)
        return d[:4], d[4:6], d[6:8]
    return None, None, None


def convert_file(key, dest_prefix):
    """Convert a single CSV to Parquet."""
    year, month, day = extract_date(key)
    if not year:
        return None, 0

    # Read CSV from S3
    response = s3.get_object(Bucket=BUCKET, Key=key)
    df = pd.read_csv(BytesIO(response['Body'].read()), low_memory=False)
    df.columns = df.columns.str.strip().str.lower()

    # Parse date column
    if 'quotedate' in df.columns:
        df['quotedate'] = pd.to_datetime(df['quotedate'], errors='coerce')

    # Add partition columns
    df['year'] = int(year)
    df['month'] = int(month)

    # Write to Parquet
    table = pa.Table.from_pandas(df, preserve_index=False)
    buffer = BytesIO()
    pq.write_table(table, buffer, compression='snappy')
    buffer.seek(0)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    dest_key = f"{dest_prefix}/year={year}/month={month}/data_{timestamp}.parquet"

    s3.put_object(Bucket=BUCKET, Key=dest_key, Body=buffer.getvalue())
    return dest_key, len(df)


def convert_dataset(file_pattern, dest_prefix, name):
    """Convert all files matching pattern."""
    print(f"\n{'='*60}")
    print(f"Converting {name}")
    print(f"{'='*60}")

    files = list_files(file_pattern)
    print(f"Found {len(files)} files")

    total_rows = 0
    for i, key in enumerate(files, 1):
        filename = key.split('/')[-1]
        print(f"[{i}/{len(files)}] {filename}...", end=" ")
        try:
            dest, rows = convert_file(key, dest_prefix)
            if dest:
                total_rows += rows
                print(f"OK - {rows:,} rows")
            else:
                print("SKIPPED")
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\nTotal: {total_rows:,} rows")
    return total_rows


def main():
    print("STOCK DATA CONVERSION TO PARQUET")
    print("=" * 60)

    # Convert stock quotes
    stock_rows = convert_dataset(
        file_pattern="stockquotes_",
        dest_prefix="parquet/stocks",
        name="Stock Quotes (OHLCV)"
    )

    # Convert option stats
    stats_rows = convert_dataset(
        file_pattern="optionstats_",
        dest_prefix="parquet/optionstats",
        name="Option Stats (Aggregate IV/Volume)"
    )

    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print(f"Stock quotes: {stock_rows:,} rows")
    print(f"Option stats: {stats_rows:,} rows")
    print("=" * 60)


if __name__ == "__main__":
    main()
