"""
Upload CSV files from thumb drive ZIPs to S3 and convert to Parquet.
Processes all data from Jan 2002 - Oct 2013.
"""

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from io import BytesIO
import zipfile
import re
import os
import json
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Configuration
BUCKET = "opt-data-staging-project"
THUMB_DRIVE = Path("E:/")
S3_RAW_PREFIX = "General Backup/Raw Org Data by Year"
PROGRESS_FILE = ROOT / "upload_progress.json"

AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-2')

s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)


def load_progress():
    """Load progress from file."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed_zips": [], "stats": {"options": 0, "stocks": 0, "optionstats": 0}}


def save_progress(progress):
    """Save progress to file."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def find_all_zips():
    """Find all ZIP files on thumb drive."""
    zips = []
    for year_dir in sorted(THUMB_DRIVE.iterdir()):
        if year_dir.is_dir() and year_dir.name.isdigit():
            for zip_file in sorted(year_dir.glob("*.zip")):
                zips.append(zip_file)
    return zips


def extract_date(filename):
    """Extract year, month, day from filename like options_20020401.csv"""
    match = re.search(r'_(\d{8})\.csv', filename)
    if match:
        d = match.group(1)
        return d[:4], d[4:6], d[6:8]
    return None, None, None


def upload_csv_to_s3(file_path, year):
    """Upload a single CSV file to S3."""
    filename = os.path.basename(file_path)
    s3_key = f"{S3_RAW_PREFIX}/{year}/{filename}"

    with open(file_path, 'rb') as f:
        s3.put_object(Bucket=BUCKET, Key=s3_key, Body=f.read())

    return s3_key


def convert_csv_to_parquet(file_path, file_type):
    """Convert a CSV file to Parquet and upload to S3."""
    filename = os.path.basename(file_path)
    year, month, day = extract_date(filename)

    if not year:
        return None, 0

    # Read CSV
    df = pd.read_csv(file_path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()

    # Parse date columns based on file type
    if file_type == "options":
        if 'datadate' in df.columns:
            df['datadate'] = pd.to_datetime(df['datadate'], errors='coerce')
        if 'expiration' in df.columns:
            df['expiration'] = pd.to_datetime(df['expiration'], errors='coerce')
    elif file_type in ["stocks", "optionstats"]:
        if 'quotedate' in df.columns:
            df['quotedate'] = pd.to_datetime(df['quotedate'], errors='coerce')

    # Add partition columns
    df['year'] = int(year)
    df['month'] = int(month)

    # Determine destination prefix
    if file_type == "options":
        dest_prefix = "parquet/options"
    elif file_type == "stocks":
        dest_prefix = "parquet/stocks"
    else:
        dest_prefix = "parquet/optionstats"

    # Write to Parquet
    table = pa.Table.from_pandas(df, preserve_index=False)
    buffer = BytesIO()
    pq.write_table(table, buffer, compression='snappy')
    buffer.seek(0)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    dest_key = f"{dest_prefix}/year={year}/month={int(month):02d}/data_{timestamp}.parquet"

    s3.put_object(Bucket=BUCKET, Key=dest_key, Body=buffer.getvalue())
    return dest_key, len(df)


def process_zip(zip_path, progress):
    """Process a single ZIP file: extract, upload CSVs to S3, convert to Parquet."""
    zip_name = zip_path.name
    year = zip_path.parent.name

    print(f"\n{'='*60}")
    print(f"Processing: {zip_name}")
    print(f"{'='*60}")

    stats = {"options": 0, "stocks": 0, "optionstats": 0}

    # Create temp directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Extract ZIP
        print(f"Extracting to temp folder...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(temp_path)
        except Exception as e:
            print(f"ERROR extracting ZIP: {e}")
            return stats

        # Find all CSV files (may be in subdirectory)
        csv_files = list(temp_path.rglob("*.csv"))
        print(f"Found {len(csv_files)} CSV files")

        for csv_file in sorted(csv_files):
            filename = csv_file.name

            # Skip hidden/system files
            if filename.startswith('.'):
                continue

            # Determine file type
            if filename.startswith('options_'):
                file_type = "options"
            elif filename.startswith('stockquotes_'):
                file_type = "stocks"
            elif filename.startswith('optionstats_'):
                file_type = "optionstats"
            else:
                print(f"  Skipping unknown file: {filename}")
                continue

            print(f"  [{file_type}] {filename}...", end=" ", flush=True)

            try:
                # Upload CSV to S3
                upload_csv_to_s3(csv_file, year)

                # Convert to Parquet
                dest_key, rows = convert_csv_to_parquet(csv_file, file_type)

                if dest_key:
                    stats[file_type] += rows
                    print(f"OK - {rows:,} rows")
                else:
                    print("SKIPPED")
            except Exception as e:
                print(f"ERROR: {e}")

    return stats


def main():
    print("=" * 60)
    print("UPLOAD AND CONVERT: Thumb Drive -> S3 -> Parquet")
    print("=" * 60)

    # Load progress
    progress = load_progress()
    completed = set(progress["completed_zips"])

    # Find all ZIPs
    all_zips = find_all_zips()
    print(f"\nFound {len(all_zips)} ZIP files on thumb drive")
    print(f"Already completed: {len(completed)}")

    # Filter to remaining
    remaining = [z for z in all_zips if str(z) not in completed]
    print(f"Remaining to process: {len(remaining)}")

    if not remaining:
        print("\nAll files already processed!")
        return

    total_stats = progress["stats"]

    for i, zip_path in enumerate(remaining, 1):
        print(f"\n[{i}/{len(remaining)}] {zip_path.name}")

        try:
            stats = process_zip(zip_path, progress)

            # Update totals
            for key in stats:
                total_stats[key] += stats[key]

            # Mark as completed
            progress["completed_zips"].append(str(zip_path))
            progress["stats"] = total_stats
            save_progress(progress)

            print(f"\nRunning totals:")
            print(f"  Options:     {total_stats['options']:,} rows")
            print(f"  Stocks:      {total_stats['stocks']:,} rows")
            print(f"  Optionstats: {total_stats['optionstats']:,} rows")

        except Exception as e:
            print(f"ERROR processing {zip_path}: {e}")
            continue

    print("\n" + "=" * 60)
    print("UPLOAD AND CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Total options rows:     {total_stats['options']:,}")
    print(f"Total stocks rows:      {total_stats['stocks']:,}")
    print(f"Total optionstats rows: {total_stats['optionstats']:,}")


if __name__ == "__main__":
    main()
