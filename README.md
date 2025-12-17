# Options Research Pipeline

Beta-neutral short-premium options trading strategy research using historical options data.

## Project Structure

```
Options/
  config.yml          # S3 bucket, prefixes, data schema (Claude reads this)
  state.json          # Pipeline progress/memory (Claude reads this)
  .env                # AWS credentials (NOT committed)
  .env.example        # Template for credentials
  requirements.txt    # Python dependencies
  src/
    options_db.py     # Database query interface
  sql/                # SQL queries and views
  docs/               # Research notes and documentation
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up credentials:**
   ```bash
   cp .env.example .env
   # Edit .env with your AWS credentials
   ```

3. **Query the data:**
   ```python
   from src.options_db import OptionsDB

   db = OptionsDB()

   # See available data
   db.available_data()

   # Query a specific month
   df = db.query_month(2002, 4)

   # Query a ticker
   df = db.query_ticker('AAPL', year=2012)

   # Custom SQL
   df = db.sql("SELECT * FROM options WHERE Delta BETWEEN 0.4 AND 0.6 LIMIT 100")
   ```

## For Claude Code

When resuming work on this project:

1. Open this folder in VS Code
2. Read `config.yml` to understand S3 connection and data schema
3. Read `state.json` to see what's been completed
4. Use `src/options_db.py` to query the data

**Key commands:**
```python
from src.options_db import OptionsDB
db = OptionsDB()
```

## Data Summary

| Metric | Value |
|--------|-------|
| Total Rows | 35,408,466 |
| Format | Parquet (partitioned by year/month) |
| Location | `s3://opt-data-staging-project/parquet/options/` |
| Years | 2002, 2004, 2005, 2011, 2012, 2013 |

## Research Goal

Develop a profitable beta-neutral short-premium options trading strategy:
- Harvest variance risk premium
- Hedge market beta using SPX/ES futures
- Implement crash protection overlays
