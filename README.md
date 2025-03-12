# McClellan Oscillator & Summation Index

A Python application for calculating, analyzing, and visualizing the McClellan Oscillator and Summation Index based on market breadth data.

## Overview

The McClellan Oscillator and Summation Index are technical market breadth indicators used to evaluate the health and direction of the overall market. This program:

1. Fetches advancing and declining stocks data from a PostgreSQL database
2. Calculates the McClellan Oscillator using 19-day and 39-day EMAs of market breadth 
3. Calculates the McClellan Summation Index by accumulating the oscillator values
4. Provides visualizations and analysis of market conditions

## Prerequisites

- Python 3.9+
- PostgreSQL database(neon.tech)
- Schwab API credentials (for data loading)

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/McClellan.git
cd McClellan
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Set up environment variables by copying the example file:
```
cp .env.example .env
```

4. Edit the `.env` file with your credentials:
```
API_KEY=your_schwab_api_key
APP_SECRET=your_schwab_app_secret
CALLBACK_URL=your_callback_url
TOKEN_PATH=path_to_token
POSTGRES_HOST=your_db_host
POSTGRES_USER=your_db_user
POSTGRES_PASSWORD=your_db_password
POSTGRES_DB=your_db_name
POSTGRES_PORT=5432
POSTGRES_SSLMODE=require
```

## Database Setup

1. Create a PostgreSQL database:
```
createdb your_db_name
```

2. Set up the required tables:
```
psql -U your_db_user -d your_db_name -f setup_db.sql
```

This will create the `price_history` table with the necessary indices.

## Loading SPX Data

To load market breadth data ($ADVN and $DECN), use the `symbol_historical_data.py` script:

```
# Load advancing stocks data
python symbol_historical_data.py --symbol $ADVN --start-date 2022-01-01 --store-db

# Load declining stocks data
python symbol_historical_data.py --symbol $DECN --start-date 2022-01-01 --store-db
```

You can adjust the start date based on how much historical data you want to analyze.

## Running the McClellan Analysis

To calculate the McClellan Oscillator and Summation Index:

```
python McClellan_Summation_Index.py --start-date 2022-01-01 --end-date 2023-12-31 --output-csv results.csv
```

Optional parameters:
- `--print-rows N`: Print the last N rows of data (default: 50)
- `--ratio-adjusted`: Use ratio-adjusted breadth calculation
- `--output-csv FILENAME`: Save results to a CSV file

## Interpreting the Results

The program outputs:
- Advancing and declining numbers
- Raw or ratio-adjusted breadth
- 19-day and 39-day EMAs
- McClellan Oscillator values
- McClellan Summation Index values

General interpretation:
- McClellan Oscillator above zero: Positive market momentum
- McClellan Oscillator below zero: Negative market momentum
- McClellan Summation Index above 1000: Strong bullish market
- McClellan Summation Index below -1000: Strong bearish market

