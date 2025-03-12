import os
import argparse
from schwab import auth, client
import pandas as pd
from datetime import datetime, timezone
import dotenv
import logging
import psycopg2
from psycopg2 import sql

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_environment_variables():
    """Load environment variables from .env file first, fallback to os.environ"""
    # Load .env file
    env_file = dotenv.find_dotenv()
    if env_file:
        logger.info(f"Found .env file at: {env_file}")
        config = dotenv.dotenv_values(env_file)
    else:
        logger.warning("No .env file found")
        config = {}
    
    # Load and log each variable's source
    vars_dict = {}
    for var_name in ['API_KEY', 'APP_SECRET', 'CALLBACK_URL', 'TOKEN_PATH',
                     'POSTGRES_DB', 'POSTGRES_USER', 'POSTGRES_PASSWORD',
                     'POSTGRES_HOST', 'POSTGRES_PORT', 'POSTGRES_SSLMODE']:
        env_value = config.get(var_name)
        if env_value:
            logger.debug(f"{var_name} found in .env file")
        else:
            env_value = os.getenv(var_name)
            if env_value:
                logger.debug(f"{var_name} found in environment")
            else:
                logger.debug(f"{var_name} not found")
        
        # Special handling for variables with defaults
        if var_name == 'POSTGRES_PORT':
            vars_dict[var_name] = env_value or '5432'
        elif var_name == 'POSTGRES_SSLMODE':
            vars_dict[var_name] = env_value or 'require'
        else:
            vars_dict[var_name] = env_value
    
    return vars_dict

def create_indexes(conn):
    """Create necessary indexes on price_history table"""
    with conn.cursor() as cur:
        # Create index on symbol
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_price_history_symbol 
            ON price_history(symbol)
        """)
        
        # # Create composite index on symbol and datetime
        # cur.execute("""
        #     CREATE INDEX IF NOT EXISTS idx_price_history_symbol_datetime 
        #     ON price_history(symbol, datetime)
        # """)
    conn.commit()
    logger.info("Created indexes on price_history table")

def create_table(conn):
    """Create price_history table if it doesn't exist"""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS price_history (
                datetime TIMESTAMP WITH TIME ZONE,
                symbol VARCHAR(10),
                open DECIMAL,
                high DECIMAL,
                low DECIMAL,
                close DECIMAL,
                volume BIGINT,
                PRIMARY KEY (datetime, symbol)
            )
        """)
        
    # Create indexes after table creation
    create_indexes(conn)
    
    conn.commit()

def store_in_postgresql(df, env_vars, batch_size=1000):
    """Store DataFrame in PostgreSQL database"""
    try:
        conn = psycopg2.connect(
            dbname=env_vars['POSTGRES_DB'],
            user=env_vars['POSTGRES_USER'],
            password=env_vars['POSTGRES_PASSWORD'],
            host=env_vars['POSTGRES_HOST'],
            port=env_vars['POSTGRES_PORT'],
            sslmode=env_vars['POSTGRES_SSLMODE']
        )
        
        # Ensure datetime is timezone-aware
        df.index = pd.to_datetime(df.index).tz_localize('UTC')
        
        with conn.cursor() as cur:
            sql_query = """
                INSERT INTO price_history (datetime, symbol, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (datetime, symbol) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
            """
            
            # Convert DataFrame to list of tuples for batch insert
            data = [
                (idx, row.name, row['open'], row['high'], row['low'], row['close'], row['volume'])
                for idx, row in df.iterrows()
            ]
            
            # Insert in batches
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                cur.executemany(sql_query, batch)
                conn.commit()
                logger.info(f"Inserted batch of {len(batch)} records")
        
        logger.info(f"Successfully stored {len(df)} records in database")
        
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        raise
    finally:
        if 'conn' in locals() and conn:
            conn.close()

def get_price_history(symbol, start_datetime=None, end_datetime=None, store_db=False):
    """Get historical price data for a symbol"""
    try:
        # Load environment variables
        env_vars = load_environment_variables()
        
        # Create Schwab client with authentication
        c = auth.easy_client(
            env_vars['API_KEY'],
            env_vars['APP_SECRET'],
            env_vars['CALLBACK_URL'],
            env_vars['TOKEN_PATH']
        )
        
        # Get price history - pass datetime objects directly
        response = c.get_price_history_every_day(
            symbol,
            start_datetime=start_datetime,  # Pass datetime object directly
            end_datetime=end_datetime       # Pass datetime object directly
        )
        
        # Check if request was successful
        if response.status_code != 200:
            logger.error(f"Failed to fetch price history. Status code: {response.status_code}")
            return None
            
        # Parse response to JSON
        data = response.json()
        print(data)
        
        # Convert response to DataFrame
        if data and 'candles' in data:
            df = pd.DataFrame(data['candles'])
            
            # Convert datetime column
            df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
            df.set_index('datetime', inplace=True)
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Reorder columns
            columns = ['symbol', 'open', 'high', 'low', 'close', 'volume']
            df = df[columns]
            
            logger.info(f"DataFrame Shape: {df.shape}")
            logger.info(f"Date Range: {df.index.min()} to {df.index.max()}")
            
            # Store in PostgreSQL if requested
            if store_db:
                store_in_postgresql(df, env_vars)
            
            return df
        else:
            logger.error("No candles data in response")
            return None

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(f"Response data: {data if 'data' in locals() else 'No response data'}")
        return None

def parse_datetime(datetime_str):
    """Parse datetime string in format YYYY-MM-DD"""
    try:
        return datetime.strptime(datetime_str, '%Y-%m-%d')
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid date format. Use YYYY-MM-DD: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch historical price data from Schwab API')
    parser.add_argument('--symbol', type=str, required=True, help='Symbol to fetch data for')
    parser.add_argument('--start-date', type=parse_datetime, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=parse_datetime, help='End date (YYYY-MM-DD)')
    parser.add_argument('--store-db', action='store_true', help='Store results in PostgreSQL')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for database inserts')
    parser.add_argument('--output-csv', type=str, help='Save results to CSV file')
    
    args = parser.parse_args()
    
    df = get_price_history(
        args.symbol,
        start_datetime=args.start_date,
        end_datetime=args.end_date,
        store_db=args.store_db
    )
    
    if df is not None and args.output_csv:
        df.to_csv(args.output_csv)
        logger.info(f"Results saved to {args.output_csv}")