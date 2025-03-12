import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
import dotenv
import os
import logging
from datetime import datetime
import mplcursors
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_environment_variables():
    """Load environment variables from .env file first, fallback to os.environ"""
    env_file = dotenv.find_dotenv()
    if env_file:
        logger.info(f"Found .env file at: {env_file}")
        config = dotenv.dotenv_values(env_file)
    else:
        logger.warning("No .env file found")
        config = {}
    
    vars_dict = {}
    for var_name in ['POSTGRES_DB', 'POSTGRES_USER', 'POSTGRES_PASSWORD',
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
        
        if var_name == 'POSTGRES_PORT':
            vars_dict[var_name] = env_value or '5432'
        elif var_name == 'POSTGRES_SSLMODE':
            vars_dict[var_name] = env_value or 'require'
        else:
            vars_dict[var_name] = env_value
    
    return vars_dict

def get_price_history_from_db(symbol, start_date=None, end_date=None):
    """Fetch price history from PostgreSQL database"""
    env_vars = load_environment_variables()
    
    try:
        conn = psycopg2.connect(
            dbname=env_vars['POSTGRES_DB'],
            user=env_vars['POSTGRES_USER'],
            password=env_vars['POSTGRES_PASSWORD'],
            host=env_vars['POSTGRES_HOST'],
            port=env_vars['POSTGRES_PORT'],
            sslmode=env_vars['POSTGRES_SSLMODE']
        )
        
        query = """
            SELECT datetime, symbol, open, high, low, close, volume 
            FROM price_history 
            WHERE symbol = %s
        """
        params = [symbol]
        
        if start_date:
            query += " AND datetime >= %s"
            params.append(start_date)
        if end_date:
            query += " AND datetime <= %s"
            params.append(end_date)
            
        query += " ORDER BY datetime"
        
        df = pd.read_sql_query(query, conn, params=params, parse_dates=['datetime'])
        df.set_index('datetime', inplace=True)
        
        logger.info(f"Retrieved {len(df)} records for {symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        raise
    finally:
        if 'conn' in locals() and conn:
            conn.close()

def calculate_mcclellan_summation(advn_df, decn_df, ratio_adjusted=False):
    """
    Calculate McClellan Summation Index from advance/decline data
    
    Parameters:
    advn_df (pd.DataFrame): DataFrame containing advancing stocks data ($ADVN)
    decn_df (pd.DataFrame): DataFrame containing declining stocks data ($DECN)
    ratio_adjusted (bool): Whether to use ratio-adjusted breadth calculation (default: True)
    
    Returns:
    pd.DataFrame: DataFrame with all calculated metrics
    """
    # Create a DataFrame with advancing and declining numbers
    df = pd.DataFrame({
        'advancing': advn_df['close'],
        'declining': decn_df['close']
    })
    
    # Calculate breadth according to ThinkOrSwim logic
    if ratio_adjusted:
        # Ratio-adjusted breadth calculation
        df['breadth'] = 1000 * (df['advancing'] - df['declining']) / (df['advancing'] + df['declining'])
    else:
        # Raw breadth calculation
        df['breadth'] = df['advancing'] - df['declining']
    
    # Calculate EMAs using pandas ewm() function which is more efficient and accurate
    # The span parameter is (period * 2 - 1) for equivalent traditional EMA calculation
    df['EMA_19'] = df['breadth'].ewm(span=19, adjust=False).mean()
    df['EMA_39'] = df['breadth'].ewm(span=39, adjust=False).mean()
    
    # Calculate McClellan Oscillator
    df['McClellan Oscillator'] = df['EMA_19'] - df['EMA_39']
    
    # Calculate Summation Index with proper initialization and accumulation
    df['McClellan Summation Index'] = 1000.0  # Standard starting value
    start_idx = df['McClellan Oscillator'].first_valid_index()
    
    if start_idx is not None:
        # Get the integer location of the start index
        start_loc = df.index.get_loc(start_idx)
        
        for i in range(start_loc, len(df)):
            if i == start_loc:
                df.iloc[i, df.columns.get_loc('McClellan Summation Index')] = 1000.0 + df.iloc[i]['McClellan Oscillator']
            else:
                df.iloc[i, df.columns.get_loc('McClellan Summation Index')] = (
                    df.iloc[i-1]['McClellan Summation Index'] + 
                    df.iloc[i]['McClellan Oscillator']
                )
    
    return df

def analyze_mcclellan(df):
    """
    Generate trading signals based on McClellan Summation Index
    
    Parameters:
    df (pd.DataFrame): DataFrame with McClellan Summation Index calculations
    
    Returns:
    pd.DataFrame: DataFrame with added signals and analysis
    """
    analysis_df = df.copy()
    
    # Add signal column
    analysis_df['Signal'] = 0
    
    # Generate signals based on traditional thresholds
    # Adjusted thresholds based on the new calculation method
    analysis_df.loc[analysis_df['McClellan Summation Index'] > 2000, 'Signal'] = 1  # Bullish
    analysis_df.loc[analysis_df['McClellan Summation Index'] < -2000, 'Signal'] = -1  # Bearish
    
    # Add rate of change
    analysis_df['MSI_Change'] = analysis_df['McClellan Summation Index'].diff()
    
    # Add moving averages for trend confirmation
    analysis_df['MSI_MA20'] = analysis_df['McClellan Summation Index'].rolling(window=20).mean()
    analysis_df['MSI_MA50'] = analysis_df['McClellan Summation Index'].rolling(window=50).mean()
    
    return analysis_df

def plot_mcclellan_summation(df):
    """Plot McClellan Summation Index with Advance/Decline Data"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    # Plot Advancing and Declining
    ax1.plot(df.index, df['advancing'], label="Advancing", color='green')
    ax1.plot(df.index, df['declining'], label="Declining", color='red')
    ax1.set_title("Advancing and Declining Stocks")
    ax1.set_ylabel("Number of Stocks")
    ax1.grid(True)
    ax1.legend()
    
    # Plot McClellan Oscillator
    ax2.plot(df.index, df["McClellan Oscillator"], 
             label="McClellan Oscillator", color='orange')
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_title("McClellan Oscillator")
    ax2.set_ylabel("Oscillator Value")
    ax2.grid(True)
    ax2.legend()
    
    # Plot McClellan Summation Index with Moving Averages
    ax3.plot(df.index, df["McClellan Summation Index"], 
             label="McClellan Summation Index", color="purple")
    ax3.plot(df.index, df["MSI_MA20"], 
             label="20-day MA", color="blue", alpha=0.6)
    ax3.plot(df.index, df["MSI_MA50"], 
             label="50-day MA", color="red", alpha=0.6)
    ax3.axhline(y=1000, color='g', linestyle='--', label="Bullish Threshold")
    ax3.axhline(y=-1000, color='r', linestyle='--', label="Bearish Threshold")
    ax3.set_title("McClellan Summation Index")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Summation Index Value")
    ax3.grid(True)
    ax3.legend()
    
    # Set x-axis limits to prevent black areas
    min_date = df.index.min()
    max_date = df.index.max()
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(min_date, max_date)
    
    plt.tight_layout()
    plt.show()

def print_initial_values(df, n=100):
    """Print the first n values of the McClellan analysis"""
    print(f"\nFirst {n} values of McClellan Analysis:")
    print("\n{:^10} | {:^9} | {:^9} | {:^11} | {:^7} | {:^7} | {:^10} | {:^14}".format(
        "Date", "Advancing", "Declining", "Net Advances", "EMA_19", "EMA_39", "Oscillator", "Sum Index"
    ))
    print("-" * 90)
    
    # Get first n rows
    initial_data = df.head(n)
    
    # Print formatted data
    for idx, row in initial_data.iterrows():
        print("{:10} | {:9,d} | {:9,d} | {:11.0f} | {:7.2f} | {:7.2f} | {:10.2f} | {:14.2f}".format(
            idx.strftime('%Y-%m-%d'),
            int(row['advancing']),
            int(row['declining']),
            row['breadth'],
            row['EMA_19'] if not np.isnan(row['EMA_19']) else 0.0,
            row['EMA_39'] if not np.isnan(row['EMA_39']) else 0.0,
            row['McClellan Oscillator'],
            row['McClellan Summation Index']
        ))

def print_last_values(df, n=50):
    """Print the last n values of the McClellan analysis"""
    print(f"\nLast {n} values of McClellan Analysis:")
    print("\n{:^10} | {:^9} | {:^9} | {:^11} | {:^7} | {:^7} | {:^10} | {:^14}".format(
        "Date", "Advancing", "Declining", "Net Advances", "EMA_19", "EMA_39", "Oscillator", "Sum Index"
    ))
    print("-" * 90)
    
    # Get last n rows
    final_data = df.tail(n)
    
    # Print formatted data
    for idx, row in final_data.iterrows():
        print("{:10} | {:9,d} | {:9,d} | {:11.0f} | {:7.2f} | {:7.2f} | {:10.2f} | {:14.2f}".format(
            idx.strftime('%Y-%m-%d'),
            int(row['advancing']),
            int(row['declining']),
            row['breadth'],
            row['EMA_19'] if not np.isnan(row['EMA_19']) else 0.0,
            row['EMA_39'] if not np.isnan(row['EMA_39']) else 0.0,
            row['McClellan Oscillator'],
            row['McClellan Summation Index']
        ))

def main():
    """Main function to run the McClellan Summation Index analysis"""
    def parse_date(date_str):
        try:
            return datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"Invalid date format. Use YYYY-MM-DD: {str(e)}")
    
    parser = argparse.ArgumentParser(description='Calculate McClellan Summation Index from advance/decline data')
    parser.add_argument('--start-date', type=parse_date, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=parse_date, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output-csv', type=str, help='Save results to CSV file')
    parser.add_argument('--print-rows', type=int, default=50, 
                       help='Number of last rows to print (default: 50)')
    parser.add_argument('--ratio-adjusted', action='store_true', 
                       help='Use ratio-adjusted breadth calculation')
    
    args = parser.parse_args()
    
    try:
        # Get advance/decline data from database
        logger.info("Fetching advance/decline data...")
        advn_df = get_price_history_from_db('$ADVN', args.start_date, args.end_date)
        print(advn_df.tail(50))
        decn_df = get_price_history_from_db('$DECN', args.start_date, args.end_date)
        
        if advn_df is not None and decn_df is not None and not advn_df.empty and not decn_df.empty:
            # Calculate McClellan Summation Index
            logger.info("Calculating McClellan Summation Index...")
            df = calculate_mcclellan_summation(advn_df, decn_df, args.ratio_adjusted)
            
            # Add analysis
            logger.info("Performing technical analysis...")
            #df = analyze_mcclellan(df)
            
            # Print initial values
            print_initial_values(df)
            
            # Print last values
            print_last_values(df, args.print_rows)
            
            # Plot results
            logger.info("Generating plots...")
            #plot_mcclellan_summation(df)
            
            # Save to CSV if requested
            if args.output_csv:
                df.to_csv(args.output_csv)
                logger.info(f"Results saved to {args.output_csv}")
            
            # # Display summary statistics
            # logger.info("\nSummary Statistics:")
            # logger.info(f"Date Range: {df.index.min()} to {df.index.max()}")
            # logger.info(f"Number of records: {len(df)}")
            # logger.info(f"Latest Summation Index: {df['McClellan Summation Index'].iloc[-1]:.2f}")
            # logger.info(f"Latest McClellan Oscillator: {df['McClellan Oscillator'].iloc[-1]:.2f}")
            # logger.info(f"Latest Net Advances: {df['breadth'].iloc[-1]:.0f}")
            # logger.info(f"Latest Signal: {df['Signal'].iloc[-1]}")
            
            # # Market condition analysis
            # latest_msi = df['McClellan Summation Index'].iloc[-1]
            # if latest_msi > 1000:
            #     logger.info("Market Condition: BULLISH - Strong positive momentum")
            # elif latest_msi < -1000:
            #     logger.info("Market Condition: BEARISH - Strong negative momentum")
            # else:
            #     logger.info("Market Condition: NEUTRAL - No clear trend")
            
        else:
            logger.error("No data retrieved from database")
            
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()