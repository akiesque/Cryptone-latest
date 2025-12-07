from binance.client import Client
from datetime import datetime, timedelta
import pandas as pd
import time
import streamlit as st

# Set up the Binance client with your API Key and Secret
API_KEY = st.secrets["BINANCE_API_KEY"]
API_SECRET = st.secrets["BINANCE_API_SECRET"]
client = Client(API_KEY, API_SECRET)

# Function to fetch historical data from a start date to end date
def fetch_historical_prices(symbol, start_date, end_date=None):
    """
    Fetch historical daily prices from Binance.
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        start_date: Start date as string 'YYYY-MM-DD' or datetime
        end_date: End date as string 'YYYY-MM-DD' or datetime (default: today)
    
    Returns:
        DataFrame with columns: Date, Price
    """
    # Convert start_date to datetime if string
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    # Set end_date to today if not provided
    if end_date is None:
        end_date = datetime.now()
    elif isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    price_rec = []
    current_date = start_date
    
    print(f"Fetching {symbol} prices from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    
    # Binance allows fetching up to 1000 klines per request
    # We'll fetch in chunks to avoid rate limits
    while current_date <= end_date:
        try:
            # Fetch up to 1000 days at a time
            formatted_date = current_date.strftime('%d %b, %Y')
            klines = client.get_historical_klines(
                symbol, 
                Client.KLINE_INTERVAL_1DAY, 
                start_str=formatted_date,
                limit=1000  # Maximum allowed by Binance
            )
            
            if klines:
                for kline in klines:
                    # kline format: [timestamp, open, high, low, close, volume, ...]
                    timestamp = datetime.fromtimestamp(kline[0] / 1000)
                    close_price = float(kline[4])
                    
                    # Only add if within our date range
                    if timestamp.date() <= end_date.date():
                        price_rec.append({
                            "Date": timestamp.strftime('%m/%d/%Y'),
                            "BTC_Price": close_price
                        })
                
                # Update current_date to the day after the last fetched date
                last_timestamp = datetime.fromtimestamp(klines[-1][0] / 1000)
                current_date = last_timestamp + timedelta(days=1)
            else:
                # No more data available
                break
            
            # Rate limiting: wait a bit between requests
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error fetching data for {current_date}: {e}")
            # Skip to next day on error
            current_date += timedelta(days=1)
            time.sleep(1)  # Wait longer on error
    
    price_df = pd.DataFrame(price_rec)
    
    # Remove duplicates and sort by date
    if not price_df.empty:
        price_df = price_df.drop_duplicates(subset=['Date'])
        price_df = price_df.sort_values('Date')
    
    return price_df

# Fetch Bitcoin prices from January 1, 2022 to present
print("Starting Bitcoin price data fetch...")
btc_df = fetch_historical_prices('BTCUSDT', '2022-01-01')

# Print summary
print(f"\nFetched {len(btc_df)} days of Bitcoin price data")
if not btc_df.empty:
    print(f"Date range: {btc_df['Date'].iloc[0]} to {btc_df['Date'].iloc[-1]}")
    print(f"\nFirst few rows:")
    print(btc_df.head())
    print(f"\nLast few rows:")
    print(btc_df.tail())

# Export to CSV
output_file = "btc_prices_2022_present.csv"
btc_df.to_csv(output_file, index=False)
print(f"\n✅ Data saved to {output_file}")
