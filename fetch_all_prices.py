"""
Unified script to fetch historical prices for BTC, ETH, and XRP
from January 1, 2022 to present. (DONE)
"""
from binance.client import Client
from datetime import datetime, timedelta
import pandas as pd
import time
import os
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
    
    coin_name = symbol.replace('USDT', '')
    print(f"\n{'='*60}")
    print(f"Fetching {coin_name} prices from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    print(f"{'='*60}")
    
    # Binance allows fetching up to 1000 klines per request
    # We'll fetch in chunks to avoid rate limits
    request_count = 0
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
                            f"{coin_name}_Price": close_price
                        })
                
                # Update current_date to the day after the last fetched date
                last_timestamp = datetime.fromtimestamp(klines[-1][0] / 1000)
                current_date = last_timestamp + timedelta(days=1)
                request_count += 1
                
                # Progress indicator
                if request_count % 2 == 0:
                    print(f"  Progress: Fetched {len(price_rec)} days so far...")
            else:
                # No more data available
                break
            
            # Rate limiting: wait a bit between requests
            time.sleep(0.1)
            
        except Exception as e:
            print(f"  ⚠️  Error fetching data for {current_date}: {e}")
            # Skip to next day on error
            current_date += timedelta(days=1)
            time.sleep(1)  # Wait longer on error
    
    price_df = pd.DataFrame(price_rec)
    
    # Remove duplicates and sort by date
    if not price_df.empty:
        price_df = price_df.drop_duplicates(subset=['Date'])
        price_df = price_df.sort_values('Date')
    
    return price_df

# Main execution
if __name__ == "__main__":
    print("\n" + "="*60)
    print("CRYPTONE - Historical Price Data Fetcher")
    print("Fetching data from January 1, 2022 to present")
    print("="*60)
    
    # Define coins to fetch
    coins = [
        ('BTCUSDT', 'BTC'),
        ('ETHUSDT', 'ETH'),
        ('XRPUSDT', 'XRP')
    ]
    
    start_date = '2020-01-01'
    results = {}
    
    # Create output directory if it doesn't exist
    output_dir = "dataset/price"
    os.makedirs(output_dir, exist_ok=True)
    
    # Fetch prices for each coin
    for symbol, coin_name in coins:
        try:
            df = fetch_historical_prices(symbol, start_date)
            results[coin_name] = df
            
            # Print summary
            if not df.empty:
                print(f"\n✅ {coin_name}: Fetched {len(df)} days")
                print(f"   Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
                
                # Save to CSV in dataset/price directory
                output_file = os.path.join(output_dir, f"{coin_name.lower()}_prices_2022_present.csv")
                df.to_csv(output_file, index=False)
                print(f"   💾 Saved to: {output_file}")
            else:
                print(f"\n❌ {coin_name}: No data fetched")
                
        except Exception as e:
            print(f"\n❌ {coin_name}: Error - {e}")
        
        # Wait between coins to avoid rate limits
        print("\n⏳ Waiting 2 seconds before next coin...")
        time.sleep(2)
    
    # Final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for coin_name, df in results.items():
        if not df.empty:
            print(f"  {coin_name}: {len(df)} days of data")
    print("="*60)
    print(f"\n✅ All done! Historical price data saved to {output_dir}/")
    print("   Files ready for model training.\n")

