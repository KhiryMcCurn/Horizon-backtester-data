"""
Update Market Data Script
Downloads latest stock data and pushes to HuggingFace Space
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
from huggingface_hub import HfApi
import time

# Configuration
REPO_ID = "JakeFake222/horizon-backtester"
DATA_DIR = "data"

# All tickers to update
TICKERS = [
    # Tech stocks
    'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMZN', 'NFLX', 
    'NET', 'PLTR', 'SPOT', 'COIN', 'CRWD', 'FUTU', 'HOOD',
    # Finance
    'BLK', 'GS', 'V', 'RY',
    # Retail/Consumer
    'COST', 'WMT', 'UBER', 'TTWO',
    # ETFs and Funds
    'SPY', 'QQQ', 'VTI', 'VOO', 'VXUS', 'DIA', 'IWM', 'TQQQ', 
    'BND', 'TLT', 'IEF', 'GLD', 'DBC',
    # International
    'SFTBY', 'HWM'
]


def main():
    print(f"=== Market Data Update Started ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tickers to update: {len(TICKERS)}")
    
    # Create data directory
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Date range: 5 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    success = 0
    failed = []
    
    # Download in batches to avoid rate limits
    batch_size = 5
    for i in range(0, len(TICKERS), batch_size):
        batch = TICKERS[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(TICKERS) + batch_size - 1) // batch_size
        
        print(f"\nBatch {batch_num}/{total_batches}: {', '.join(batch)}")
        
        try:
            # Download batch
            data = yf.download(
                batch,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                auto_adjust=True,
                threads=False,
                progress=False
            )
            
            if not data.empty:
                # Save each ticker
                for ticker in batch:
                    try:
                        # Handle single vs multiple tickers
                        if len(batch) == 1:
                            ticker_data = pd.DataFrame({
                                'Open': data['Open'],
                                'High': data['High'],
                                'Low': data['Low'],
                                'Close': data['Close'],
                                'Volume': data['Volume']
                            })
                        else:
                            if ticker in data['Close'].columns:
                                ticker_data = pd.DataFrame({
                                    'Open': data['Open'][ticker],
                                    'High': data['High'][ticker],
                                    'Low': data['Low'][ticker],
                                    'Close': data['Close'][ticker],
                                    'Volume': data['Volume'][ticker]
                                })
                            else:
                                failed.append(ticker)
                                continue
                        
                        ticker_data = ticker_data.dropna()
                        if not ticker_data.empty:
                            filepath = os.path.join(DATA_DIR, f'{ticker}.csv')
                            ticker_data.to_csv(filepath)
                            success += 1
                            print(f"  ✅ {ticker}: {len(ticker_data)} rows")
                        else:
                            failed.append(ticker)
                            print(f"  ⚠️ {ticker}: No data")
                    except Exception as e:
                        failed.append(ticker)
                        print(f"  ❌ {ticker}: {str(e)[:50]}")
            else:
                failed.extend(batch)
                print(f"  ❌ Batch failed: No data returned")
            
            # Delay between batches
            if i + batch_size < len(TICKERS):
                time.sleep(2)
                
        except Exception as e:
            failed.extend(batch)
            print(f"  ❌ Batch error: {str(e)[:100]}")
    
    # Save timestamp
    timestamp_file = os.path.join(DATA_DIR, 'last_update.txt')
    with open(timestamp_file, 'w') as f:
        f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    print(f"\n=== Download Complete ===")
    print(f"Success: {success}/{len(TICKERS)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    
    # Push to HuggingFace Space
    print(f"\n=== Pushing to HuggingFace Space ===")
    try:
        token = os.environ.get('HF_TOKEN')
        if not token:
            print("❌ HF_TOKEN not found in environment variables")
            return
        
        api = HfApi(token=token)
        
        # Upload all CSV files
        csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        print(f"Uploading {len(csv_files)} files...")
        
        for csv_file in csv_files:
            filepath = os.path.join(DATA_DIR, csv_file)
            api.upload_file(
                path_or_fileobj=filepath,
                path_in_repo=f'data/{csv_file}',
                repo_id=REPO_ID,
                repo_type='space'
            )
        
        # Upload timestamp
        api.upload_file(
            path_or_fileobj=timestamp_file,
            path_in_repo='data/last_update.txt',
            repo_id=REPO_ID,
            repo_type='space'
        )
        
        print(f"✅ Pushed {len(csv_files)} files to {REPO_ID}")
        
    except Exception as e:
        print(f"❌ Push failed: {str(e)}")
    
    print(f"\n=== Update Complete ===")


if __name__ == "__main__":
    main()
