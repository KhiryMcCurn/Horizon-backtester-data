"""
DATA UPDATE SCRIPT FOR HORIZON BACKTESTER
Automatically updates all ticker CSV files with latest market data from Yahoo Finance
This script is designed to run as a scheduled job at end of trading day

Features:
- Batched downloads to avoid rate limiting
- Batched uploads to avoid timeout
- Delays between batches
- Retry logic for failed downloads
- Supports ~600 tickers (S&P 500 + Nasdaq 100 + recent IPOs + ETFs)
"""

import yfinance as yf
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from huggingface_hub import HfApi, login
import sys

# =============================================================================
# TICKER UNIVERSE
# S&P 500 + Nasdaq 100 + Recent IPOs (2022-2024) + Notable 2020-2021 IPOs + ETFs
# =============================================================================

TICKERS = [
    # S&P 500 Components
    'A', 'AAL', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP',
    'AES', 'AFL', 'AIG', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALL', 'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMGN',
    'AMP', 'AMT', 'AMZN', 'ANET', 'AON', 'AOS', 'APA', 'APD', 'APH', 'APTV', 'ARE', 'ATO', 'AVB', 'AVGO',
    'AVY', 'AWK', 'AXON', 'AXP', 'AZO', 'BA', 'BAC', 'BALL', 'BAX', 'BBWI', 'BBY', 'BDX', 'BEN', 'BF-B', 'BG',
    'BIIB', 'BIO', 'BK', 'BKNG', 'BKR', 'BLDR', 'BLK', 'BMY', 'BR', 'BRK-B', 'BRO', 'BSX', 'BWA', 'BX', 'BXP',
    'C', 'CAG', 'CAH', 'CARR', 'CAT', 'CB', 'CBOE', 'CBRE', 'CCI', 'CCL', 'CDNS', 'CDW', 'CE', 'CEG', 'CF',
    'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CINF', 'CL', 'CLX', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP',
    'COF', 'COO', 'COP', 'COR', 'COST', 'CPAY', 'CPB', 'CPRT', 'CPT', 'CRL', 'CRM', 'CRWD', 'CSCO', 'CSGP', 'CSX',
    'CTAS', 'CTRA', 'CTSH', 'CTVA', 'CVS', 'CVX', 'CZR', 'D', 'DAL', 'DAY', 'DD', 'DE', 'DECK',
    'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DLR', 'DLTR', 'DOC', 'DOV', 'DOW', 'DPZ', 'DRI', 'DTE', 'DUK', 'DVA',
    'DVN', 'DXCM', 'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EG', 'EIX', 'EL', 'ELV', 'EMN', 'EMR', 'ENPH', 'EOG',
    'EPAM', 'EQIX', 'EQR', 'EQT', 'ERIE', 'ES', 'ESS', 'ETN', 'ETR', 'EVRG', 'EW', 'EXC', 'EXPD', 'EXPE', 'EXR',
    'F', 'FANG', 'FAST', 'FCX', 'FDS', 'FDX', 'FE', 'FFIV', 'FI', 'FICO', 'FIS', 'FITB', 'FMC', 'FOX', 'FOXA',
    'FRT', 'FSLR', 'FTNT', 'FTV', 'GD', 'GDDY', 'GE', 'GEHC', 'GEN', 'GEV', 'GILD', 'GIS', 'GL', 'GLW', 'GM',
    'GNRC', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS', 'GWW', 'HAL', 'HAS', 'HBAN', 'HCA', 'HD', 'HIG',
    'HII', 'HLT', 'HOLX', 'HON', 'HPE', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUBB', 'HUM', 'HWM', 'IBM', 'ICE',
    'IDXX', 'IEX', 'IFF', 'INCY', 'INTC', 'INTU', 'INVH', 'IP', 'IPG', 'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW',
    'IVZ', 'J', 'JBHT', 'JBL', 'JCI', 'JKHY', 'JNJ', 'JPM', 'K', 'KDP', 'KEY', 'KEYS', 'KHC', 'KIM',
    'KKR', 'KLAC', 'KMB', 'KMI', 'KMX', 'KO', 'KR', 'KVUE', 'L', 'LDOS', 'LEN', 'LH', 'LHX', 'LIN', 'LKQ',
    'LLY', 'LMT', 'LNT', 'LOW', 'LRCX', 'LULU', 'LUV', 'LVS', 'LW', 'LYB', 'LYV', 'MA', 'MAA', 'MAR', 'MAS',
    'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'META', 'MGM', 'MHK', 'MKC', 'MKTX', 'MLM', 'MMC', 'MMM',
    'MNST', 'MO', 'MOH', 'MOS', 'MPC', 'MPWR', 'MRK', 'MRNA', 'MRVL', 'MS', 'MSCI', 'MSFT', 'MSI', 'MTB', 'MTCH',
    'MTD', 'MU', 'NCLH', 'NDAQ', 'NDSN', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'NOC', 'NOW', 'NRG', 'NSC', 'NTAP',
    'NTRS', 'NUE', 'NVDA', 'NVR', 'NWS', 'NWSA', 'NXPI', 'O', 'ODFL', 'OKE', 'OMC', 'ON', 'ORCL', 'ORLY', 'OTIS',
    'OXY', 'PANW', 'PAYC', 'PAYX', 'PCAR', 'PCG', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM',
    'PKG', 'PLD', 'PLTR', 'PM', 'PNC', 'PNR', 'PNW', 'PODD', 'POOL', 'PPG', 'PPL', 'PRU', 'PSA', 'PSX', 'PTC',
    'PWR', 'PYPL', 'QCOM', 'QRVO', 'RCL', 'REG', 'REGN', 'RF', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST',
    'RSG', 'RTX', 'RVTY', 'SBAC', 'SBUX', 'SCHW', 'SHW', 'SJM', 'SLB', 'SMCI', 'SNA', 'SNPS', 'SO', 'SOLV', 'SPG',
    'SPGI', 'SRE', 'STE', 'STLD', 'STT', 'STX', 'STZ', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T', 'TAP', 'TDG',
    'TDY', 'TECH', 'TEL', 'TER', 'TFC', 'TFX', 'TGT', 'TJX', 'TMO', 'TMUS', 'TPR', 'TRGP', 'TRMB', 'TROW', 'TRV',
    'TSCO', 'TSLA', 'TSN', 'TT', 'TTWO', 'TXN', 'TXT', 'TYL', 'UAL', 'UBER', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNP',
    'UPS', 'URI', 'USB', 'V', 'VICI', 'VLO', 'VLTO', 'VMC', 'VRSK', 'VRSN', 'VRTX', 'VST', 'VTR', 'VTRS', 'VZ',
    'WAB', 'WAT', 'WBD', 'WDC', 'WEC', 'WELL', 'WFC', 'WM', 'WMB', 'WMT', 'WRB', 'WST', 'WTW', 'WY',
    'WYNN', 'XEL', 'XOM', 'XYL', 'YUM', 'ZBH', 'ZBRA', 'ZTS',
    
    # Nasdaq 100 additions (not in S&P 500)
    'AZN', 'APP', 'ARM', 'CCEP', 'DASH', 'DDOG', 'GFS', 'MELI', 'PDD', 'TEAM', 'WDAY',
    
    # ETFs and Benchmarks
    'SPY', 'QQQ', 'VTI', 'VOO', 'VXUS', 'DIA', 'IWM', 'TQQQ', 'SQQQ', 'BND', 'TLT', 'IEF', 'GLD', 'DBC', 'VNQ',
    'XLF', 'XLK', 'XLE', 'XLV', 'ARKK', 'ARKW', 'SOXL', 'SOXS', 'UPRO', 'SPXU', 'VEA', 'VWO', 'EFA', 'EEM',
    'AGG', 'LQD', 'HYG', 'SHY', 'IEMG', 'VIG', 'SCHD', 'VYM', 'DVY', 'JEPI',
    
    # Notable 2020-2021 IPOs
    'RIVN', 'LCID', 'RBLX', 'COIN', 'HOOD', 'SNOW', 'U', 'CPNG', 'COUR', 'OSCR', 'SOFI', 'UPST', 'AFRM', 'PATH',
    'BROS', 'DUOL', 'ASAN', 'FVRR', 'DOCS', 'DNUT', 'YOU', 'AI', 'DLO', 'JAMF', 'NCNO', 'JMIA', 
    'DKNG', 'NKLA', 'BLNK', 'QS', 'GOEV', 'LAZR', 'LMND', 'OPEN',
    
    # 2022 IPOs (notable)
    'TPG', 'CRDO', 'MBLY', 'ACLX', 'BLTE', 'GCT', 'IE', 'CRBG',
    
    # 2023 IPOs (notable)
    'BIRK', 'CART', 'KVYO', 'CAVA', 'ATMU', 'KGS', 'ODD', 'APGE', 'GPCR', 'ENLT', 'NXT',
    
    # 2024 IPOs (notable)
    'RDDT', 'ALAB', 'VIK', 'LOAR', 'RBRK', 'AHR', 'IBTA', 'TEM', 'WAY', 'NNE',
    
    # Diamond Horizon specific (ensuring these are included)
    'FUTU', 'SFTBY', 'RY', 'NET'
]

# Remove duplicates and sort
TICKERS = sorted(list(set(TICKERS)))

# Repository configuration
REPO_ID = "JakeFake222/horizon-backtester"
REPO_TYPE = "space"
DATA_DIR = "data"

# Rate limiting configuration
BATCH_SIZE = 50  # Download 50 tickers at a time
BATCH_DELAY = 5  # Wait 5 seconds between batches
MAX_RETRIES = 2  # Retry failed downloads up to 2 times
UPLOAD_BATCH_SIZE = 100  # Upload 100 files at a time


def download_ticker_data(ticker, start_date, end_date, retry_count=0):
    """
    Download historical data for a single ticker from Yahoo Finance
    """
    try:
        data = yf.download(
            ticker, 
            start=start_date, 
            end=end_date,
            auto_adjust=True,
            progress=False,
            threads=False
        )
        
        if data.empty:
            return None
            
        return data
        
    except Exception as e:
        if retry_count < MAX_RETRIES:
            time.sleep(2)
            return download_ticker_data(ticker, start_date, end_date, retry_count + 1)
        print(f"❌ Error downloading {ticker} after {MAX_RETRIES} retries: {e}")
        return None


def download_batch(tickers_batch, start_date, end_date, batch_num, total_batches):
    """
    Download a batch of tickers
    """
    print(f"\n📦 Batch {batch_num}/{total_batches} ({len(tickers_batch)} tickers)")
    print(f"   Tickers: {', '.join(tickers_batch[:10])}{'...' if len(tickers_batch) > 10 else ''}")
    
    results = {}
    
    for ticker in tickers_batch:
        data = download_ticker_data(ticker, start_date, end_date)
        if data is not None and not data.empty:
            results[ticker] = data
            print(f"   ✅ {ticker}: {len(data)} rows")
        else:
            print(f"   ⚠️  {ticker}: no data")
    
    return results


def update_all_tickers():
    """
    Update CSV files for all tickers with batching and rate limiting
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    print(f"\n{'='*70}")
    print(f"🚀 HORIZON BACKTESTER - BULK DATA UPDATE")
    print(f"{'='*70}")
    print(f"📅 Date range: {start_date.date()} to {end_date.date()}")
    print(f"📊 Total tickers: {len(TICKERS)}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    print(f"⏱️  Delay between batches: {BATCH_DELAY}s")
    print(f"{'='*70}")
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    batches = [TICKERS[i:i + BATCH_SIZE] for i in range(0, len(TICKERS), BATCH_SIZE)]
    total_batches = len(batches)
    
    success_count = 0
    error_count = 0
    
    for batch_num, batch in enumerate(batches, 1):
        results = download_batch(batch, start_date, end_date, batch_num, total_batches)
        
        for ticker, data in results.items():
            filepath = os.path.join(DATA_DIR, f"{ticker}.csv")
            data.to_csv(filepath)
            success_count += 1
        
        error_count += len(batch) - len(results)
        
        if batch_num < total_batches:
            print(f"   ⏳ Waiting {BATCH_DELAY}s before next batch...")
            time.sleep(BATCH_DELAY)
    
    # Write timestamp file
    timestamp_file = os.path.join(DATA_DIR, "last_update.txt")
    with open(timestamp_file, 'w') as f:
        f.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        f.write(f"Total tickers: {len(TICKERS)}\n")
        f.write(f"Successful: {success_count}\n")
        f.write(f"Failed: {error_count}\n")
    
    print(f"\n{'='*70}")
    print(f"✅ DATA UPDATE COMPLETE")
    print(f"{'='*70}")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {error_count}")
    print(f"📁 Data saved to: {DATA_DIR}/")
    print(f"{'='*70}\n")
    
    return success_count, error_count


def push_to_huggingface(token):
    """
    Push updated CSV files to Hugging Face repository in batches
    """
    try:
        print(f"\n{'='*70}")
        print("🚀 PUSHING TO HUGGING FACE (batched uploads)")
        print(f"{'='*70}\n")
        
        login(token=token)
        api = HfApi()
        
        # Get list of all files to upload
        all_files = []
        for filename in os.listdir(DATA_DIR):
            filepath = os.path.join(DATA_DIR, filename)
            if os.path.isfile(filepath):
                all_files.append((filepath, f"data/{filename}"))
        
        print(f"📁 Total files to upload: {len(all_files)}")
        
        # Upload in batches
        total_batches = (len(all_files) + UPLOAD_BATCH_SIZE - 1) // UPLOAD_BATCH_SIZE
        
        for batch_num in range(total_batches):
            start_idx = batch_num * UPLOAD_BATCH_SIZE
            end_idx = min(start_idx + UPLOAD_BATCH_SIZE, len(all_files))
            batch = all_files[start_idx:end_idx]
            
            print(f"\n📤 Upload batch {batch_num + 1}/{total_batches} ({len(batch)} files)")
            
            for filepath, repo_path in batch:
                try:
                    api.upload_file(
                        path_or_fileobj=filepath,
                        path_in_repo=repo_path,
                        repo_id=REPO_ID,
                        repo_type=REPO_TYPE,
                        commit_message=f"Update {os.path.basename(filepath)}"
                    )
                except Exception as e:
                    print(f"   ⚠️  Failed to upload {filepath}: {e}")
            
            print(f"   ✅ Batch {batch_num + 1} complete")
            
            # Small delay between upload batches
            if batch_num < total_batches - 1:
                time.sleep(2)
        
        print(f"\n✅ Successfully pushed to Hugging Face!")
        print(f"{'='*70}\n")
        return True
        
    except Exception as e:
        print(f"❌ Error pushing to Hugging Face: {e}")
        return False


def main():
    """Main execution function"""
    hf_token = os.environ.get("HF_TOKEN")
    
    if not hf_token:
        print("❌ ERROR: HF_TOKEN environment variable not set!")
        sys.exit(1)
    
    print(f"\n🎯 Horizon Backtester - Bulk Data Update")
    print(f"   Universe: {len(TICKERS)} tickers")
    print(f"   Estimated time: ~{(len(TICKERS) // BATCH_SIZE) * BATCH_DELAY // 60 + 10} minutes\n")
    
    success_count, error_count = update_all_tickers()
    
    if success_count > 0:
        push_success = push_to_huggingface(hf_token)
        if push_success:
            print("🎉 Bulk data update job completed successfully!")
            sys.exit(0)
        else:
            print("⚠️  Data downloaded but failed to push to Hugging Face")
            sys.exit(1)
    else:
        print("❌ No data was downloaded - skipping push to Hugging Face")
        sys.exit(1)


if __name__ == "__main__":
    main()
