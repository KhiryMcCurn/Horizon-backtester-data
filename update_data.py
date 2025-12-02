"""
DATA UPDATE SCRIPT FOR HORIZON BACKTESTER
Downloads market data from Yahoo Finance and saves to data/ folder.
Upload to HuggingFace is handled by git push in the workflow.
"""

import yfinance as yf
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from huggingface_hub import hf_hub_download, login
import sys

# =============================================================================
# TICKER UNIVERSE
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
    
    # Nasdaq 100 additions
    'AZN', 'APP', 'ARM', 'CCEP', 'DASH', 'DDOG', 'GFS', 'MELI', 'PDD', 'TEAM', 'WDAY',
    
    # ETFs and Benchmarks
    'SPY', 'QQQ', 'VTI', 'VOO', 'VXUS', 'DIA', 'IWM', 'TQQQ', 'SQQQ', 'BND', 'TLT', 'IEF', 'GLD', 'DBC', 'VNQ',
    'XLF', 'XLK', 'XLE', 'XLV', 'ARKK', 'ARKW', 'SOXL', 'SOXS', 'UPRO', 'SPXU', 'VEA', 'VWO', 'EFA', 'EEM',
    'AGG', 'LQD', 'HYG', 'SHY', 'IEMG', 'VIG', 'SCHD', 'VYM', 'DVY', 'JEPI',
    
    # Notable IPOs 2020-2024
    'RIVN', 'LCID', 'RBLX', 'COIN', 'HOOD', 'SNOW', 'U', 'CPNG', 'COUR', 'OSCR', 'SOFI', 'UPST', 'AFRM', 'PATH',
    'BROS', 'DUOL', 'ASAN', 'FVRR', 'DOCS', 'DNUT', 'YOU', 'AI', 'DLO', 'JAMF', 'NCNO', 'JMIA', 
    'DKNG', 'NKLA', 'BLNK', 'QS', 'GOEV', 'LAZR', 'LMND', 'OPEN',
    'TPG', 'CRDO', 'MBLY', 'ACLX', 'BLTE', 'GCT', 'IE', 'CRBG',
    'BIRK', 'CART', 'KVYO', 'CAVA', 'ATMU', 'KGS', 'ODD', 'APGE', 'GPCR', 'ENLT', 'NXT',
    'RDDT', 'ALAB', 'VIK', 'LOAR', 'RBRK', 'AHR', 'IBTA', 'TEM', 'WAY', 'NNE',
    
    # Horizon specific
    'FUTU', 'SFTBY', 'RY', 'NET'
]

TICKERS = sorted(list(set(TICKERS)))

# Configuration
REPO_ID = "JakeFake222/horizon-backtester"
REPO_TYPE = "space"
DATA_DIR = "data"
BATCH_SIZE = 50
BATCH_DELAY = 5
MAX_RETRIES = 2


def fetch_requested_tickers(token):
    """Fetch requests.txt from HuggingFace and return list of requested tickers."""
    try:
        login(token=token)
        filepath = hf_hub_download(
            repo_id=REPO_ID,
            filename="requests.txt",
            repo_type=REPO_TYPE
        )
        with open(filepath, 'r') as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
        return list(set(tickers))
    except Exception as e:
        print(f"ℹ️  No requests.txt found: {e}")
        return []


def download_ticker_data(ticker, start_date, end_date, retry_count=0):
    """Download historical data for a single ticker from Yahoo Finance."""
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
        
        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in expected_cols):
            print(f"   ⚠️  {ticker}: Missing expected columns")
            return None
            
        return data[expected_cols]
        
    except Exception as e:
        if retry_count < MAX_RETRIES:
            time.sleep(2)
            return download_ticker_data(ticker, start_date, end_date, retry_count + 1)
        print(f"❌ Error downloading {ticker}: {e}")
        return None


def download_batch(tickers_batch, start_date, end_date, batch_num, total_batches):
    """Download a batch of tickers."""
    print(f"\n📦 Batch {batch_num}/{total_batches} ({len(tickers_batch)} tickers)")
    
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
    """Download and save all ticker data."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    print(f"\n{'='*70}")
    print(f"🚀 HORIZON BACKTESTER - DATA DOWNLOAD")
    print(f"{'='*70}")
    print(f"📅 Date range: {start_date.date()} to {end_date.date()}")
    print(f"📊 Total tickers: {len(TICKERS)}")
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
            print(f"   ⏳ Waiting {BATCH_DELAY}s...")
            time.sleep(BATCH_DELAY)
    
    # Write timestamp
    with open(os.path.join(DATA_DIR, "last_update.txt"), 'w') as f:
        f.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        f.write(f"Successful: {success_count}\n")
        f.write(f"Failed: {error_count}\n")
    
    print(f"\n{'='*70}")
    print(f"✅ DOWNLOAD COMPLETE: {success_count} successful, {error_count} failed")
    print(f"{'='*70}\n")
    
    return success_count, error_count


def main():
    global TICKERS
    
    hf_token = os.environ.get("HF_TOKEN")
    
    # Check for requested tickers
    if hf_token:
        print(f"\n📥 Checking for ticker requests...")
        requested = fetch_requested_tickers(hf_token)
        if requested:
            print(f"   Found {len(requested)} requested ticker(s)")
            TICKERS = sorted(list(set(TICKERS + requested)))
    
    success_count, error_count = update_all_tickers()
    
    if success_count > 0:
        print("🎉 Data download complete! Workflow will commit and push to HuggingFace.")
        sys.exit(0)
    else:
        print("❌ No data downloaded")
        sys.exit(1)


if __name__ == "__main__":
    main()
