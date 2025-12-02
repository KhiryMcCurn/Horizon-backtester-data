"""
HORIZON PORTFOLIO BACKTESTER
A Portfolio Visualizer clone built with Gradio
Deploy to Hugging Face Spaces

Features:
- Interactive UI for portfolio backtesting
- REST API endpoint for programmatic access
- Claude AI integration for natural language queries
"""

import gradio as gr
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import json
import os
import re
import httpx
warnings.filterwarnings('ignore')

# Claude API configuration
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-20250514"

# Preset portfolios
PRESETS = {
    "Custom": [],
    "Horizon Growth": [
        'BLK', 'COST', 'GS', 'SPOT', 'META', 'CRWD', 'MSFT', 'V', 'GOOGL', 'AAPL',
        'COIN', 'TTWO', 'AMZN', 'HWM', 'NET', 'NVDA', 'PLTR', 'FUTU', 'RY', 'WMT',
        'HOOD', 'NFLX', 'UBER', 'SFTBY', 'TQQQ'
    ],
    "FAANG": ['META', 'AAPL', 'AMZN', 'NFLX', 'GOOGL'],
    "Magnificent 7": ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'],
    "Classic 60/40": ['VTI', 'BND'],
    "Three Fund Portfolio": ['VTI', 'VXUS', 'BND'],
    "All Weather": ['VTI', 'TLT', 'IEF', 'GLD', 'DBC'],
}

DEFAULT_BENCHMARKS = ['QQQ', 'SPY', 'VTI', 'IWM', 'DIA', 'VOO']

# Local data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def get_available_tickers():
    """Get list of all tickers available in the local data directory"""
    tickers = []
    if os.path.exists(DATA_DIR):
        for filename in os.listdir(DATA_DIR):
            if filename.endswith('.csv'):
                ticker = filename.replace('.csv', '')
                tickers.append(ticker)
    # Sort alphabetically, but put common benchmarks first
    benchmark_order = ['QQQ', 'SPY', 'VTI', 'IWM', 'DIA', 'VOO']
    sorted_tickers = []
    for b in benchmark_order:
        if b in tickers:
            sorted_tickers.append(b)
            tickers.remove(b)
    sorted_tickers.extend(sorted(tickers))
    return sorted_tickers if sorted_tickers else DEFAULT_BENCHMARKS


# Dynamic benchmark list from available data
BENCHMARKS = get_available_tickers()

# Request queue file
REQUESTS_FILE = os.path.join(os.path.dirname(__file__), 'requests.txt')


def is_ticker_available(ticker: str) -> bool:
    """Check if a ticker is available in the local dataset"""
    ticker = ticker.strip().upper()
    filepath = os.path.join(DATA_DIR, f'{ticker}.csv')
    return os.path.exists(filepath)


def request_ticker(ticker: str) -> dict:
    """
    Add a ticker to the request queue.
    Returns status of the request.
    """
    ticker = ticker.strip().upper()
    
    # Validate ticker format
    if not ticker or not ticker.isalpha() or len(ticker) > 5:
        return {
            "success": False,
            "message": f"Invalid ticker format: {ticker}. Use 1-5 letters only."
        }
    
    # Check if already available
    if is_ticker_available(ticker):
        return {
            "success": False,
            "message": f"{ticker} is already available in the dataset!"
        }
    
    # Check if already requested
    pending = get_pending_requests()
    if ticker in pending:
        return {
            "success": False,
            "message": f"{ticker} is already in the request queue."
        }
    
    # Add to request file
    try:
        with open(REQUESTS_FILE, 'a') as f:
            f.write(f"{ticker}\n")
        return {
            "success": True,
            "message": f"✅ {ticker} added to request queue! It will be available after the next data update (daily at ~9:30 PM UTC)."
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error adding request: {e}"
        }


def get_pending_requests() -> list:
    """Get list of pending ticker requests"""
    if not os.path.exists(REQUESTS_FILE):
        return []
    try:
        with open(REQUESTS_FILE, 'r') as f:
            return [line.strip().upper() for line in f if line.strip()]
    except Exception:
        return []


def check_tickers_availability(tickers: list) -> dict:
    """
    Check which tickers are available and which are missing.
    Returns dict with 'available' and 'missing' lists.
    """
    available = []
    missing = []
    for ticker in tickers:
        ticker = ticker.strip().upper()
        if is_ticker_available(ticker):
            available.append(ticker)
        else:
            missing.append(ticker)
    return {"available": available, "missing": missing}


def load_local_data(ticker, start_date, end_date):
    """Load data from local CSV if available"""
    filepath = os.path.join(DATA_DIR, f'{ticker}.csv')
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            # Filter to date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            if not df.empty:
                return df
        except Exception:
            pass
    return None


def get_price_data(tickers, start_date, end_date):
    """
    Get price data - tries local files first, falls back to yfinance.
    Returns DataFrame with Close prices.
    """
    close_data = {}
    missing_tickers = []
    
    # Try local data first
    for ticker in tickers:
        local = load_local_data(ticker, start_date, end_date)
        if local is not None and 'Close' in local.columns:
            close_data[ticker] = local['Close']
        else:
            missing_tickers.append(ticker)
    
    # Download missing tickers from yfinance
    if missing_tickers:
        try:
            data = yf.download(
                missing_tickers, 
                start=start_date, 
                end=end_date, 
                auto_adjust=True, 
                progress=False,
                threads=False
            )
            if not data.empty:
                if len(missing_tickers) == 1:
                    if 'Close' in data.columns:
                        close_data[missing_tickers[0]] = data['Close']
                else:
                    for ticker in missing_tickers:
                        if ticker in data['Close'].columns:
                            close_data[ticker] = data['Close'][ticker]
        except Exception as e:
            print(f"yfinance error: {e}")
    
    # Combine into DataFrame
    if close_data:
        result = pd.DataFrame(close_data)
        result = result.dropna(how='all')
        return result
    return pd.DataFrame()


# ===== CLAUDE API INTEGRATION =====

def call_claude_api(prompt: str, system_prompt: str = None) -> str:
    """Call Claude API to parse natural language backtest requests"""
    
    if not ANTHROPIC_API_KEY:
        return None
    
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    
    system = system_prompt or """You are a portfolio backtesting assistant. 
    Parse user requests into structured backtest parameters.
    Always respond with valid JSON only, no other text."""
    
    data = {
        "model": CLAUDE_MODEL,
        "max_tokens": 1024,
        "system": system,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            result = response.json()
            return result["content"][0]["text"]
    except Exception as e:
        print(f"Claude API error: {e}")
        return None


def parse_natural_language_request(user_request: str) -> dict:
    """Use Claude to parse a natural language backtest request into parameters"""
    
    system_prompt = """You are a portfolio backtesting assistant. Parse the user's request into backtest parameters.

Available presets: "Custom", "Horizon Growth", "FAANG", "Magnificent 7", "Classic 60/40", "Three Fund Portfolio", "All Weather"
Benchmark: Any stock ticker can be used as benchmark (common ones: QQQ, SPY, VTI, IWM, NVDA, AAPL, etc.)
Contribution frequencies: "None", "Weekly", "Monthly", "Quarterly"
Rebalancing frequencies: "None", "Monthly", "Quarterly", "Annually"

Respond ONLY with a JSON object in this exact format (no other text):
{
    "preset": "preset name or Custom",
    "tickers": ["AAPL", "MSFT"] or null if using preset,
    "benchmark": "QQQ",
    "start_date": "2024-01-01" or "1y" or "5y",
    "initial_investment": 10000,
    "contribution_amount": 1000,
    "contribution_freq": "Weekly",
    "rebalance_freq": "None",
    "explanation": "Brief explanation of what you understood"
}

If the user mentions specific stocks, use "Custom" preset and list them in tickers.
If they mention a preset name, use that preset.
Any ticker can be used as benchmark - if user says "compare against NVDA" or "benchmark to AAPL", use that ticker.
Default to reasonable values if not specified."""

    response = call_claude_api(user_request, system_prompt)
    
    if not response:
        # Return defaults if Claude API fails
        return {
            "preset": "Horizon Growth",
            "tickers": None,
            "benchmark": "QQQ",
            "start_date": "2024-01-01",
            "initial_investment": 10000,
            "contribution_amount": 1000,
            "contribution_freq": "Weekly",
            "rebalance_freq": "None",
            "explanation": "Using defaults (Claude API not available)"
        }
    
    try:
        # Clean up response - remove markdown code blocks if present
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```json?\n?', '', cleaned)
            cleaned = re.sub(r'\n?```$', '', cleaned)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "preset": "Horizon Growth",
            "tickers": None,
            "benchmark": "QQQ", 
            "start_date": "2024-01-01",
            "initial_investment": 10000,
            "contribution_amount": 1000,
            "contribution_freq": "Weekly",
            "rebalance_freq": "None",
            "explanation": f"Could not parse response, using defaults. Raw: {response[:200]}"
        }


def run_backtest_from_params(params: dict) -> dict:
    """Run backtest from a parameters dictionary (for API use)"""
    
    preset = params.get("preset", "Custom")
    tickers = params.get("tickers")
    benchmark = params.get("benchmark", "QQQ")
    start_date = params.get("start_date", "2024-01-01")
    initial_investment = params.get("initial_investment", 10000)
    contribution_amount = params.get("contribution_amount", 1000)
    contribution_freq = params.get("contribution_freq", "Weekly")
    rebalance_freq = params.get("rebalance_freq", "None")
    
    # Get tickers
    if preset != "Custom" and preset in PRESETS:
        ticker_list = PRESETS[preset]
    elif tickers:
        ticker_list = [t.strip().upper() for t in tickers] if isinstance(tickers, list) else [t.strip().upper() for t in tickers.split(',')]
    else:
        return {"error": "No tickers specified"}
    
    # Parse dates
    start = parse_date(start_date)
    end = datetime.now().strftime('%Y-%m-%d')
    
    # Download data
    all_tickers = list(set(ticker_list + [benchmark]))
    
    # Use hybrid data loading (local first, then yfinance)
    close = get_price_data(all_tickers, start, end)
    
    if close.empty:
        return {"error": f"No data returned. Tickers: {all_tickers}, Start: {start}, End: {end}"}
    
    # Check we have all needed tickers
    missing = [t for t in ticker_list if t not in close.columns]
    if missing:
        return {"error": f"Missing data for tickers: {missing}. These tickers are not in our dataset. You can request them in the 'Request Ticker' tab."}
    
    trading_dates = close.index
    num_stocks = len(ticker_list)
    
    # Contribution dates
    if contribution_freq == "Weekly":
        contrib_dates = [d for d in trading_dates if d.weekday() == 2]
    elif contribution_freq == "Monthly":
        contrib_dates = close.resample('M').last().index
    elif contribution_freq == "Quarterly":
        contrib_dates = close.resample('Q').last().index
    else:
        contrib_dates = [trading_dates[0]] if len(trading_dates) > 0 else []
    
    # Rebalance dates
    if rebalance_freq == "Monthly":
        rebal_dates = close.resample('M').last().index
    elif rebalance_freq == "Quarterly":
        rebal_dates = close.resample('Q').last().index
    elif rebalance_freq == "Annually":
        rebal_dates = close.resample('Y').last().index
    else:
        rebal_dates = []
    
    # Initialize
    shares = {t: 0.0 for t in ticker_list}
    cost_basis = 0.0
    bench_shares = 0.0
    
    # Initial investment
    if initial_investment > 0 and len(trading_dates) > 0:
        first_date = trading_dates[0]
        per_stock = initial_investment / num_stocks
        for t in ticker_list:
            if t in close.columns and not pd.isna(close.loc[first_date, t]):
                shares[t] += per_stock / close.loc[first_date, t]
        cost_basis += initial_investment
        if benchmark in close.columns:
            bench_shares += initial_investment / close.loc[first_date, benchmark]
    
    # Simulation
    for date in trading_dates:
        if date in contrib_dates and contribution_amount > 0:
            per_stock = contribution_amount / num_stocks
            for t in ticker_list:
                if t in close.columns and not pd.isna(close.loc[date, t]):
                    shares[t] += per_stock / close.loc[date, t]
            cost_basis += contribution_amount
            if benchmark in close.columns and not pd.isna(close.loc[date, benchmark]):
                bench_shares += contribution_amount / close.loc[date, benchmark]
        
        if date in rebal_dates and rebalance_freq != "None":
            total_val = sum(shares[t] * close.loc[date, t] for t in ticker_list 
                          if t in close.columns and not pd.isna(close.loc[date, t]))
            if total_val > 0:
                target = total_val / num_stocks
                for t in ticker_list:
                    if t in close.columns and not pd.isna(close.loc[date, t]):
                        shares[t] = target / close.loc[date, t]
    
    # Final calculations
    last_date = trading_dates[-1]
    final_port = sum(shares[t] * close.loc[last_date, t] for t in ticker_list 
                    if t in close.columns and not pd.isna(close.loc[last_date, t]))
    final_bench = bench_shares * close.loc[last_date, benchmark] if benchmark in close.columns else 0
    
    port_return = (final_port - cost_basis) / cost_basis * 100
    bench_return = (final_bench - cost_basis) / cost_basis * 100
    alpha = port_return - bench_return
    
    # Holdings breakdown
    holdings = {}
    for t in ticker_list:
        if t in close.columns and not pd.isna(close.loc[last_date, t]):
            value = shares[t] * close.loc[last_date, t]
            holdings[t] = {
                "shares": round(shares[t], 4),
                "price": round(close.loc[last_date, t], 2),
                "value": round(value, 2),
                "weight": round(value / final_port * 100, 2) if final_port > 0 else 0
            }
    
    return {
        "success": True,
        "period": {"start": start, "end": end},
        "parameters": {
            "preset": preset,
            "tickers": ticker_list,
            "benchmark": benchmark,
            "initial_investment": initial_investment,
            "contribution_amount": contribution_amount,
            "contribution_freq": contribution_freq,
            "rebalance_freq": rebalance_freq
        },
        "results": {
            "total_invested": round(cost_basis, 2),
            "portfolio_value": round(final_port, 2),
            "benchmark_value": round(final_bench, 2),
            "portfolio_return": round(port_return, 2),
            "benchmark_return": round(bench_return, 2),
            "alpha": round(alpha, 2)
        },
        "holdings": holdings
    }


def natural_language_backtest(user_request: str) -> str:
    """
    Process a natural language backtest request.
    
    Examples:
    - "Backtest the Magnificent 7 for the last 2 years with $500 weekly contributions"
    - "Compare AAPL, MSFT, GOOGL against SPY starting from 2023 with quarterly rebalancing"
    - "Run Horizon Growth portfolio from January 2024 with $10k initial and $1000 monthly DCA"
    
    Returns JSON with backtest results.
    """
    
    if not user_request.strip():
        return json.dumps({"error": "Please provide a backtest request"}, indent=2)
    
    # Parse the request using Claude
    params = parse_natural_language_request(user_request)
    
    # Run the backtest
    results = run_backtest_from_params(params)
    
    # Add the parsed interpretation
    results["interpretation"] = params.get("explanation", "")
    
    return json.dumps(results, indent=2, default=str)


def api_backtest(
    preset: str = "Horizon Growth",
    tickers: str = "",
    benchmark: str = "QQQ",
    start_date: str = "2024-01-01",
    initial_investment: float = 10000,
    contribution_amount: float = 1000,
    contribution_freq: str = "Weekly",
    rebalance_freq: str = "None"
) -> str:
    """
    Programmatic API endpoint for backtesting.
    
    Parameters:
    - preset: Portfolio preset name or "Custom"
    - tickers: Comma-separated tickers (only used if preset is "Custom")
    - benchmark: Benchmark ticker (QQQ, SPY, VTI, etc.)
    - start_date: Start date (YYYY-MM-DD) or relative (1y, 3m, 5y)
    - initial_investment: Initial investment amount
    - contribution_amount: Periodic contribution amount
    - contribution_freq: None, Weekly, Monthly, Quarterly
    - rebalance_freq: None, Monthly, Quarterly, Annually
    
    Returns: JSON string with backtest results
    """
    
    params = {
        "preset": preset,
        "tickers": [t.strip() for t in tickers.split(',')] if tickers and preset == "Custom" else None,
        "benchmark": benchmark,
        "start_date": start_date,
        "initial_investment": initial_investment,
        "contribution_amount": contribution_amount,
        "contribution_freq": contribution_freq,
        "rebalance_freq": rebalance_freq
    }
    
    results = run_backtest_from_params(params)
    return json.dumps(results, indent=2, default=str)


def parse_date(date_str):
    """Parse relative date strings like '3m', '1y', '5y'"""
    today = datetime.now()
    if date_str.endswith('m'):
        months = int(date_str[:-1])
        return (today - timedelta(days=months*30)).strftime('%Y-%m-%d')
    elif date_str.endswith('y'):
        years = int(date_str[:-1])
        return (today - timedelta(days=years*365)).strftime('%Y-%m-%d')
    else:
        return date_str


def calculate_metrics(returns, risk_free_rate=0.04):
    """Calculate comprehensive performance metrics"""
    if len(returns) < 2:
        return {}
    
    # Basic stats
    total_return = (1 + returns).prod() - 1
    cagr = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    
    # Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Risk-adjusted returns
    excess_returns = returns - risk_free_rate/252
    sharpe = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    # Sortino (downside deviation)
    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 0
    sortino = (cagr - risk_free_rate) / downside_std if downside_std > 0 else 0
    
    # Best/Worst
    best_day = returns.max()
    worst_day = returns.min()
    
    # Win rate
    positive_days = (returns > 0).sum()
    total_days = len(returns)
    win_rate = positive_days / total_days if total_days > 0 else 0
    
    return {
        'Total Return': f"{total_return*100:.2f}%",
        'CAGR': f"{cagr*100:.2f}%",
        'Volatility': f"{volatility*100:.2f}%",
        'Max Drawdown': f"{max_drawdown*100:.2f}%",
        'Sharpe Ratio': f"{sharpe:.2f}",
        'Sortino Ratio': f"{sortino:.2f}",
        'Best Day': f"{best_day*100:.2f}%",
        'Worst Day': f"{worst_day*100:.2f}%",
        'Win Rate': f"{win_rate*100:.1f}%",
    }


def run_backtest(
    preset_name,
    custom_tickers,
    benchmark,
    start_date,
    initial_investment,
    contribution_amount,
    contribution_freq,
    rebalance_freq,
    progress=gr.Progress()
):
    """Main backtest function"""
    
    progress(0, desc="Starting backtest...")
    
    # Get tickers
    if preset_name == "Custom":
        tickers = [t.strip().upper() for t in custom_tickers.split(',') if t.strip()]
    else:
        tickers = PRESETS.get(preset_name, [])
    
    if not tickers:
        return None, None, None, "❌ Error: No tickers specified", None
    
    # Parse dates
    start = parse_date(start_date)
    end = datetime.now().strftime('%Y-%m-%d')
    
    progress(0.1, desc="Downloading market data...")
    
    # Download data
    all_tickers = list(set(tickers + [benchmark]))
    
    # Use hybrid data loading (local first, then yfinance)
    close = get_price_data(all_tickers, start, end)
    
    if close.empty:
        return None, None, None, f"❌ Error: No data returned. Tried: {all_tickers}", None
    
    # Check we have the tickers we need
    missing = [t for t in tickers if t not in close.columns]
    if missing:
        return None, None, None, f"❌ Error: Missing data for: {missing}", None
    
    progress(0.3, desc="Running simulation...")
    
    trading_dates = close.index
    num_stocks = len(tickers)
    
    # Determine contribution frequency
    if contribution_freq == "Weekly":
        contrib_dates = [d for d in trading_dates if d.weekday() == 2]  # Wednesdays
    elif contribution_freq == "Monthly":
        contrib_dates = close.resample('M').last().index
    elif contribution_freq == "Quarterly":
        contrib_dates = close.resample('Q').last().index
    else:  # None
        contrib_dates = [trading_dates[0]] if len(trading_dates) > 0 else []
    
    # Determine rebalance dates
    if rebalance_freq == "Monthly":
        rebal_dates = close.resample('M').last().index
    elif rebalance_freq == "Quarterly":
        rebal_dates = close.resample('Q').last().index
    elif rebalance_freq == "Annually":
        rebal_dates = close.resample('Y').last().index
    else:  # None
        rebal_dates = []
    
    # Initialize portfolio
    shares = {t: 0.0 for t in tickers}
    cost_basis = 0.0
    history = []
    
    # Benchmark tracking
    bench_shares = 0.0
    bench_history = []
    
    # Initial investment
    if initial_investment > 0 and len(trading_dates) > 0:
        first_date = trading_dates[0]
        per_stock = initial_investment / num_stocks
        for t in tickers:
            if t in close.columns and not pd.isna(close.loc[first_date, t]):
                shares[t] += per_stock / close.loc[first_date, t]
        cost_basis += initial_investment
        
        if benchmark in close.columns:
            bench_shares += initial_investment / close.loc[first_date, benchmark]
    
    progress(0.5, desc="Processing contributions and rebalancing...")
    
    # Run simulation
    for i, date in enumerate(trading_dates):
        # Contributions
        if date in contrib_dates and contribution_amount > 0:
            per_stock = contribution_amount / num_stocks
            for t in tickers:
                if t in close.columns and not pd.isna(close.loc[date, t]):
                    shares[t] += per_stock / close.loc[date, t]
            cost_basis += contribution_amount
            
            if benchmark in close.columns and not pd.isna(close.loc[date, benchmark]):
                bench_shares += contribution_amount / close.loc[date, benchmark]
        
        # Rebalancing
        if date in rebal_dates and rebalance_freq != "None":
            total_val = sum(shares[t] * close.loc[date, t] for t in tickers 
                          if t in close.columns and not pd.isna(close.loc[date, t]))
            if total_val > 0:
                target = total_val / num_stocks
                for t in tickers:
                    if t in close.columns and not pd.isna(close.loc[date, t]):
                        shares[t] = target / close.loc[date, t]
        
        # Calculate values
        port_val = sum(shares[t] * close.loc[date, t] for t in tickers 
                      if t in close.columns and not pd.isna(close.loc[date, t]))
        bench_val = bench_shares * close.loc[date, benchmark] if benchmark in close.columns else 0
        
        history.append({'date': date, 'value': port_val, 'cost_basis': cost_basis})
        bench_history.append({'date': date, 'value': bench_val})
    
    progress(0.7, desc="Calculating metrics...")
    
    port_df = pd.DataFrame(history).set_index('date')
    bench_df = pd.DataFrame(bench_history).set_index('date')
    
    # Calculate returns
    port_returns = port_df['value'].pct_change().dropna()
    bench_returns = bench_df['value'].pct_change().dropna()
    
    # Calculate metrics
    port_metrics = calculate_metrics(port_returns)
    bench_metrics = calculate_metrics(bench_returns)
    
    # Final values
    final_port = port_df['value'].iloc[-1]
    final_bench = bench_df['value'].iloc[-1]
    total_invested = port_df['cost_basis'].iloc[-1]
    
    progress(0.85, desc="Creating visualizations...")
    
    # ===== CREATE CHARTS =====
    
    # Chart 1: Portfolio Value Over Time
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=port_df.index, y=port_df['value'], name='Portfolio', 
                              line=dict(color='#00D4AA', width=2)))
    fig1.add_trace(go.Scatter(x=bench_df.index, y=bench_df['value'], name=benchmark,
                              line=dict(color='#FF6B6B', width=2, dash='dash')))
    fig1.add_trace(go.Scatter(x=port_df.index, y=port_df['cost_basis'], name='Cost Basis',
                              line=dict(color='#4A90D9', width=1, dash='dot')))
    fig1.update_layout(
        title='Portfolio Value Over Time',
        xaxis_title='Date',
        yaxis_title='Value ($)',
        template='plotly_dark',
        hovermode='x unified',
        yaxis_tickformat='$,.0f'
    )
    
    # Chart 2: Drawdown
    port_cummax = port_df['value'].cummax()
    port_dd = (port_df['value'] - port_cummax) / port_cummax * 100
    bench_cummax = bench_df['value'].cummax()
    bench_dd = (bench_df['value'] - bench_cummax) / bench_cummax * 100
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=port_df.index, y=port_dd, fill='tozeroy', name='Portfolio',
                              line=dict(color='#00D4AA')))
    fig2.add_trace(go.Scatter(x=bench_df.index, y=bench_dd, name=benchmark,
                              line=dict(color='#FF6B6B', dash='dash')))
    fig2.update_layout(
        title='Drawdown Analysis',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        template='plotly_dark',
        hovermode='x unified'
    )
    
    # Chart 3: Holdings breakdown (current weights)
    current_values = {}
    last_date = trading_dates[-1]
    for t in tickers:
        if t in close.columns and not pd.isna(close.loc[last_date, t]):
            current_values[t] = shares[t] * close.loc[last_date, t]
    
    fig3 = go.Figure(data=[go.Pie(
        labels=list(current_values.keys()),
        values=list(current_values.values()),
        hole=0.4,
        textinfo='label+percent',
        hovertemplate='%{label}: $%{value:,.2f}<extra></extra>'
    )])
    fig3.update_layout(
        title='Current Holdings Breakdown',
        template='plotly_dark'
    )
    
    # Chart 4: Monthly returns heatmap
    port_df['returns'] = port_df['value'].pct_change()
    monthly = port_df['returns'].resample('M').apply(lambda x: (1+x).prod()-1) * 100
    
    # Create year-month matrix
    monthly_df = monthly.to_frame('return')
    monthly_df['year'] = monthly_df.index.year
    monthly_df['month'] = monthly_df.index.month
    pivot = monthly_df.pivot(index='year', columns='month', values='return')
    
    fig4 = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        y=pivot.index,
        colorscale='RdYlGn',
        zmid=0,
        text=np.round(pivot.values, 1),
        texttemplate='%{text:.1f}%',
        textfont={"size": 10},
        hovertemplate='%{y} %{x}: %{z:.2f}%<extra></extra>'
    ))
    fig4.update_layout(
        title='Monthly Returns Heatmap (%)',
        template='plotly_dark'
    )
    
    progress(0.95, desc="Generating report...")
    
    # Build results summary
    port_return = (final_port - total_invested) / total_invested * 100
    bench_return = (final_bench - total_invested) / total_invested * 100
    alpha = port_return - bench_return
    
    summary = f"""
## 📊 Backtest Results

**Period:** {start} to {end}  
**Strategy:** {preset_name if preset_name != "Custom" else "Custom Portfolio"}  
**Tickers:** {', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''} ({len(tickers)} total)  
**Benchmark:** {benchmark}

---

### 💰 Performance Summary

| Metric | Portfolio | {benchmark} |
|--------|-----------|-------------|
| **Total Invested** | ${total_invested:,.2f} | ${total_invested:,.2f} |
| **Final Value** | ${final_port:,.2f} | ${final_bench:,.2f} |
| **Total Return** | {port_return:+.2f}% | {bench_return:+.2f}% |
| **CAGR** | {port_metrics.get('CAGR', 'N/A')} | {bench_metrics.get('CAGR', 'N/A')} |
| **Max Drawdown** | {port_metrics.get('Max Drawdown', 'N/A')} | {bench_metrics.get('Max Drawdown', 'N/A')} |
| **Sharpe Ratio** | {port_metrics.get('Sharpe Ratio', 'N/A')} | {bench_metrics.get('Sharpe Ratio', 'N/A')} |
| **Sortino Ratio** | {port_metrics.get('Sortino Ratio', 'N/A')} | {bench_metrics.get('Sortino Ratio', 'N/A')} |
| **Volatility** | {port_metrics.get('Volatility', 'N/A')} | {bench_metrics.get('Volatility', 'N/A')} |
| **Win Rate** | {port_metrics.get('Win Rate', 'N/A')} | {bench_metrics.get('Win Rate', 'N/A')} |

---

### {'🏆' if alpha > 0 else '📉'} Alpha vs {benchmark}: **{alpha:+.2f}%**

"""
    
    # Holdings table
    holdings_data = []
    for t in tickers:
        if t in close.columns:
            start_price = close[t].dropna().iloc[0]
            end_price = close[t].dropna().iloc[-1]
            price_chg = (end_price / start_price - 1) * 100
            value = shares[t] * end_price
            weight = value / final_port * 100 if final_port > 0 else 0
            holdings_data.append({
                'Ticker': t,
                'Shares': f"{shares[t]:.4f}",
                'Price': f"${end_price:.2f}",
                'Value': f"${value:,.2f}",
                'Weight': f"{weight:.2f}%",
                'Return': f"{price_chg:+.2f}%"
            })
    
    holdings_df = pd.DataFrame(holdings_data)
    
    progress(1.0, desc="Done!")
    
    return fig1, fig2, fig3, summary, holdings_df


def update_tickers(preset_name):
    """Update ticker textbox when preset changes"""
    if preset_name == "Custom":
        return gr.update(value="", interactive=True, visible=True)
    else:
        tickers = PRESETS.get(preset_name, [])
        return gr.update(value=", ".join(tickers), interactive=False, visible=True)


# ===== BUILD GRADIO INTERFACE =====

with gr.Blocks(
    title="Horizon Portfolio Backtester",
    theme=gr.themes.Soft(
        primary_hue="emerald",
        secondary_hue="slate",
    ),
    css="""
    .gradio-container { max-width: 1400px !important; }
    .metric-box { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                  border-radius: 10px; padding: 20px; margin: 10px 0; }
    """
) as demo:
    
    gr.Markdown("""
    # 🚀 Horizon Portfolio Backtester
    
    A comprehensive portfolio backtesting tool inspired by Portfolio Visualizer.  
    Analyze historical performance, risk metrics, and compare against benchmarks.
    
    ---
    """)
    
    with gr.Row():
        # Left column - Inputs
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Portfolio Configuration")
            
            preset_dropdown = gr.Dropdown(
                choices=list(PRESETS.keys()),
                value="Horizon Growth",
                label="Portfolio Preset",
                info="Select a preset or choose Custom"
            )
            
            custom_tickers = gr.Textbox(
                label="Tickers (comma-separated)",
                value=", ".join(PRESETS["Horizon Growth"]),
                placeholder="AAPL, MSFT, GOOGL...",
                interactive=False,
                lines=3
            )
            
            benchmark = gr.Dropdown(
                choices=BENCHMARKS,
                value="QQQ",
                label="Benchmark",
                info="Compare against any ticker in the dataset"
            )
            
            gr.Markdown("### 📅 Time Period")
            
            start_date = gr.Textbox(
                label="Start Date",
                value="2024-01-01",
                placeholder="YYYY-MM-DD or 1y, 3m, 5y",
                info="Use '3m', '1y', '5y' for relative dates"
            )
            
            gr.Markdown("### 💵 Investment Strategy")
            
            initial_investment = gr.Number(
                label="Initial Investment ($)",
                value=10000,
                minimum=0
            )
            
            contribution_amount = gr.Number(
                label="Periodic Contribution ($)",
                value=1000,
                minimum=0
            )
            
            contribution_freq = gr.Dropdown(
                choices=["None", "Weekly", "Monthly", "Quarterly"],
                value="Weekly",
                label="Contribution Frequency"
            )
            
            rebalance_freq = gr.Dropdown(
                choices=["None", "Monthly", "Quarterly", "Annually"],
                value="None",
                label="Rebalancing Frequency"
            )
            
            run_btn = gr.Button("🚀 Run Backtest", variant="primary", size="lg")
        
        # Right column - Results
        with gr.Column(scale=2):
            gr.Markdown("### 📈 Results")
            
            summary_output = gr.Markdown()
            
            with gr.Tabs():
                with gr.TabItem("📊 Performance"):
                    chart1 = gr.Plot(label="Portfolio Value")
                
                with gr.TabItem("📉 Drawdown"):
                    chart2 = gr.Plot(label="Drawdown Analysis")
                
                with gr.TabItem("🥧 Holdings"):
                    chart3 = gr.Plot(label="Current Allocation")
                
                with gr.TabItem("📋 Holdings Table"):
                    holdings_table = gr.Dataframe(
                        headers=['Ticker', 'Shares', 'Price', 'Value', 'Weight', 'Return'],
                        label="Individual Holdings"
                    )
    
    # Event handlers
    preset_dropdown.change(
        fn=update_tickers,
        inputs=[preset_dropdown],
        outputs=[custom_tickers]
    )
    
    run_btn.click(
        fn=run_backtest,
        inputs=[
            preset_dropdown,
            custom_tickers,
            benchmark,
            start_date,
            initial_investment,
            contribution_amount,
            contribution_freq,
            rebalance_freq
        ],
        outputs=[chart1, chart2, chart3, summary_output, holdings_table]
    )
    
    gr.Markdown("""
    ---
    
    ### 📖 Quick Start Guide
    
    1. **Select a preset** or enter custom tickers
    2. **Choose a benchmark** (QQQ, SPY, etc.)
    3. **Set your time period** (e.g., "2024-01-01" or "5y" for 5 years)
    4. **Configure contributions** (initial + periodic)
    5. **Set rebalancing** (None, Monthly, Quarterly, Annually)
    6. **Click Run Backtest!**
    
    ---
    
    *Built with ❤️ using Gradio | Data from Yahoo Finance*
    """)
    
    # ===== API TAB =====
    gr.Markdown("""
    ---
    
    ## 🔌 API Access
    
    This Space exposes API endpoints for programmatic access. Use these to integrate backtesting into your workflows.
    """)
    
    with gr.Tabs():
        with gr.TabItem("🤖 Natural Language (Claude)"):
            gr.Markdown("""
            **Ask Claude to run a backtest in plain English!**
            
            Examples:
            - "Backtest the Magnificent 7 for the last 2 years with $500 weekly contributions"
            - "Compare AAPL, MSFT, GOOGL against SPY starting from 2023"
            - "Run Horizon Growth from January 2024 with monthly rebalancing"
            """)
            
            nl_input = gr.Textbox(
                label="Your backtest request",
                placeholder="e.g., Backtest FAANG stocks for 5 years with $1000 monthly DCA",
                lines=3
            )
            nl_button = gr.Button("🚀 Run with Claude", variant="primary")
            nl_output = gr.Code(label="Results (JSON)", language="json")
            
            nl_button.click(
                fn=natural_language_backtest,
                inputs=[nl_input],
                outputs=[nl_output],
                api_name="natural_language_backtest"
            )
        
        with gr.TabItem("⚡ Direct API"):
            gr.Markdown("""
            **Call the API directly with parameters.**
            
            Endpoint: `/api/backtest`
            
            ```python
            import requests
            
            response = requests.post(
                "https://YOUR-SPACE.hf.space/api/backtest",
                json={
                    "preset": "Horizon Growth",
                    "benchmark": "QQQ",
                    "start_date": "2024-01-01",
                    "initial_investment": 10000,
                    "contribution_amount": 1000,
                    "contribution_freq": "Weekly",
                    "rebalance_freq": "None"
                }
            )
            print(response.json())
            ```
            """)
            
            with gr.Row():
                with gr.Column():
                    api_preset = gr.Dropdown(
                        choices=list(PRESETS.keys()),
                        value="Horizon Growth",
                        label="Preset"
                    )
                    api_tickers = gr.Textbox(
                        label="Custom Tickers (if preset=Custom)",
                        placeholder="AAPL, MSFT, GOOGL"
                    )
                    api_benchmark = gr.Dropdown(
                        choices=BENCHMARKS,
                        value="QQQ",
                        label="Benchmark",
                        info="Any ticker in the dataset"
                    )
                    api_start = gr.Textbox(
                        label="Start Date",
                        value="2024-01-01"
                    )
                
                with gr.Column():
                    api_initial = gr.Number(
                        label="Initial Investment",
                        value=10000
                    )
                    api_contrib = gr.Number(
                        label="Contribution Amount",
                        value=1000
                    )
                    api_contrib_freq = gr.Dropdown(
                        choices=["None", "Weekly", "Monthly", "Quarterly"],
                        value="Weekly",
                        label="Contribution Frequency"
                    )
                    api_rebal = gr.Dropdown(
                        choices=["None", "Monthly", "Quarterly", "Annually"],
                        value="None",
                        label="Rebalancing"
                    )
            
            api_button = gr.Button("📊 Run API Backtest", variant="secondary")
            api_output = gr.Code(label="API Response (JSON)", language="json")
            
            api_button.click(
                fn=api_backtest,
                inputs=[api_preset, api_tickers, api_benchmark, api_start, 
                       api_initial, api_contrib, api_contrib_freq, api_rebal],
                outputs=[api_output],
                api_name="backtest"
            )
        
        with gr.TabItem("📚 API Documentation"):
            gr.Markdown("""
            ## API Endpoints
            
            ### 1. Natural Language Backtest
            
            **Endpoint:** `POST /api/natural_language_backtest`
            
            Uses Claude AI to parse your request and run a backtest.
            
            **Request:**
            ```json
            {
                "user_request": "Backtest the Magnificent 7 for 2 years with $500 weekly DCA"
            }
            ```
            
            **Response:**
            ```json
            {
                "success": true,
                "interpretation": "Running Magnificent 7 preset from 2023-01-01...",
                "results": {
                    "total_invested": 62000,
                    "portfolio_value": 85432.10,
                    "portfolio_return": 37.79,
                    "alpha": 12.45
                },
                "holdings": {...}
            }
            ```
            
            ---
            
            ### 2. Direct Backtest API
            
            **Endpoint:** `POST /api/backtest`
            
            **Parameters:**
            | Parameter | Type | Default | Description |
            |-----------|------|---------|-------------|
            | preset | string | "Horizon Growth" | Preset name or "Custom" |
            | tickers | string | "" | Comma-separated tickers (for Custom) |
            | benchmark | string | "QQQ" | Any ticker in the dataset |
            | start_date | string | "2024-01-01" | Start date or relative (1y, 5y) |
            | initial_investment | number | 10000 | Initial investment |
            | contribution_amount | number | 1000 | Periodic contribution |
            | contribution_freq | string | "Weekly" | None/Weekly/Monthly/Quarterly |
            | rebalance_freq | string | "None" | None/Monthly/Quarterly/Annually |
            
            ---
            
            ### Python Client Example
            
            ```python
            from gradio_client import Client
            
            client = Client("YOUR-USERNAME/horizon-backtester")
            
            # Natural language
            result = client.predict(
                user_request="Backtest FAANG for 3 years",
                api_name="/natural_language_backtest"
            )
            
            # Direct API
            result = client.predict(
                preset="Magnificent 7",
                tickers="",
                benchmark="SPY",
                start_date="2022-01-01",
                initial_investment=10000,
                contribution_amount=500,
                contribution_freq="Weekly",
                rebalance_freq="Quarterly",
                api_name="/backtest"
            )
            ```
            
            ---
            
            ### cURL Example
            
            ```bash
            curl -X POST "https://YOUR-SPACE.hf.space/api/backtest" \\
                -H "Content-Type: application/json" \\
                -d '{
                    "preset": "FAANG",
                    "benchmark": "QQQ",
                    "start_date": "2023-01-01",
                    "initial_investment": 10000,
                    "contribution_amount": 1000,
                    "contribution_freq": "Weekly"
                }'
            ```
            """)
        
        with gr.TabItem("📥 Request Ticker"):
            gr.Markdown("""
            ## Request a New Ticker
            
            Can't find a ticker in our dataset? Request it here!  
            Requested tickers are added during the next daily data update (~9:30 PM UTC).
            
            **Current dataset:** ~613 tickers including S&P 500, Nasdaq 100, major ETFs, and recent IPOs.
            """)
            
            with gr.Row():
                with gr.Column():
                    request_input = gr.Textbox(
                        label="Ticker Symbol",
                        placeholder="e.g., PLTR, SOFI, RKLB",
                        info="Enter a valid stock ticker (1-5 letters)"
                    )
                    request_btn = gr.Button("📥 Request Ticker", variant="primary")
                    request_output = gr.Markdown()
                
                with gr.Column():
                    gr.Markdown("### 📋 Pending Requests")
                    pending_display = gr.Markdown()
                    refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
            
            def handle_request(ticker):
                result = request_ticker(ticker)
                pending = get_pending_requests()
                pending_text = ", ".join(pending) if pending else "None"
                return result["message"], f"**Queue:** {pending_text}"
            
            def refresh_pending():
                pending = get_pending_requests()
                pending_text = ", ".join(pending) if pending else "None"
                return f"**Queue:** {pending_text}"
            
            request_btn.click(
                fn=handle_request,
                inputs=[request_input],
                outputs=[request_output, pending_display]
            )
            
            refresh_btn.click(
                fn=refresh_pending,
                inputs=[],
                outputs=[pending_display]
            )


if __name__ == "__main__":
    demo.launch(show_error=True)
