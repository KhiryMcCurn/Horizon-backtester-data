---
title: Horizon Portfolio Backtester
emoji: 📈
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# Horizon Portfolio Backtester

A Portfolio Visualizer clone built with Gradio.

## Features
- Interactive UI for portfolio backtesting
- 600+ tickers (S&P 500, Nasdaq 100, ETFs, recent IPOs)
- Daily automated data updates via GitHub Actions
- REST API for programmatic access

## How it works
1. GitHub Actions downloads market data from yfinance daily
2. Data is committed to GitHub
3. Changes are automatically pushed to HuggingFace Spaces
