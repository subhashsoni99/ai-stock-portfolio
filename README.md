# Stock Predictor - Subhash Chand Soni
## Hybrid LSTM + Technical Trading Strategy (TCS)

This project demonstrates a practical, reproducible pipeline for building an AI-driven trading strategy using historical equity data (TCS). It includes data ingestion, feature engineering (SMA, EMA, RSI, MACD, ATR, Bollinger Bands, OBV, momentum, volatility), LSTM model training with class-weighting, and a hybrid trading rule that combines LSTM probability signals with trend filters (EMA crossover) and RSI safety checks. The strategy is backtested with transaction costs and produces realistic performance metrics and visualizations. Results and code are organized for easy reuse and deployment: notebooks, model, scaler, and backtest outputs are saved in the `models/` and `results/` folders.  

**Key outputs:** processed dataset, trained LSTM model (`models/lstm_final.h5`), test results (`results/df_test_hybrid_final.csv`), cumulative return plots, and the diagnostics dictionary summarizing returns, trade counts, and model accuracy.  


**Short:** Predict next-day stock direction using classical ML & LSTM; deployed with Streamlit.

## About
Subhash Chand Soni — Technical Lead | Full-Stack Developer (Node.js, Python, Cloud) | 10+ years experience.
Focus: Python, AI/ML, Time-Series & Stock Market Analytics.

## Project Overview
- Data: Yahoo Finance (yfinance)
- Models: Logistic Regression, RandomForest, LSTM (TensorFlow)
- Backtesting: MA-strategy and model-driven signals

## How to run (local)
1. Create venv and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
Contact

Phone: +91-9179557071
Location: Bilaspur, Chhattisgarh
>EOF
