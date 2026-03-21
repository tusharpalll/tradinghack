"""
Trading Algorithm: RSI-based signals for BTCUSD, Nifty 50, and Sensex
=======================================================================
Assets:
  - BTCUSD   : Bitcoin / US Dollar (Crypto)
  - ^NSEI    : Nifty 50 (India, NSE)
  - ^BSESN   : Sensex   (India, BSE)

Strategy:
  RSI < 30  --> Oversold  --> BUY  signal
  RSI > 70  --> Overbought --> SELL signal
  Otherwise --> HOLD signal
"""

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ASSETS = {
    "BTCUSD": "BTC-USD",
    "Nifty 50": "^NSEI",
    "Sensex": "^BSESN",
}

RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
DATA_PERIOD = "6mo"   # last 6 months of daily data
DATA_INTERVAL = "1d"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def calculate_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Compute Wilder's RSI for a price series."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    # Avoid division by zero: when there are no losses RSI is 100
    rsi = pd.Series(np.where(avg_loss == 0, 100.0, 100 - (100 / (1 + avg_gain / avg_loss))),
                    index=series.index)
    rsi[avg_gain.isna()] = np.nan
    return rsi


def generate_signal(rsi_value: float) -> str:
    """Return BUY, SELL, or HOLD based on RSI value."""
    if pd.isna(rsi_value):
        return "N/A"
    if rsi_value < RSI_OVERSOLD:
        return "BUY"
    if rsi_value > RSI_OVERBOUGHT:
        return "SELL"
    return "HOLD"


def fetch_and_analyze(name: str, ticker: str) -> dict:
    """
    Download data for *ticker*, compute RSI, and return a result dictionary
    with the last few signal rows and a summary.
    """
    print(f"  Fetching data for {name} ({ticker}) ...")
    raw = yf.download(ticker, period=DATA_PERIOD, interval=DATA_INTERVAL,
                      auto_adjust=True, progress=False)

    if raw.empty:
        print(f"  WARNING: No data returned for {name} ({ticker}).")
        return {"name": name, "ticker": ticker, "error": "No data available"}

    # Flatten MultiIndex columns that yfinance sometimes returns
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Close"]].copy()
    df.columns = ["Close"]
    df["RSI"] = calculate_rsi(df["Close"])
    df["Signal"] = df["RSI"].apply(generate_signal)

    latest = df.dropna(subset=["RSI"]).iloc[-1]

    return {
        "name": name,
        "ticker": ticker,
        "latest_close": round(float(latest["Close"]), 4),
        "latest_rsi": round(float(latest["RSI"]), 2),
        "latest_signal": latest["Signal"],
        "data": df,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  RSI Trading Algorithm")
    print(f"  Assets : {', '.join(ASSETS.keys())}")
    print(f"  Period : {DATA_PERIOD}  |  RSI window: {RSI_PERIOD}")
    print("=" * 60)

    results = {}
    for name, ticker in ASSETS.items():
        results[name] = fetch_and_analyze(name, ticker)

    print()
    print("=" * 60)
    print("  SUMMARY — Latest RSI Signals")
    print("=" * 60)
    print(f"{'Asset':<12} {'Ticker':<10} {'Close':>14} {'RSI':>8} {'Signal':>8}")
    print("-" * 60)

    for name, res in results.items():
        if "error" in res:
            print(f"{name:<12} {res['ticker']:<10}  {'ERROR':>14}  {'--':>8}  {res['error']:>8}")
        else:
            print(
                f"{res['name']:<12} {res['ticker']:<10} "
                f"{res['latest_close']:>14,.4f} "
                f"{res['latest_rsi']:>8.2f} "
                f"{res['latest_signal']:>8}"
            )

    print()
    print("=" * 60)
    print("  RECENT SIGNALS (last 5 trading days per asset)")
    print("=" * 60)

    for name, res in results.items():
        if "error" in res:
            continue
        df = res["data"].dropna(subset=["RSI"]).tail(5)[["Close", "RSI", "Signal"]]
        print(f"\n--- {res['name']} ({res['ticker']}) ---")
        print(df.to_string())

    print()
    print("Legend:  RSI < 30 → BUY  |  RSI > 70 → SELL  |  Otherwise → HOLD")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
