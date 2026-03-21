"""
BTCUSD RSI Trading Algorithm
=============================
This script fetches historical BTC-USD price data using yfinance,
calculates the Relative Strength Index (RSI) indicator, and generates
basic buy/sell signals based on standard RSI thresholds:
  - Buy signal  : RSI < 30  (oversold territory)
  - Sell signal : RSI > 70  (overbought territory)

Dependencies: yfinance, pandas, numpy
Install via:  pip install yfinance pandas numpy
"""

import sys

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    sys.exit(
        "Error: 'yfinance' is not installed. Run: pip install yfinance"
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TICKER = "BTC-USD"          # Yahoo Finance ticker for Bitcoin / US Dollar
PERIOD = "6mo"              # Data period: 6 months of daily bars
INTERVAL = "1d"             # Bar interval: daily
RSI_PERIOD = 14             # Standard RSI look-back period
RSI_OVERSOLD = 30           # RSI below this value → Buy signal
RSI_OVERBOUGHT = 70         # RSI above this value → Sell signal


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_price_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance for the given ticker.

    Args:
        ticker:   Yahoo Finance ticker symbol (e.g. 'BTC-USD').
        period:   Lookback period string accepted by yfinance (e.g. '6mo').
        interval: Bar interval string accepted by yfinance (e.g. '1d').

    Returns:
        DataFrame with at least a 'Close' column indexed by date.

    Raises:
        ValueError: If the download returns an empty DataFrame.
    """
    print(f"Fetching {interval} price data for {ticker} over the last {period}...")
    df = yf.download(ticker, period=period, interval=interval, progress=False)

    if df.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}'. "
            "Check your internet connection or the ticker symbol."
        )

    # Flatten multi-level columns that yfinance sometimes returns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print(f"  Retrieved {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}.")
    return df


# ---------------------------------------------------------------------------
# RSI calculation
# ---------------------------------------------------------------------------

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate the Relative Strength Index (RSI) for a price series.

    Uses the Wilder (exponential moving average) smoothing method, which is
    the original definition used by J. Welles Wilder Jr.

    Args:
        prices: A pandas Series of closing prices.
        period: Look-back window; defaults to 14.

    Returns:
        A pandas Series of RSI values (0–100), aligned with *prices*.
        The first *period* values will be NaN due to the warm-up requirement.
    """
    if len(prices) < period + 1:
        raise ValueError(
            f"Need at least {period + 1} price bars to compute RSI({period}), "
            f"but only {len(prices)} bars were provided."
        )

    delta = prices.diff()

    # Separate gains (positive changes) and losses (absolute negative changes)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Initial averages using simple mean for the seed value
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    # When avg_loss == 0 the asset only moved up → RSI = 100.
    # Using np.where avoids a division-by-zero warning while preserving NaN
    # values produced during the EWM warm-up period.
    rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
    nan_mask = np.logical_or(np.isnan(avg_gain), np.isnan(avg_loss))
    rsi = pd.Series(
        np.where(nan_mask, np.nan, 100 - (100 / (1 + rs))),
        index=prices.index,
    )

    return rsi


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_signals(df: pd.DataFrame, rsi_col: str = "RSI") -> pd.DataFrame:
    """Add a 'Signal' column to *df* based on RSI thresholds.

    Signal values:
        'BUY'  – RSI crossed below RSI_OVERSOLD (30)
        'SELL' – RSI crossed above RSI_OVERBOUGHT (70)
        ''     – No actionable signal

    Args:
        df:      DataFrame that contains an RSI column.
        rsi_col: Name of the RSI column in *df*.

    Returns:
        The same DataFrame with a new 'Signal' column appended.
    """
    conditions = [
        df[rsi_col] < RSI_OVERSOLD,
        df[rsi_col] > RSI_OVERBOUGHT,
    ]
    choices = ["BUY", "SELL"]
    df["Signal"] = np.select(conditions, choices, default="")
    return df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    """Print a concise summary of the latest RSI value and recent signals."""
    latest = df.dropna(subset=["RSI"]).iloc[-1]
    print(f"\n--- Latest bar: {latest.name.date()} ---")
    print(f"  Close price : ${latest['Close']:,.2f}")
    print(f"  RSI({RSI_PERIOD})      : {latest['RSI']:.2f}")
    print(f"  Signal      : {latest['Signal'] or 'HOLD'}")

    # Show most recent buy/sell signals (up to 5)
    signals = df[df["Signal"].isin(["BUY", "SELL"])].tail(5)
    if signals.empty:
        print("\nNo BUY/SELL signals generated in the selected period.")
    else:
        print(f"\nMost recent signals (last {len(signals)}):")
        for ts, row in signals.iterrows():
            print(
                f"  {ts.date()}  {row['Signal']:4s}  "
                f"Close=${row['Close']:,.2f}  RSI={row['RSI']:.2f}"
            )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate data fetching, RSI calculation, signal generation, and output."""
    try:
        # 1. Fetch historical price data
        df = fetch_price_data(TICKER, PERIOD, INTERVAL)

        # 2. Calculate RSI on the closing price
        df["RSI"] = calculate_rsi(df["Close"], period=RSI_PERIOD)

        # 3. Generate buy/sell signals based on RSI thresholds
        df = generate_signals(df)

        # 4. Display results
        print_summary(df)

        # 5. Optionally show the last 10 rows for a quick sanity check
        print("\nLast 10 bars:")
        display_cols = ["Close", "RSI", "Signal"]
        print(df[display_cols].tail(10).to_string())

    except ValueError as exc:
        sys.exit(f"Data error: {exc}")
    except (ValueError, RuntimeError, OSError) as exc:
        sys.exit(f"Unexpected error: {exc}")


if __name__ == "__main__":
    main()
