"""
News sentiment trading signals.

Scores recent headlines for a list of US stocks with VADER (plus a small
financial phrase lexicon) and combines the result with RSI and an ATR
volatility filter to flag BUY / SELL / HOLD signals.

Usage:
    export NEWS_API_KEY=your_key
    python news_sentiment_signals.py                # default watchlist
    python news_sentiment_signals.py AAPL MSFT NVDA # your own tickers
"""

import os
import re
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LARGE_CAPS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B",
    "AVGO", "LLY", "JPM", "UNH", "V", "XOM", "MA", "HD", "PG", "JNJ",
    "WMT", "KO", "PEP", "ADBE", "CRM", "COST", "TMO",
]
WATCHLIST = ["ORCL", "AMD", "QCOM", "NFLX", "ENPH"]
DEFAULT_TICKERS = LARGE_CAPS + WATCHLIST

NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")
NEWS_URL = "https://newsapi.org/v2/everything"

NEWS_DAYS_BACK = 2
RECENT_ARTICLES = 8         # articles used for the average score
NEG_THRESHOLD = -0.3        # compound score counted as negative
NEG_STREAK = 2              # most recent N articles must all be negative
BUY_SENTIMENT = 0.1
RSI_PERIOD = 14
ATR_PERIOD = 14
RSI_OVERBOUGHT = 65
RSI_OVERSOLD = 35
MIN_ATR_PCT = 0.005         # skip names whose ATR is below 0.5% of price

# VADER scores single tokens, so multi-word phrases are joined into one
# token before scoring and the joined token is added to the lexicon.
FINANCIAL_PHRASES = {
    "missed guidance": -1.5,
    "miss guidance": -1.5,
    "weak outlook": -1.2,
    "cut guidance": -1.5,
    "beat estimates": 1.3,
    "raised guidance": 1.5,
    "strong outlook": 1.2,
}
FINANCIAL_WORDS = {"downgrade": -1.8, "downgraded": -1.8,
                   "upgrade": 1.8, "upgraded": 1.8}

analyzer = SentimentIntensityAnalyzer()
analyzer.lexicon.update(FINANCIAL_WORDS)
analyzer.lexicon.update({p.replace(" ", "_"): s for p, s in FINANCIAL_PHRASES.items()})


def join_phrases(text):
    """Replace known financial phrases with single underscore-joined tokens."""
    for phrase in FINANCIAL_PHRASES:
        text = re.sub(rf"\b{phrase}\b", phrase.replace(" ", "_"), text, flags=re.IGNORECASE)
    return text


def score_text(text):
    return analyzer.polarity_scores(join_phrases(text))["compound"]


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def rsi(close, period=RSI_PERIOD):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)


def atr(df, period=ATR_PERIOD):
    prev_close = df["Close"].shift(1)
    true_range = pd.concat(
        [df["High"] - df["Low"],
         (df["High"] - prev_close).abs(),
         (df["Low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period).mean()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def get_prices(ticker, days=60):
    """Daily OHLC with RSI and ATR columns, or None if no data."""
    try:
        df = yf.Ticker(ticker).history(period=f"{days}d", interval="1d")
    except Exception as exc:
        print(f"  price error for {ticker}: {exc}")
        return None
    if df.empty:
        return None
    df = df.reset_index()
    df["RSI"] = rsi(df["Close"])
    df["ATR"] = atr(df)
    return df


def get_news(ticker, days_back=NEWS_DAYS_BACK):
    """Recent articles for a ticker, newest first, each with a sentiment score."""
    params = {
        "q": f'"{ticker}" OR ${ticker} OR {ticker} stock',
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 20,
        "from": (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d"),
        "apiKey": NEWS_API_KEY,
    }
    try:
        data = requests.get(NEWS_URL, params=params, timeout=10).json()
    except requests.exceptions.RequestException as exc:
        print(f"  news error for {ticker}: {exc}")
        return []

    if data.get("status") != "ok":
        print(f"  NewsAPI: {data.get('message', 'request failed')}")
        return []

    articles = []
    for item in data.get("articles", []):
        title = item.get("title") or ""
        description = item.get("description") or ""
        if len(title) <= 5:
            continue
        articles.append({
            "title": title[:80],
            "sentiment": score_text(f"{title}. {description}"),
        })
    return articles


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------
def make_signal(ticker):
    result = {"ticker": ticker, "signal": "HOLD", "price": np.nan, "rsi": np.nan,
              "atr": np.nan, "sentiment": np.nan, "neg_streak": 0,
              "news_count": 0, "reason": ""}

    news = get_news(ticker)
    result["news_count"] = len(news)
    if not news:
        result["reason"] = "no recent news"
        return result

    recent = [a["sentiment"] for a in news[:RECENT_ARTICLES]]   # newest first
    result["sentiment"] = float(np.mean(recent))
    result["neg_streak"] = sum(s < NEG_THRESHOLD for s in recent[:NEG_STREAK])

    df = get_prices(ticker)
    if df is None or df["ATR"].dropna().empty or df["RSI"].dropna().empty:
        result["reason"] = "not enough price data"
        return result

    last = df.iloc[-1]
    result.update(price=last["Close"], rsi=last["RSI"], atr=last["ATR"])
    volatile = last["ATR"] > last["Close"] * MIN_ATR_PCT

    if not volatile:
        result["reason"] = f"low volatility (ATR {last['ATR']:.2f})"
    elif result["neg_streak"] >= NEG_STREAK and last["RSI"] > RSI_OVERBOUGHT:
        result["signal"] = "SELL"
        result["reason"] = f"negative news streak, RSI {last['RSI']:.0f}"
    elif result["sentiment"] > BUY_SENTIMENT and last["RSI"] < RSI_OVERSOLD:
        result["signal"] = "BUY"
        result["reason"] = f"positive news {result['sentiment']:.2f}, RSI {last['RSI']:.0f}"
    else:
        result["reason"] = f"news {result['sentiment']:.2f}, RSI {last['RSI']:.0f}"
    return result


def run(tickers):
    print(f"Scanning {len(tickers)} tickers  ({datetime.now():%Y-%m-%d %H:%M})\n")
    rows = []
    for ticker in tickers:
        row = make_signal(ticker)
        rows.append(row)
        print(f"{row['ticker']:6} {row['signal']:4}  {row['reason']}")

    df = pd.DataFrame(rows)
    buys = df.loc[df["signal"] == "BUY", "ticker"].tolist()
    sells = df.loc[df["signal"] == "SELL", "ticker"].tolist()
    print("\nBUY: " + (", ".join(buys) or "none"))
    print("SELL: " + (", ".join(sells) or "none"))
    return df


def save(df, folder="output"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"signals_{datetime.now():%Y%m%d_%H%M%S}.csv")
    df.to_csv(path, index=False)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    if not NEWS_API_KEY:
        sys.exit("Set the NEWS_API_KEY environment variable first.")
    tickers = [t.upper() for t in sys.argv[1:]] or DEFAULT_TICKERS
    save(run(tickers))
