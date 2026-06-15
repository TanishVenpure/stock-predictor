import ta
import pandas as pd
import numpy as np
import yfinance as yf
import datetime


def get_nifty_features(start="2018-01-01", end=None):
    if end is None:
        end = datetime.date.today().strftime("%Y-%m-%d")
    nifty = yf.download("^NSEI", start=start, end=end, auto_adjust=True)
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)
    nifty["nifty_return_1d"] = nifty["Close"].pct_change(1)
    nifty["nifty_return_5d"] = nifty["Close"].pct_change(5)
    nifty["nifty_rsi"]       = ta.momentum.RSIIndicator(nifty["Close"], window=14).rsi()
    nifty["nifty_above_ema"] = (
        nifty["Close"] > ta.trend.EMAIndicator(nifty["Close"], window=50).ema_indicator()
    ).astype(int)
    return nifty[["nifty_return_1d", "nifty_return_5d",
                   "nifty_rsi", "nifty_above_ema"]]


def add_candlestick_patterns(df):
    o = df["Open"]
    h = df["High"]
    l = df["Low"]
    c = df["Close"]
    body         = c - o
    body_abs     = body.abs()
    candle_range = h - l
    lower_wick   = df[["Open","Close"]].min(axis=1) - l
    upper_wick   = h - df[["Open","Close"]].max(axis=1)

    df["cdl_doji"]            = ((body_abs <= 0.1 * candle_range) & (candle_range > 0)).astype(int)
    df["cdl_hammer"]          = ((lower_wick >= 2 * body_abs) & (upper_wick <= 0.1 * candle_range) & (candle_range > 0)).astype(int)
    df["cdl_shooting_star"]   = ((upper_wick >= 2 * body_abs) & (lower_wick <= 0.1 * candle_range) & (candle_range > 0)).astype(int)
    df["cdl_engulfing"]       = ((body > 0) & (body.shift(1) < 0) & (o < c.shift(1)) & (c > o.shift(1))).astype(int)
    df["cdl_marubozu"]        = ((body_abs >= 0.9 * candle_range) & (candle_range > 0)).astype(int)
    df["cdl_morning_star"]    = ((body.shift(2) < 0) & (body_abs.shift(1) <= 0.3 * candle_range.shift(1)) & (body > 0) & (c > (o.shift(2) + c.shift(2)) / 2)).astype(int)
    df["cdl_evening_star"]    = ((body.shift(2) > 0) & (body_abs.shift(1) <= 0.3 * candle_range.shift(1)) & (body < 0) & (c < (o.shift(2) + c.shift(2)) / 2)).astype(int)
    df["cdl_3white_soldiers"] = ((body > 0) & (body.shift(1) > 0) & (body.shift(2) > 0) & (o > o.shift(1)) & (c > c.shift(1))).astype(int)

    pattern_cols             = [col for col in df.columns if col.startswith("cdl_")]
    df["pattern_strength"]   = df[pattern_cols].sum(axis=1)
    df["pattern_count"]      = (df[pattern_cols] != 0).sum(axis=1)
    return df


def build_features(df, nifty_features):
    df["rsi"]         = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    macd              = ta.trend.MACD(df["Close"])
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()
    df["ema_20"]      = ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator()
    df["ema_50"]      = ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator()
    df["ema_cross"]   = (df["ema_20"] > df["ema_50"]).astype(int)
    bb                = ta.volatility.BollingerBands(df["Close"])
    df["bb_upper"]    = bb.bollinger_hband()
    df["bb_lower"]    = bb.bollinger_lband()
    df["bb_width"]    = (df["bb_upper"] - df["bb_lower"]) / df["Close"]
    df["atr"]         = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
    df["volume_ma"]    = df["Volume"].rolling(20).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_ma"]
    df["return_1d"]    = df["Close"].pct_change(1)
    df["return_5d"]    = df["Close"].pct_change(5)
    df["return_10d"]   = df["Close"].pct_change(10)
    df["lag_1"]        = df["return_1d"].shift(1)
    df["lag_2"]        = df["return_1d"].shift(2)
    df["lag_3"]        = df["return_1d"].shift(3)
    df = add_candlestick_patterns(df)
    df = df.merge(nifty_features, left_index=True, right_index=True, how="left")
    df["rel_strength_1d"] = df["return_1d"] - df["nifty_return_1d"]
    df["rel_strength_5d"] = df["return_5d"] - df["nifty_return_5d"]
    df["rsi_vs_nifty"]    = df["rsi"] - df["nifty_rsi"]
    df.dropna(inplace=True)
    return df