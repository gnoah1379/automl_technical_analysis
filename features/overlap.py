import pandas as pd
import pandas_ta as ta


def ohlc4_feature_engineering(df):
    ohlc4 = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    df["ohlc4_pct"] = ohlc4.pct_change()
    df["ohlc4_pct_sma"] = ta.sma(df["ohlc4_pct"])
    df["ohlc4_pct_ema"] = ta.ema(df["ohlc4_pct"])


def close_feature_engineering(df):
    df["close_pct"] = df["close"].pct_change()
    df["close_pct_sma"] = ta.sma(df["close_pct"])
    df["close_pct_ema"] = ta.ema(df["close_pct"])


def sma_feature_engineering(df):
    sma10 = ta.sma(df["close"], length=10)
    df["sma10_pct"] = sma10.pct_change()
    df["sma10_pct_sma"] = ta.sma(df["sma10_pct"])
    df["sma10_pct_ema"] = ta.ema(df["sma10_pct"])
    df["sma10_above_close"] = ta.above(sma10, df["close"])
    df["sma10_below_close"] = ta.below(sma10, df["close"])
    df["sma10_cross_above_close"] = ta.cross(sma10, df["close"], above=True)
    df["sma10_cross_below_close"] = ta.cross(sma10, df["close"], above=False)

    sma20 = ta.sma(df["close"], length=20)
    df["sma20_pct"] = sma20.pct_change()
    df["sma20_pct_sma"] = ta.sma(df["sma20_pct"])
    df["sma20_pct_ema"] = ta.ema(df["sma20_pct"])
    df["sma20_above_close"] = ta.above(sma20, df["close"])
    df["sma20_below_close"] = ta.below(sma20, df["close"])
    df["sma20_cross_above_close"] = ta.cross(sma20, df["close"], above=True)
    df["sma20_cross_below_close"] = ta.cross(sma20, df["close"], above=False)
    df["sma20_cross_above_sma10"] = ta.cross(sma20, sma10, above=True)
    df["sma20_cross_below_sma10"] = ta.cross(sma20, sma10, above=False)

    sma50 = ta.sma(df["close"], length=50)
    df["sma50_pct"] = sma50.pct_change()
    df["sma50_pct_sma"] = ta.sma(df["sma50_pct"])
    df["sma50_pct_ema"] = ta.ema(df["sma50_pct"])
    df["sma50_above_close"] = ta.above(sma50, df["close"])
    df["sma50_below_close"] = ta.below(sma50, df["close"])
    df["sma50_cross_above_close"] = ta.cross(sma50, df["close"], above=True)
    df["sma50_cross_below_close"] = ta.cross(sma50, df["close"], above=False)
    df["sma50_cross_above_sma20"] = ta.cross(sma50, sma20, above=True)
    df["sma50_cross_below_sma20"] = ta.cross(sma50, sma20, above=False)


def ema_feature_engineering(df):
    ema10 = ta.ema(df["close"], length=10)
    df["ema10_pct"] = ema10.pct_change()
    df["ema10_pct_sma"] = ta.sma(df["ema10_pct"])
    df["ema10_pct_ema"] = ta.ema(df["ema10_pct"])
    df["ema10_cross_above_close"] = ta.cross(ema10, df["close"], above=True)
    df["ema10_cross_below_close"] = ta.cross(ema10, df["close"], above=False)
    df["ema10_above_close"] = ta.above(ema10, df["close"])
    df["ema10_below_close"] = ta.below(ema10, df["close"])

    ema20 = ta.ema(df["close"], length=20)
    df["ema20_pct"] = ema20.pct_change()
    df["ema20_pct_sma"] = ta.sma(df["ema20_pct"])
    df["ema20_pct_ema"] = ta.ema(df["ema20_pct"])
    df["ema20_above_close"] = ta.above(ema20, df["close"])
    df["ema20_below_close"] = ta.below(ema20, df["close"])
    df["ema20_cross_above_close"] = ta.cross(ema20, df["close"], above=True)
    df["ema20_cross_below_close"] = ta.cross(ema20, df["close"], above=False)
    df["ema20_cross_above_ema10"] = ta.cross(ema20, ema10, above=True)
    df["ema20_cross_below_ema10"] = ta.cross(ema20, ema10, above=False)

    ema50 = ta.ema(df["close"], length=50)
    df["ema50_pct"] = ema50.pct_change()
    df["ema50_pct_sma"] = ta.sma(df["ema50_pct"])
    df["ema50_pct_ema"] = ta.ema(df["ema50_pct"])
    df["ema50_above_close"] = ta.above(ema50, df["close"])
    df["ema50_below_close"] = ta.below(ema50, df["close"])
    df["ema50_cross_above_close"] = ta.cross(ema50, df["close"], above=True)
    df["ema50_cross_below_close"] = ta.cross(ema50, df["close"], above=False)
    df["ema50_cross_above_ema20"] = ta.cross(ema50, ema20, above=True)
    df["ema50_cross_below_ema20"] = ta.cross(ema50, ema20, above=False)


def overlap_feature_engineering(df):
    ohlc4_feature_engineering(df)
    close_feature_engineering(df)
    sma_feature_engineering(df)
    ema_feature_engineering(df)
