import pandas as pd
import pandas_ta as ta


def macd_feature_engineering(df):
    macd_df = df.ta.macd(signal_indicators=True)
    macd = macd_df["MACD_12_26_9"]
    macd_signal = macd_df["MACDh_12_26_9"]
    macd_hist = macd_df["MACDs_12_26_9"]
    df["macd_pct"] = macd.pct_change()
    df["macd_pct_sma"] = ta.sma(df["macd_pct"])
    df["macd_pct_ema"] = ta.ema(df["macd_pct"])
    df["macd_signal_pct"] = macd_signal.pct_change()
    df["macd_signal_pct_sma"] = ta.sma(df["macd_signal_pct"])
    df["macd_signal_pct_ema"] = ta.ema(df["macd_signal_pct"])
    df["macd_positive"] = ta.above_value(macd, 0)
    df["macd_hist_positive"] = ta.above_value(macd_hist, 0)
    df["macd_cross_above_signal"] = ta.cross(macd, macd_signal, above=True)
    df["macd_cross_below_signal"] = ta.cross(macd, macd_signal, above=False)


def cci_feature_engineering(df):
    cci = df.ta.cci()
    df["cci"] = cci
    df["cci_pct"] = cci.pct_change()
    df["cci_pct_sma"] = ta.sma(df["cci_pct"])
    df["cci_pct_ema"] = ta.ema(df["cci_pct"])
    df["cci_overbought"] = ta.above_value(cci, 100)
    df["cci_oversold"] = ta.below_value(cci, -100)


def rsi_feature_engineering(df):
    rsi = df.ta.rsi()
    df["rsi"] = rsi
    df["rsi_pct"] = rsi.pct_change()
    df["rsi_pct_sma"] = ta.sma(df["rsi_pct"])
    df["rsi_pct_ema"] = ta.ema(df["rsi_pct"])
    df["rsi_overbought"] = ta.above_value(rsi, 70)
    df["rsi_oversold"] = ta.below_value(rsi, 30)

    stochrsi = df.ta.stochrsi()
    df["stochrsi_k"] = stochrsi["STOCHRSIk_14_14_3_3"]
    df["stochrsi_d"] = stochrsi["STOCHRSId_14_14_3_3"]
    df["stochrsi_k_overbought"] = ta.above_value(df["stochrsi_k"], 80)
    df["stochrsi_k_oversold"] = ta.below_value(df["stochrsi_k"], 20)
    df["stochrsi_d_overbought"] = ta.above_value(df["stochrsi_d"], 80)
    df["stochrsi_d_oversold"] = ta.below_value(df["stochrsi_d"], 20)
    df["stochrsi_k_cross_above_d"] = ta.cross(df["stochrsi_k"], df["stochrsi_d"], above=True)
    df["stochrsi_k_cross_below_d"] = ta.cross(df["stochrsi_k"], df["stochrsi_d"], above=False)
    df["stochrsi_k_pct"] = df["stochrsi_k"].pct_change()
    df["stochrsi_d_pct"] = df["stochrsi_d"].pct_change()
    df["stochrsi_k_pct_sma"] = ta.sma(df["stochrsi_k_pct"])
    df["stochrsi_d_pct_sma"] = ta.sma(df["stochrsi_d_pct"])


def roc_feature_engineering(df):
    roc = df.ta.roc()
    df["roc_pct"] = roc.pct_change()
    df["roc_pct_sma"] = ta.sma(df["roc_pct"])
    df["roc_pct_ema"] = ta.ema(df["roc_pct"])
    df["roc_positive"] = ta.above_value(roc, 0)
    df["roc_cross_above_zero"] = ta.cross_value(roc, 0, above=True)
    df["roc_cross_below_zero"] = ta.cross_value(roc, 0, above=False)
    df["roc_above_zero"] = ta.above_value(roc, 0)
    df["roc_below_zero"] = ta.below_value(roc, 0)


def bias_feature_engineering(df):
    bias = df.ta.bias()
    df["bias"] = bias


def momentum_feature_engineering(df):
    macd_feature_engineering(df)
    rsi_feature_engineering(df)
    bias_feature_engineering(df)
    roc_feature_engineering(df)
    cci_feature_engineering(df)
