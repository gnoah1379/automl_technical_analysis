import pandas as pd
import pandas_ta as ta
from features import feature_engineering
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


def target(df, periods):
    df["target"] = df['close'].pct_change(periods).shift(-periods)


def main():
    # Load data
    df = pd.read_csv('data/ETC-USDT.csv')
    target(df, 7)
    feature_engineering(df)
    df.dropna(inplace=True)
    df.drop(columns=['time', 'open', 'high', 'low', 'close', 'volume'], inplace=True)
    df.tail(100).to_csv('data/features.csv', index=False)


def save_test_results(x_test, y_test, y_predict):
    x_test['target'] = y_test
    x_test['predict'] = y_predict
    x_test.to_csv('test_results.csv', index=False)


if __name__ == '__main__':
    main()
