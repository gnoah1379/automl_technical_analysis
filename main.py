import pickle

import pandas as pd
import pandas_ta as ta
from flaml import AutoML
from flaml.automl.ml import sklearn_metric_loss_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, \
    mean_absolute_error, mean_absolute_percentage_error

from features import feature_engineering
from warnings import simplefilter

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


def target(df: pd.DataFrame, periods):
    pct_sum = df['close'].pct_change().rolling(periods).sum().shift(-periods) * 100

    p30 = pct_sum.quantile(0.3)
    p50 = pct_sum.quantile(0.5)
    p70 = pct_sum.quantile(0.7)

    df["target"] = pct_sum.apply(lambda x:
                                 -2 if x <= p30 else
                                 -1 if p30 < x <= p50 else
                                 1 if p50 < x <= p70 else
                                 2 if x > p70 else
                                 0)


def debug_target():
    periods = 7
    df = pd.read_csv('data/ETC-USDT.csv')
    # target(df, periods)
    df["close_pct_avg_7"] = df['close'].pct_change().rolling(periods).mean().shift(-periods) * 100
    df["close_pct_sum_7"] = df['close'].pct_change().rolling(periods).sum().shift(-periods) * 100
    df["close_pct_next_7"] = df['close'].pct_change(periods).shift(-periods) * 100
    # df["close_pct_7"] = df["close_pct"]

    df.to_csv("result/debug_target.csv")


def train():
    # Load data
    df = pd.read_csv('data/ETC-USDT.csv')
    target(df, 7)
    feature_engineering(df)
    df.dropna(inplace=True)
    df.drop(columns=['time', 'open', 'high', 'low', 'close', 'volume'], inplace=True)
    y = df["target"]
    X = df.drop(columns=["target"])
    # print(X.tail())
    # print(y.tail())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    settings = {
        "time_budget": 60 * 60,
        "metric": "accuracy",
        "task": "classification",
        "estimator_list": ["lgbm"],
        "log_file_name": "result/automl.log",
    }
    automl = AutoML(**settings)
    print("training...")
    automl.fit(X_train, y_train)
    print("testing...")
    y_pred = automl.predict(X_test)
    print("saving...")
    save_test_results(X_test, y_test, y_pred)
    save_model(automl)
    save_feature_importance_percent(automl)
    print_metric_scores(y_test, y_pred)


def save_model(automl):
    with open("result/automl.pkl", "wb") as f:
        pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)


def load_model():
    with open("result/automl.pkl", "rb") as f:
        return pickle.load(f)


def save_feature_importance_percent(automl):
    feature_importance = pd.DataFrame()
    feature_importance["feature"] = automl.feature_names_in_
    importance = automl.feature_importances_
    feature_importance["importance"] = (importance / importance.sum()) * 100
    feature_importance.sort_values("importance", ascending=False, inplace=True)
    feature_importance.to_csv("result/feature_importance_percent.csv", index=False)


def print_metric_scores(y_test, y_pred):
    print(f"accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"r2: {r2_score(y_test, y_pred)}")
    print(f"mse: {mean_squared_error(y_test, y_pred)}")
    print(f"mae: {mean_absolute_error(y_test, y_pred)}")
    print(f"mape: {mean_absolute_percentage_error(y_test, y_pred)}")


def save_test_results(X_test, y_test, y_pred):
    X_test["y_test"] = y_test
    X_test["y_pred"] = y_pred
    X_test.to_csv("result/test_results.csv")


if __name__ == '__main__':
    # print_metric_scores(df["y_test"], df["y_pred"])
    # automl = load_model()
    # save_feature_importance_percent(automl)
    train()
    # debug_target()
