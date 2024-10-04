import pickle

import pandas as pd
import pandas_ta as ta
from flaml import AutoML
from flaml.automl.ml import sklearn_metric_loss_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error, \
    mean_absolute_error

from features import feature_engineering
from warnings import simplefilter

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


def target(df, periods):
    # target is 1 if avg pct of close in next 7 days is positive
    pct = df['close'].pct_change() * 100
    pct_p20 = pct.quantile(0.2)
    pct_40 = pct.quantile(0.4)
    pct_60 = pct.quantile(0.6)
    pct_p80 = pct.quantile(0.8)

    pct_avg = pct.rolling(periods).mean()
    pct_avg_next_period = pct_avg.shift(-periods)

    # if pct_avg_next_period < pct_p20: -2
    # elif pct_p20 > pct_avg_next_period < pct_40: -1
    # elif pct_40 > pct_avg_next_period < pct_60: 0
    # elif pct_60 > pct_avg_next_period < pct_p80: 1
    # else: 2
    df["target"] = pct_avg_next_period.apply(lambda x:
                                             -2 if x < pct_p20 else
                                             -1 if pct_p20 < x < pct_40 else
                                             0 if pct_40 < x < pct_60 else
                                             1 if pct_60 < x < pct_p80 else 2)
    # df["target"] = pct_avg_next_period.apply(lambda x: 1 if x > 0 else 0)


def debug_target():
    periods = 7
    df = pd.read_csv('data/ETC-USDT.csv')
    target(df, periods)
    df["close_pct"] = df['close'].pct_change() * 100
    df["close_pct_avg_next_period"] = df["close_pct"].rolling(periods).mean().shift(-periods)
    df.to_csv("debug_target.csv")


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
        "time_budget": 60 * 5,
        "metric": "accuracy",
        "task": "classification",
        "estimator_list": ["lgbm"],
        "log_file_name": "automl.log",
    }
    old_automl = load_model()
    automl = AutoML(**settings)
    print("training...")
    automl.fit(X_train, y_train, starting_points=old_automl.best_config_per_estimator)
    print("testing...")
    y_pred = automl.predict(X_test)
    print("saving...")
    save_test_results(X_test, y_test, y_pred)
    save_model(automl)
    save_feature_importance_percent(automl)
    print_metric_scores(y_test, y_pred)


def save_model(automl):
    with open("automl.pkl", "wb") as f:
        pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)


def load_model():
    with open("automl.pkl", "rb") as f:
        return pickle.load(f)


def save_feature_importance_percent(automl):
    feature_importance = pd.DataFrame()
    feature_importance["feature"] = automl.feature_names_in_
    importance = automl.feature_importances_
    feature_importance["importance"] = (importance / importance.sum()) * 100
    feature_importance.sort_values("importance", ascending=False, inplace=True)
    feature_importance.to_csv("feature_importance_percent.csv", index=False)


def print_metric_scores(y_test, y_pred):
    # metrics = ["accuracy", "precision", "recall", "f1", "r2", "mse", "mae"]
    # for metric in metrics:
    #     print(f"{metric}: {sklearn_metric_loss_score(metric, y_test, y_pred)}")
    print(f"accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"r2: {r2_score(y_test, y_pred)}")
    print(f"mse: {mean_squared_error(y_test, y_pred)}")
    print(f"mae: {mean_absolute_error(y_test, y_pred)}")


def save_test_results(X_test, y_test, y_pred):
    X_test["y_test"] = y_test
    X_test["y_pred"] = y_pred
    X_test.to_csv("test_results.csv")


if __name__ == '__main__':
    df = pd.read_csv('test_results.csv')
    pct = df["close_pct"]
    y_test = df["y_test"]
    y_pred = df["y_pred"]
    pct_p20 = pct.quantile(0.2)
    pct_40 = pct.quantile(0.4)
    pct_50 = pct.quantile(0.5)
    pct_60 = pct.quantile(0.6)
    pct_p80 = pct.quantile(0.8)
    print("===== quantiles =====")
    print(f"pct_p20: {pct_p20}")
    print(f"pct_40: {pct_40}")
    print(f"pct_50: {pct_50}")
    print(f"pct_60: {pct_60}")
    print(f"pct_p80: {pct_p80}")
    print("===== metrics =====")
    print_metric_scores(y_test, y_pred)
    print("===== overall =====")
    # percent when y_test = y_pred
    print(f"chính xác: {df[(df["y_test"] == df["y_pred"])].shape[0] / df.shape[0] * 100}")
    # percent when y_test > y_pred
    print(f"đánh giá thấp: {df[(df['y_test'] > df['y_pred'])].shape[0] / df.shape[0] * 100}")
    # percent when y_test < y_pred
    print(f"đánh giá cao: {df[(df['y_test'] < df['y_pred'])].shape[0] / df.shape[0] * 100}")

    print("===== lỗi  =====")
    print(f"gây lỗ: {df[(df["y_test"] <= 0) & (df["y_pred"] > 0)].shape[0] / df.shape[0] * 100}")
    print(f"không gây lỗ: {df[(df["y_test"] > 0) & (df["y_pred"] <= 0)].shape[0] / df.shape[0] * 100}")

    print(f"mất lãi ít: {df[(df["y_test"] == 0) & (df["y_pred"] < 0)].shape[0] / df.shape[0] * 100}")
    print(f"mất lãi nhiều: {df[(df["y_test"] > 0) & (df["y_pred"] == 0)].shape[0] / df.shape[0] * 100}")

    # print_metric_scores(df["y_test"], df["y_pred"])
    # automl = load_model()
    # save_feature_importance_percent(automl)
    # train()
    # debug_target()
