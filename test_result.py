import pandas as pd


def test():
    df = pd.read_csv('result/test_results.csv')
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
    # print_metric_scores(y_test, y_pred)
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