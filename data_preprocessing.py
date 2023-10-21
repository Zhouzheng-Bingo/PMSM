import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(file_path, lags=5):
    data = pd.read_csv(file_path, encoding='gbk')
    print(data.columns)
    # 创建滞后变量
    for i in range(1, lags + 1):
        data[f'指令_lag_{i}'] = data['指令'].shift(i)
        data[f'实际_lag_{i}'] = data['实际'].shift(i)
    data.dropna(inplace=True)

    # 分割数据
    features = ['指令'] + [f'指令_lag_{i}' for i in range(1, lags + 1)] + [f'实际_lag_{i}' for i in range(1, lags + 1)]
    X = data[features]
    y = data['实际']

    # 数据归一化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    return X_train, X_test, y_train, y_test
