import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(file_path, lags=1):
    data = pd.read_csv(file_path)

    # 选择相关的列
    relevant_columns = ['time', 'id_command', 'id_feedback', 'iq_command', 'iq_feedback']
    data = data[relevant_columns]

    # 创建反馈值的滞后变量
    for i in range(1, lags + 1):
        data[f'id_feedback_lag_{i}'] = data['id_feedback'].shift(i)
        data[f'iq_feedback_lag_{i}'] = data['iq_feedback'].shift(i)

    data.dropna(inplace=True)

    # 分割数据
    feature_cols = ['time', 'id_command', 'iq_command'] + [f'id_feedback_lag_{i}' for i in range(1, lags + 1)] + [
        f'iq_feedback_lag_{i}' for i in range(1, lags + 1)]
    output_cols = ['id_feedback', 'iq_feedback']

    X = data[feature_cols]
    y = data[output_cols]

    # 数据归一化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    return X_train, X_test, y_train, y_test
