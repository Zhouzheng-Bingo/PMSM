import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(file_path, lags=5):
    data = pd.read_csv(file_path, encoding='gbk')
    print(data.columns)
    # 创建滞后变量
    for i in range(1, lags + 1):
        # data[f'指令_lag_{i}'] = data['指令'].shift(i)
        data[f'实际_lag_{i}'] = data['实际'].shift(i)
    data.dropna(inplace=True)

    # 分割数据
    # features = ['指令'] + [f'指令_lag_{i}' for i in range(1, lags + 1)] + [f'实际_lag_{i}' for i in range(1, lags + 1)]
    features = ['指令'] + [f'实际_lag_{i}' for i in range(1, lags + 1)]
    X = data[features]
    y = data['实际']
    # print(X.shape, y.shape)
    # 数据归一化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    return X_train, X_test, y_train, y_test


# id_iq数据预处理部分
def load_and_preprocess_data_id(file_path, lags=5):
    data = pd.read_csv(file_path)

    # 选择相关的列
    relevant_columns = ['time', 'id_command', 'iq_command', 'id_feedback']
    data = data[relevant_columns]

    # 创建反馈值的滞后变量
    lag_data = pd.concat([data[col].shift(i) for i in range(1, lags + 1) for col in ['id_feedback']], axis=1)
    lag_columns = [f'{col}_lag_{i}' for i in range(1, lags + 1) for col in ['id_feedback']]
    lag_data.columns = lag_columns
    data = pd.concat([data, lag_data], axis=1)

    # 去掉含有NA的行
    data.dropna(inplace=True)

    # 数据归一化
    feature_cols = ['time', 'id_command', 'iq_command'] + [f'id_feedback_lag_{i}' for i in range(1, lags + 1)]
    output_cols = 'id_feedback'

    X = data[feature_cols]
    y = data[output_cols]
    # print(X.shape, y.shape)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    return X_train, X_test, y_train, y_test


# id_iq数据预处理部分
def load_and_preprocess_data_iq(file_path, lags=5):
    data = pd.read_csv(file_path)

    # 选择相关的列
    relevant_columns = ['time', 'id_command', 'iq_command', 'iq_feedback']
    data = data[relevant_columns]

    # 创建反馈值的滞后变量
    lag_data = pd.concat([data[col].shift(i) for i in range(1, lags + 1) for col in ['iq_feedback']], axis=1)
    lag_columns = [f'{col}_lag_{i}' for i in range(1, lags + 1) for col in ['iq_feedback']]
    lag_data.columns = lag_columns
    data = pd.concat([data, lag_data], axis=1)

    # 去掉含有NA的行
    data.dropna(inplace=True)

    # 数据归一化
    feature_cols = ['time', 'id_command', 'iq_command'] + [f'iq_feedback_lag_{i}' for i in range(1, lags + 1)]
    output_cols = 'iq_feedback'

    X = data[feature_cols]
    y = data[output_cols]
    # print(X.shape, y.shape)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    return X_train, X_test, y_train, y_test


def load_and_preprocess_data_angle(file_path, lags=5):
    """
        这里的25是因为我们有1个 time 特征,
        1个rotation_angle_command特征,1个id_command，1个iq_command和1个motor_speed_command
        每个滞后变量5个，共4个这样的变量组
    """

    data = pd.read_csv(file_path)

    # 选择相关的列
    relevant_columns = ['time', 'id_feedback', 'iq_feedback', 'motor_speed_feedback',
                        'rotation_angle_command', 'rotation_angle_feedback',
                        'id_command', 'iq_command', 'motor_speed_command']
    data = data[relevant_columns]

    # 创建滞后变量
    for col in ['id_feedback', 'iq_feedback', 'motor_speed_feedback', 'rotation_angle_feedback']:
        for i in range(1, lags + 1):
            data[f'{col}_lag_{i}'] = data[col].shift(i)

    # 去掉含有NA的行
    data.dropna(inplace=True)

    # 数据归一化
    feature_cols = ['time', 'rotation_angle_command', 'id_command', 'iq_command', 'motor_speed_command'] + \
                   [f'{col}_lag_{i}' for col in
                    ['id_feedback', 'iq_feedback', 'motor_speed_feedback', 'rotation_angle_feedback'] for i in
                    range(1, lags + 1)]
    output_cols = 'rotation_angle_feedback'

    X = data[feature_cols]
    y = data[output_cols]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    return X_train, X_test, y_train, y_test


def load_and_preprocess_data_multi_output(file_path, lags=5):
    """
        这里的25是因为我们有1个 time 特征,
        1个rotation_angle_command特征,1个id_command，1个iq_command和1个motor_speed_command
        每个滞后变量5个，共4个这样的变量组
    """

    data = pd.read_csv(file_path)

    # 选择相关的列
    relevant_columns = ['time', 'id_feedback', 'iq_feedback', 'motor_speed_feedback',
                        'rotation_angle_command', 'rotation_angle_feedback',
                        'id_command', 'iq_command', 'motor_speed_command']
    data = data[relevant_columns]

    # 创建滞后变量
    for col in ['id_feedback', 'iq_feedback', 'motor_speed_feedback', 'rotation_angle_feedback']:
        for i in range(1, lags + 1):
            data[f'{col}_lag_{i}'] = data[col].shift(i)

    # 去掉含有NA的行
    data.dropna(inplace=True)

    # 数据归一化
    feature_cols = ['time', 'rotation_angle_command', 'id_command', 'iq_command', 'motor_speed_command'] + \
                   [f'{col}_lag_{i}' for col in
                    ['id_feedback', 'iq_feedback', 'motor_speed_feedback', 'rotation_angle_feedback'] for i in
                    range(1, lags + 1)]
    output_cols = ['id_feedback', 'iq_feedback', 'motor_speed_feedback', 'rotation_angle_feedback']

    X = data[feature_cols]
    y = data[output_cols]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    return X_train, X_test, y_train, y_test