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
    relevant_columns = ['time', 'id_command', 'iq_command', 'motor_speed_command',
                        'rotation_angle_command', 'id_feedback', 'iq_feedback',
                        'motor_speed_feedback', 'rotation_angle_feedback']
    data = data[relevant_columns]

    # # 创建滞后变量(5个)
    # for col in ['id_feedback', 'iq_feedback', 'motor_speed_feedback', 'rotation_angle_feedback']:
    #     for i in range(1, lags + 1):
    #         data[f'{col}_lag_{i}'] = data[col].shift(i)

    # 创建滞后变量(100个)
    lag_data = {}
    for col in ['id_feedback', 'iq_feedback', 'motor_speed_feedback', 'rotation_angle_feedback']:
        for i in range(1, lags + 1):
            lag_data[f'{col}_lag_{i}'] = data[col].shift(i)

    # 添加滞后变量到原始数据框
    data = pd.concat([data, pd.DataFrame(lag_data)], axis=1)

    # 以上创建完毕，下面开始处理数据
    # 去掉含有NA的行
    data.dropna(inplace=True)

    # 数据归一化
    feature_cols = ['time', 'id_command', 'iq_command', 'motor_speed_command', 'rotation_angle_command'] + \
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


# def load_and_preprocess_data_multi_output_default_lags(file_path):
#     data = pd.read_csv(file_path)
#     relevant_columns = ['time', 'id_command', 'iq_command', 'motor_speed_command',
#                         'rotation_angle_command', 'id_feedback', 'iq_feedback',
#                         'motor_speed_feedback', 'rotation_angle_feedback']
#     data = data[relevant_columns]
#
#     # 数据集划分
#     train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
#
#     # 使用不等间隔的滞后值
#     near_lags = list(range(1, 101)) + [150, 200, 250, 300, 400, 500]
#     far_lags = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]
#     lags = near_lags + far_lags
#
#     # 在训练集上计算滞后特征并删除含有空值的数据点
#     for col in ['id_feedback', 'iq_feedback', 'motor_speed_feedback', 'rotation_angle_feedback']:
#         for lag in lags:
#             train_data[f'{col}_lag_{lag}'] = train_data[col].shift(lag)
#
#     train_data.dropna(inplace=True)
#
#     # 在测试集上使用训练集的最后一部分数据来计算滞后特征
#     for col in ['id_feedback', 'iq_feedback', 'motor_speed_feedback', 'rotation_angle_feedback']:
#         for lag in lags:
#             test_data[f'{col}_lag_{lag}'] = test_data[col].combine_first(
#                 train_data[col].iloc[-lag:].reset_index(drop=True))
#
#     feature_cols = ['time', 'id_command', 'iq_command', 'motor_speed_command', 'rotation_angle_command'] + [
#         f'{col}_lag_{lag}' for col in ['id_feedback', 'iq_feedback', 'motor_speed_feedback', 'rotation_angle_feedback']
#         for lag in lags]
#     output_cols = ['id_feedback', 'iq_feedback', 'motor_speed_feedback', 'rotation_angle_feedback']
#
#     X_train = train_data[feature_cols]
#     y_train = train_data[output_cols]
#     X_test = test_data[feature_cols]
#     y_test = test_data[output_cols]
#
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
#
#     return X_train, X_test, y_train, y_test

def load_and_preprocess_data_multi_output_default_lags(file_path):
    data = pd.read_csv(file_path)
    relevant_columns = ['time', 'id_command', 'iq_command', 'motor_speed_command',
                        'rotation_angle_command', 'id_feedback', 'iq_feedback',
                        'motor_speed_feedback', 'rotation_angle_feedback']
    data = data[relevant_columns]

    # 使用不等间隔的滞后值
    near_lags = list(range(1, 101)) + [150, 200, 250, 300, 400, 500]
    far_lags = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]
    lags = near_lags + far_lags

    # 构建所有滞后特征
    lag_data = {}
    for col in ['id_feedback', 'iq_feedback', 'motor_speed_feedback', 'rotation_angle_feedback']:
        for lag in lags:
            lag_data[f'{col}_lag_{lag}'] = data[col].shift(lag)

    # 将滞后特征加入原始数据
    data = pd.concat([data, pd.DataFrame(lag_data)], axis=1)

    # 删除含有空值的行
    data.dropna(inplace=True)

    # 数据集划分
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

    # 测试集继承训练集的滞后特征
    for col in ['id_feedback', 'iq_feedback', 'motor_speed_feedback', 'rotation_angle_feedback']:
        for lag in lags:
            test_data[f'{col}_lag_{lag}'] = train_data[f'{col}_lag_{lag}']

    feature_cols = ['time', 'id_command', 'iq_command', 'motor_speed_command', 'rotation_angle_command'] + [
        f'{col}_lag_{lag}' for col in ['id_feedback', 'iq_feedback', 'motor_speed_feedback', 'rotation_angle_feedback']
        for lag in lags]
    output_cols = ['id_feedback', 'iq_feedback', 'motor_speed_feedback', 'rotation_angle_feedback']

    X_train = train_data[feature_cols]
    y_train = train_data[output_cols]
    X_test = test_data[feature_cols]
    y_test = test_data[output_cols]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X_train_np, X_test_np, y_train_np, y_test_np = load_and_preprocess_data_multi_output_default_lags(
        "./data/多数据源位置预测_all.csv")