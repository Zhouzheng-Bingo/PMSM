import pandas as pd
from sklearn.preprocessing import StandardScaler


def ensure_column_names_are_strings(df):
    """
    Ensure all column names are strings.
    """
    df.columns = df.columns.map(lambda x: '_'.join(tuple(map(str, x))) if isinstance(x, tuple) else str(x))
    return df


def create_lag_features(data, feature_columns, lags, fill_value=None):
    """
    Create lag features for the dataset.
    """
    lag_data = pd.DataFrame(index=data.index)
    for col in feature_columns:
        for lag in lags:
            lag_data[f"{col}_lag_{lag}"] = data[col].shift(lag)
    if fill_value is not None:
        lag_data.fillna(fill_value, inplace=True)  # Fill the initial missing values with a specified value
    return lag_data


def compute_windowed_features(data, feedback_columns, window_size, sampling_rate):
    """
    Compute windowed features for the dataset.
    """
    number_of_windows = int(len(data) / (window_size * sampling_rate))
    windowed_features = []
    for window_number in range(number_of_windows):
        start = window_number * int(window_size * sampling_rate)
        end = start + int(window_size * sampling_rate)
        window_data = data[feedback_columns].iloc[start:end]
        features = window_data.agg(['mean', 'std', 'min', 'max']).unstack()
        windowed_features.append(features)
    return pd.DataFrame(windowed_features).fillna(0)


def load_and_preprocess_data_multi_output(file_path, window_size=0.2, sampling_rate=1e6, lags=[1, 2, 3, 4, 5]):
    """
    Load data and perform preprocessing including creation of lag and windowed features.
    """
    # Load the data
    data = pd.read_csv(file_path)
    print(f"Original data shape: {data.shape}")

    # Command and feedback columns
    command_columns = ['id_command', 'iq_command', 'motor_speed_command', 'rotation_angle_command']
    feedback_columns = ['id_feedback', 'iq_feedback', 'motor_speed_feedback', 'rotation_angle_feedback']

    # Compute windowed features
    windowed_features_df = compute_windowed_features(data, feedback_columns, window_size, sampling_rate)
    print(f"Windowed features shape: {windowed_features_df.shape}")

    # Compute differential features
    differential_features = data[feedback_columns].diff().fillna(0)
    print(f"Differential features shape: {differential_features.shape}")

    # Create lag features
    # For the initial values of training set, fill with 0
    lag_features = create_lag_features(data, feedback_columns, lags, fill_value=0)
    print(f"Lag features shape: {lag_features.shape}")

    # Concatenate all features
    all_features = pd.concat([windowed_features_df, differential_features, lag_features], axis=1)
    all_features = ensure_column_names_are_strings(all_features)
    print(f"All features shape: {all_features.shape}")

    # Scale all features
    scaler = StandardScaler()
    all_features_scaled = scaler.fit_transform(all_features)

    # Split the data into training and test sets
    split_index = len(data) - int(window_size * sampling_rate)  # Index for splitting the data into train and test
    X_train_scaled = all_features_scaled[:split_index]
    X_test_scaled = all_features_scaled[split_index:]
    y_train = data[feedback_columns].iloc[:split_index]
    y_test = data[feedback_columns].iloc[split_index:]

    # Print shapes of the resulting splits
    print(f"X_train shape: {X_train_scaled.shape}")
    print(f"X_test shape: {X_test_scaled.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == '__main__':
    X_train_scaled, X_test_scaled, y_train, y_test = load_and_preprocess_data_multi_output('./data/多数据源位置预测_all.csv')

    # 转换为DataFrame以便于分析
    X_train_scaled_df = pd.DataFrame(X_train_scaled)
    X_test_scaled_df = pd.DataFrame(X_test_scaled)

    # 打印训练和测试集特征的统计信息
    print("Training Set Feature Statistics:")
    print(X_train_scaled_df.describe().transpose())  # 打印训练集特征的统计信息

    print("\nTest Set Feature Statistics:")
    print(X_test_scaled_df.describe().transpose())  # 打印测试集特征的统计信息

    X_train_scaled_df = pd.DataFrame(X_train_scaled)
    X_test_scaled_df = pd.DataFrame(X_test_scaled)

    # 查找训练集和测试集中的极值
    train_max = X_train_scaled_df.max().max()
    train_min = X_train_scaled_df.min().min()
    test_max = X_test_scaled_df.max().max()
    test_min = X_test_scaled_df.min().min()

    print("Training Set - Max Value:", train_max)
    print("Training Set - Min Value:", train_min)
    print("Test Set - Max Value:", test_max)
    print("Test Set - Min Value:", test_min)
