import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def create_lag_features(data, feature_columns, lags):
    return pd.concat(
        {f"{col}_lag_{lag}": data[col].shift(lag) for lag in lags for col in feature_columns},
        axis=1
    )


def ensure_column_names_are_strings(df):
    df.columns = df.columns.map(lambda x: '_'.join(tuple(map(str, x))) if isinstance(x, tuple) else str(x))
    return df


def compute_windowed_features(data, feedback_columns, window_starts, window_size, sampling_rate):
    windowed_features = []
    for start in window_starts:
        end = start + window_size
        window_data = data.loc[start:end, feedback_columns]
        features = window_data.agg(['mean', 'std', 'min', 'max']).unstack()
        windowed_features.append(features)
    return pd.DataFrame(windowed_features)


def load_and_preprocess_data_multi_output(file_path, window_size=0.2, sampling_rate=1e6, lags=[1, 2, 3, 4, 5]):
    data = pd.read_csv(file_path)
    print(f"Original data shape: {data.shape}")

    # Command and feedback columns
    command_columns = ['id_command', 'iq_command', 'motor_speed_command', 'rotation_angle_command']
    feedback_columns = ['id_feedback', 'iq_feedback', 'motor_speed_feedback', 'rotation_angle_feedback']

    # Identify points where the command changes
    command_change_points = data[command_columns].diff().abs().sum(axis=1).ne(0)
    print(f"Number of command change points: {command_change_points.sum()}")

    # 这里我们计算窗口的开始位置，确保没有负数索引。
    window_starts = command_change_points[command_change_points].index
    adjusted_window_size = int(window_size * sampling_rate / 2)

    # 确保没有负数索引
    window_starts = window_starts[window_starts >= adjusted_window_size] - adjusted_window_size
    print(f"Number of window starts after adjustment: {len(window_starts)}")

    # Compute windowed features
    windowed_features_df = compute_windowed_features(data, feedback_columns, window_starts, adjusted_window_size,
                                                     sampling_rate)
    print(f"Windowed features shape: {windowed_features_df.shape}")

    # Compute differential features
    differential_features = data[feedback_columns].diff().fillna(0)
    print(f"Differential features shape: {differential_features.shape}")

    # Create lag features
    lag_features = create_lag_features(data, feedback_columns, lags)
    print(f"Lag features shape: {lag_features.shape}")

    # Concatenate all features
    all_features = pd.concat([windowed_features_df, differential_features, lag_features], axis=1)
    all_features = ensure_column_names_are_strings(all_features)
    print(f"All features shape (before dropping NA): {all_features.shape}")

    # Drop rows with NaN values which may be introduced by shifting operations
    all_features.dropna(inplace=True)
    print(f"All features shape (after dropping NA): {all_features.shape}")

    # Scale all features
    scaler = StandardScaler()
    all_features_scaled = scaler.fit_transform(all_features)

    # Align targets to the same index as windowed features
    aligned_targets = data.loc[windowed_features_df.index, feedback_columns]
    print(f"Aligned targets shape: {aligned_targets.shape}")

    aligned_targets = aligned_targets.loc[all_features.index]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(all_features_scaled, aligned_targets, test_size=0.2,
                                                        shuffle=False)

    # Print shapes of the resulting splits
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test

#
# # 这是降采样函数
# def downsample_data(data, downsample_rate):
#     """
#     Downsample the dataset by a specified rate.
#
#     :param data: Original DataFrame.
#     :param downsample_rate: A float representing the fraction of the data to keep.
#     :return: Downsampled DataFrame.
#     """
#     if downsample_rate <= 0 or downsample_rate > 1:
#         raise ValueError("Downsample rate must be between 0 and 1.")
#
#     downsampled_data = data.sample(frac=downsample_rate, random_state=1)
#     return downsampled_data.reset_index(drop=True)
#
#
# def load_and_preprocess_data_multi_output(file_path, window_size=0.2, sampling_rate=1e6, lags=[1, 2, 3, 4, 5],
#                                           downsample_rate=0.1):
#     data = pd.read_csv(file_path)
#     print(f"Original data shape: {data.shape}")
#
#     # 如果指定了降采样率，则进行降采样
#     if downsample_rate < 1.0:
#         print(f"Original data shape: {data.shape}")
#         data = downsample_data(data, downsample_rate)
#         print(f"Downsampled data shape: {data.shape}")
#
#     # Command and feedback columns
#     command_columns = ['id_command', 'iq_command', 'motor_speed_command', 'rotation_angle_command']
#     feedback_columns = ['id_feedback', 'iq_feedback', 'motor_speed_feedback', 'rotation_angle_feedback']
#
#     # Identify points where the command changes
#     command_change_points = data[command_columns].diff().abs().sum(axis=1).ne(0)
#     print(f"Number of command change points: {command_change_points.sum()}")
#
#     # 这里我们计算窗口的开始位置，确保没有负数索引。
#     # 由于降采样，我们可能需要调整窗口大小或开始点的计算。
#     half_window_samples = int(window_size * sampling_rate / 2)
#     window_starts = command_change_points[command_change_points].index
#
#     # 如果需要，调整窗口大小
#     adjusted_window_size = int(window_size * sampling_rate / 2 * downsample_rate)
#
#     # 确保没有负数索引
#     window_starts = window_starts[window_starts >= adjusted_window_size] - adjusted_window_size
#     print(f"Number of window starts after adjustment: {len(window_starts)}")
#
#     # Compute windowed features
#     windowed_features_df = compute_windowed_features(data, feedback_columns, window_starts,
#                                                      int(window_size * sampling_rate), sampling_rate)
#     print(f"Windowed features shape: {windowed_features_df.shape}")
#
#     # Compute differential features
#     differential_features = data[feedback_columns].diff().fillna(0)
#     print(f"Differential features shape: {differential_features.shape}")
#
#     # Create lag features
#     lag_features = create_lag_features(data, feedback_columns, lags)
#     print(f"Lag features shape: {lag_features.shape}")
#
#     # Concatenate all features
#     all_features = pd.concat([windowed_features_df, differential_features, lag_features], axis=1)
#     all_features = ensure_column_names_are_strings(all_features)
#     print(f"All features shape (before dropping NA): {all_features.shape}")
#
#     # Drop rows with NaN values which may be introduced by shifting operations
#     all_features.dropna(inplace=True)
#     print(f"All features shape (after dropping NA): {all_features.shape}")
#
#     # Scale all features
#     scaler = StandardScaler()
#     all_features_scaled = scaler.fit_transform(all_features)
#
#     # Align targets to the same index as windowed features
#     aligned_targets = data.loc[windowed_features_df.index, feedback_columns]
#     print(f"Aligned targets shape: {aligned_targets.shape}")
#
#     aligned_targets = aligned_targets.loc[all_features.index]
#
#     # Split the data into training and test sets
#     X_train, X_test, y_train, y_test = train_test_split(all_features_scaled, aligned_targets, test_size=0.2,
#                                                         shuffle=False)
#
#     # Print shapes of the resulting splits
#     print(f"X_train shape: {X_train.shape}")
#     print(f"X_test shape: {X_test.shape}")
#     print(f"y_train shape: {y_train.shape}")
#     print(f"y_test shape: {y_test.shape}")
#
#     return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # Call the modified function with the subset data file path
    # We are using a subset of the data for this execution, assuming it mirrors the full dataset's structure
    load_and_preprocess_data_multi_output('./data/多数据源位置预测_all.csv')
