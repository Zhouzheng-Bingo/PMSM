import pandas as pd
from sklearn.preprocessing import StandardScaler

def create_lag_features(data, feature_columns, lags, fill_value=0):
    """
    Create lag features for the dataset, ensuring that initial values are set to zero.
    """
    lag_data = pd.DataFrame(index=data.index)
    for col in feature_columns:
        for lag in lags:
            lag_data[f"{col}_lag_{lag}"] = data[col].shift(lag)
    lag_data.fillna(fill_value, inplace=True)
    return lag_data

def load_and_preprocess_data_multi_output(file_path, lags=[1000, 2000, 3000, 4000, 5000]):
    """
    Load data and perform preprocessing including creation of lag features.
    """
    data = pd.read_csv(file_path)

    print(f"Original data shape: {data.shape}")

    command_columns = ['id_command', 'iq_command', 'motor_speed_command', 'rotation_angle_command']
    feedback_columns = ['id_feedback', 'iq_feedback', 'motor_speed_feedback', 'rotation_angle_feedback']

    lag_features = create_lag_features(data, feedback_columns, lags)
    print("Lag features preview:")
    print(lag_features.head())

    print("Command features preview:")
    print(data[command_columns].head())

    all_features = pd.concat([data[command_columns], lag_features], axis=1)
    all_features = ensure_column_names_are_strings(all_features)

    scaler = StandardScaler()
    all_features_scaled = scaler.fit_transform(all_features)

    split_index = len(data) - int(0.2 * len(data))
    X_train_scaled = all_features_scaled[:split_index]
    X_test_scaled = all_features_scaled[split_index:]

    max_lag = max(lags)
    for i in range(max_lag):
        if i < len(X_test_scaled):
            X_test_scaled[i] = X_train_scaled[-(max_lag - i)]

    y_train = data[feedback_columns].iloc[:split_index]
    y_test = data[feedback_columns].iloc[split_index:]
    print(X_train_scaled.shape, X_test_scaled.shape, y_train.shape, y_test.shape)
    return X_train_scaled, X_test_scaled, y_train, y_test

def ensure_column_names_are_strings(df):
    """
    Ensure all column names are strings.
    """
    df.columns = df.columns.map(str)
    return df

if __name__ == '__main__':
    X_train_scaled, X_test_scaled, y_train, y_test = load_and_preprocess_data_multi_output('./data/多数据源位置预测_all_subset.csv')

    X_train_scaled_df = pd.DataFrame(X_train_scaled)
    X_test_scaled_df = pd.DataFrame(X_test_scaled)

    print("Training Set Feature Statistics:")
    print(X_train_scaled_df.describe().transpose())

    print("\nTest Set Feature Statistics:")
    print(X_test_scaled_df.describe().transpose())

    train_max = X_train_scaled_df.max().max()
    train_min = X_train_scaled_df.min().min()
    test_max = X_test_scaled_df.max().max()
    test_min = X_test_scaled_df.min().min()

    print("Training Set - Max Value:", train_max)
    print("Training Set - Min Value:", train_min)
    print("Test Set - Max Value:", test_max)
    print("Test Set - Min Value:", test_min)
