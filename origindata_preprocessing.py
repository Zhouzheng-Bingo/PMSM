import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(file_path):
    """
    Load data and perform preprocessing for neural network input without additional feature engineering.
    """
    # Load the data
    data = pd.read_csv(file_path)
    print(f"Original data shape: {data.shape}")

    # Define the columns for input features and output (feedback)
    input_columns = ['id_command', 'iq_command', 'motor_speed_command', 'rotation_angle_command']
    output_columns = ['id_feedback', 'iq_feedback', 'motor_speed_feedback', 'rotation_angle_feedback']

    # Standardize the input features
    scaler = StandardScaler()
    data[input_columns] = scaler.fit_transform(data[input_columns])

    # Split the data into training and test sets (80% for training, 20% for testing)
    split_index = int(0.8 * len(data))
    X_train = data[input_columns][:split_index]
    X_test = data[input_columns][split_index:]
    y_train = data[output_columns][:split_index]
    y_test = data[output_columns][split_index:]
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_and_preprocess_data('./data/多数据源位置预测_all_subset.csv')

    # Optionally, print some statistics or previews
    print("Training Set Feature Statistics:")
    print(X_train.describe().transpose())
    print("\nTest Set Feature Statistics:")
    print(X_test.describe().transpose())
