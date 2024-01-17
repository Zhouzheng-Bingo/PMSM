from bayes_opt import BayesianOptimization
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from origindata_preprocessing import load_and_preprocess_data
from origindata_resnet_multi_output import CombinedModel


# 确保数据预处理和模型文件在当前目录下或者在PYTHONPATH中

# def objective(lr, weight_decay):
#     model = CombinedModel(input_dim=1, hidden_dim=64, output_dim=4, num_blocks=22, num_heads=4).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#     criterion = nn.MSELoss()
#
#     # 训练模型
#     num_epochs = 50
#     for epoch in range(num_epochs):
#         model.train()  # Set model to training mode
#         running_loss = 0.0
#         for inputs, targets in train_loader:
#             inputs, targets = inputs.to(device), targets.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#
#         # 在每个epoch结束后计算验证集上的损失
#         validation_loss = 0.0
#         model.eval()  # Set model to evaluation mode
#         with torch.no_grad():
#             for inputs, targets in test_loader:
#                 inputs, targets = inputs.to(device), targets.to(device)
#                 outputs = model(inputs)
#                 loss = criterion(outputs, targets)
#                 validation_loss += loss.item()
#
#         # 这里可以添加早停或者保存模型的代码
#
#     return -validation_loss / len(test_loader)  # Return negative loss for maximization
#
#
# if __name__ == '__main__':
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     # 这里应当确保load_and_preprocess_data函数返回的是适合模型输入的数据
#     X_train, X_test, y_train, y_test = load_and_preprocess_data("./data/多数据源位置预测_all.csv")
#
#     # Convert data to torch.Tensor and adjust shape to fit LSTM
#     X_train = torch.tensor(X_train.values, dtype=torch.float32).view(-1, X_train.shape[1],
#                                                                      1)  # (batch_size, seq_len, input_dim)
#     X_test = torch.tensor(X_test.values, dtype=torch.float32).view(-1, X_test.shape[1], 1)
#     y_train = torch.tensor(y_train.values, dtype=torch.float32)
#     y_test = torch.tensor(y_test.values, dtype=torch.float32)
#
#     train_dataset = TensorDataset(X_train, y_train)
#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#     test_dataset = TensorDataset(X_test, y_test)
#     test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
#
#     pbounds = {
#         'lr': (1e-5, 1e-2),
#         'weight_decay': (0, 0.1),
#     }
#
#     optimizer = BayesianOptimization(
#         f=objective,
#         pbounds=pbounds,
#         verbose=2,
#         random_state=1,
#     )
#
#     optimizer.maximize(
#         init_points=2,
#         n_iter=10,
#     )
#
#     best_params = optimizer.max['params']
#     print("Best Parameters: ", best_params)

# 确保数据预处理和模型文件在当前目录下或者在PYTHONPATH中

def objective(lr, weight_decay):
    model = CombinedModel(input_dim=1, hidden_dim=64, output_dim=4, num_blocks=22, num_heads=4).to(device)  # 简化模型
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    num_epochs = 50  # 减少迭代次数
    best_validation_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        validation_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                validation_loss += loss.item()
        validation_loss /= len(test_loade    verbose=2,r)

        # 早停机制
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
        else:
            break

    return -best_validation_loss  # 返回负的验证损失以进行最大化


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X_train, X_test, y_train, y_test = load_and_preprocess_data("./data/多数据源位置预测_all.csv")

    # 将数据转换为torch.Tensor并调整形状以适合模型
    X_train = torch.tensor(X_train.values, dtype=torch.float32).to(device).view(-1, X_train.shape[1], 1)
    X_test = torch.tensor(X_test.values, dtype=torch.float32).to(device).view(-1, X_test.shape[1], 1)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    pbounds = {
        'lr': (1e-5, 1e-2),
        'weight_decay': (0, 0.1),
    }

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,

        random_state=1,
    )

    optimizer.maximize(
        init_points=2,
        n_iter=5,  # 减少迭代次数
    )

    best_params = optimizer.max['params']
    print("Best Parameters: ", best_params)

