from bayes_opt import BayesianOptimization, UtilityFunction
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from data_preprocessing import load_and_preprocess_data_multi_output
from resnet_multi_output import CombinedModel


def objective(lr, weight_decay):
    model = CombinedModel(input_dim=1, hidden_dim=64, output_dim=4, num_blocks=22, num_heads=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # 训练模型
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    # 在验证集上评估模型
    model.eval()
    validation_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            validation_loss += loss.item()

    return -validation_loss / len(test_loader)


if __name__ == '__main__':
    # 检查是否有可用的 GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    # X_train_np, X_test_np, y_train_np, y_test_np = load_and_preprocess_data("./data/电机转动角度(弧度).csv")
    X_train_np, X_test_np, y_train_np, y_test_np = load_and_preprocess_data_multi_output("./data/多数据源位置预测_all.csv")
    # print(X_train_np.shape, X_test_np.shape, y_train_np.shape, y_test_np.shape)
    # Convert data to torch.Tensor and adjust shape to fit LSTM
    X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device).view(-1, X_train_np.shape[1],
                                                                            1)  # (batch_size, seq_len, input_dim)
    X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device).view(-1, X_test_np.shape[1], 1)
    # y_train = torch.tensor(y_train_np.values, dtype=torch.float32).to(device).view(-1, 1)
    # y_test = torch.tensor(y_test_np.values, dtype=torch.float32).to(device).view(-1, 1)
    y_train = torch.tensor(y_train_np.values, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test_np.values, dtype=torch.float32).to(device)
    # print(y_train.shape, y_test.shape)
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # 创建验证数据加载器
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 定义超参数空间
    pbounds = {
        'lr': (1e-5, 1e-2),
        'weight_decay': (0, 0.1),
    }

    # 初始化贝叶斯优化器
    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        verbose=2,  # 1: print only when a maximum is observed, 0: silent
        random_state=1,
    )

    # 设置高斯过程参数（如果需要的话）
    # optimizer.set_gp_params(normalize_y=True)  # 举个例子，你可以根据需要设置其他参数

    # 创建采集函数的实例
    # utility = UtilityFunction(kind="ei", kappa=2.5, xi=0.0)

    # 运行贝叶斯优化
    optimizer.maximize(
        init_points=2,
        n_iter=10,
        # acq_func="ucb",  # 使用一个内置的采集函数
        # acquisition_function=utility  # 传递采集函数实例
    )

    # 获取最优超参数
    best_params = optimizer.max['params']
    print("Best Parameters: ", best_params)
