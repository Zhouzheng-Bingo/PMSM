import os
import pandas as pd
from scipy.io import loadmat


# def mat_to_csv(directory_path, var_name):
#     all_data = []
#
#     # 获取所有的.mat文件
#     mat_files = [f for f in os.listdir(directory_path) if f.endswith('.mat')]
#
#     for mat_file in mat_files:
#         file_path = os.path.join(directory_path, mat_file)
#         mat_data = loadmat(file_path)
#
#         if var_name in mat_data:
#             ts_data = mat_data[var_name]
#             ts_data = ts_data[0][0]  # 从结构中提取数据
#             time = ts_data['Time'].ravel()
#             data = ts_data['Data'].ravel()
#             all_data.append(pd.DataFrame({'Time': time, var_name: data}))
#
#     # 将所有数据合并到一个数据帧
#     merged_data = pd.concat(all_data, ignore_index=True)
#
#     # 输出到CSV
#     csv_output_path = os.path.join(directory_path, f"{var_name}.csv")
#     merged_data.to_csv(csv_output_path, index=False)
#
#
# # 设置文件夹路径
# directory_path = './data/电机参数/第一轮'  # 请确认路径正确
#
# # 获取第一个.mat文件来获取所有的变量名
# first_mat_file = [f for f in os.listdir(directory_path) if f.endswith('.mat')][0]
# mat_data = loadmat(os.path.join(directory_path, first_mat_file))
#
# variable_names = [var for var in mat_data.keys() if not var.startswith('_')]
#
# for var_name in variable_names:
#     mat_to_csv(directory_path, var_name)
#
# print("所有.mat文件的所有变量已成功转换为.csv文件")

import os
from scipy.io import loadmat

def print_mat_structure(directory_path, mat_file_name):
    file_path = os.path.join(directory_path, mat_file_name)
    mat_data = loadmat(file_path)
    for key, value in mat_data.items():
        if not key.startswith('_'):
            print(f"Variable name: {key}")
            print(f"Variable type: {type(value)}")
            print(f"Variable shape: {value.shape}")
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    print(f"  - {subkey}: {type(subvalue)}, shape: {subvalue.shape}")


directory_path = './data/电机参数/第一轮'  # 确认路径正确
mat_file_name = [f for f in os.listdir(directory_path) if f.endswith('.mat')][0]
print_mat_structure(directory_path, mat_file_name)
