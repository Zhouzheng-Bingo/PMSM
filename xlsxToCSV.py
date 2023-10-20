import pandas as pd

# 读取xlsx文件，转成csv文件
# # 读取xlsx文件
# xlsx_file = 'data/电机转速(转每秒).xlsx'
# df = pd.read_excel(xlsx_file)
#
# # 将数据保存为csv文件
# csv_file = 'data/电机转速(转每秒).csv'
# df.to_csv(csv_file, index=False)
#
# # 读取xlsx文件
# xlsx_file = 'data/电流id.xlsx'
# df = pd.read_excel(xlsx_file)
#
# # 将数据保存为csv文件
# csv_file = 'data/电流id.csv'
# df.to_csv(csv_file, index=False)
#
# # 读取xlsx文件
# xlsx_file = 'data/电流iq.xlsx'
# df = pd.read_excel(xlsx_file)

# # 将数据保存为csv文件
# csv_file = 'data/电流iq.csv'
# df.to_csv(csv_file, index=False)

# 读取xlsx文件
# xlsx_file = 'data/电机转动角度(弧度).xlsx'
# df = pd.read_excel(xlsx_file)
#
# # 将数据保存为csv文件
# csv_file = 'data/电机转动角度(弧度).csv'
# df.to_csv(csv_file, index=False)


# 读取csv，截取前100000行数据，保存为新的csv文件
"""
import pandas as pd

# 处理电流iq.csv
csv_file_iq = 'data/电流iq.csv'
# 指定'ISO-8859-1'编码方式读取CSV文件
df_iq = pd.read_csv(csv_file_iq, encoding='ISO-8859-1')
df_iq_subset = df_iq.head(100000)
csv_subset_file_iq = 'data/电流iq_subset.csv'
# 指定'UTF-8'编码方式保存CSV文件
df_iq_subset.to_csv(csv_subset_file_iq, index=False, encoding='utf-8')

# 处理电流id.csv
csv_file_id = 'data/电流id.csv'
# 指定'ISO-8859-1'编码方式读取CSV文件
df_id = pd.read_csv(csv_file_id, encoding='ISO-8859-1')
df_id_subset = df_id.head(100000)
csv_subset_file_id = 'data/电流id_subset.csv'
# 指定'UTF-8'编码方式保存CSV文件
df_id_subset.to_csv(csv_subset_file_id, index=False, encoding='utf-8')

# 处理电机转速(转每秒).csv
csv_file_speed = 'data/电机转速(转每秒).csv'
# 指定'ISO-8859-1'编码方式读取CSV文件
df_speed = pd.read_csv(csv_file_speed, encoding='ISO-8859-1')
df_speed_subset = df_speed.head(100000)
csv_subset_file_speed = 'data/电机转速(转每秒)_subset.csv'
# 指定'UTF-8'编码方式保存CSV文件
df_speed_subset.to_csv(csv_subset_file_speed, index=False, encoding='utf-8')
"""
# 处理电机转动角度(弧度).csv
csv_file_angle = 'data/电机转动角度(弧度).csv'
# 指定'ISO-8859-1'编码方式读取CSV文件
df_angle = pd.read_csv(csv_file_angle, encoding='ISO-8859-1')
df_angle_subset = df_angle.head(100000)
csv_subset_file_angle = 'data/电机转动角度(弧度)_subset.csv'
# 指定'UTF-8'编码方式保存CSV文件
df_angle_subset.to_csv(csv_subset_file_angle, index=False, encoding='utf-8')


# 电机转动角度(弧度).xlsx这个文件给的格式不好，需要修改
# import pandas as pd
#
# # 读取电机转动角度(弧度).xlsx文件
# xlsx_file = './data/电机转动角度(弧度).xlsx'
# df = pd.read_excel(xlsx_file)
#
# # 根据“时间”列的值设置“指令位置”列的值
# def set_command_position(time):
#     if 0.0 <= time < 0.2:
#         return 3
#     elif 0.2 <= time < 0.4:
#         return 1
#     elif 0.4 <= time < 0.6:
#         return 4
#     elif 0.6 <= time < 0.8:
#         return 2
#     elif 0.8 <= time <= 1.0:
#         return 1
#     else:
#         return None
#
# df['指令位置'] = df['时间'].apply(set_command_position)
#
# # 保存修改后的数据到一个新的Excel文件
# output_xlsx_file = '修改后的电机转动角度(弧度).xlsx'
# df.to_excel(output_xlsx_file, index=False)
#
# print(f"修改后的数据已保存到 {output_xlsx_file}")

print('Done!')

