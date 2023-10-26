import pandas as pd

# 读取CSV文件，假设文件名为example.csv，编码为GBK，以处理包含汉字的列头
df = pd.read_csv('data/电流iq.csv', encoding='GBK')

# 获取C列的数据
c_column = df['实际']

# 找到最大值和最小值
max_value = c_column.max()
min_value = c_column.min()

# 打印结果
print(f'C列的最大值为: {max_value}')
print(f'C列的最小值为: {min_value}')
