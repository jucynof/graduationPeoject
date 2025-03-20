import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
data1 = pd.read_csv('./Zero.csv', header=None).squeeze()
data2 = pd.read_csv('C:/Users/JucyNof/Desktop/fedavg1/One.csv', header=None).squeeze()
data3 = pd.read_csv('C:/Users/JucyNof/Desktop/fedavg2/Two.csv', header=None).squeeze()

# 统一横坐标
max_length = max(len(data1), len(data2), len(data3))
x = range(max_length)
data1 = data1.reindex(x, fill_value=None)
data2 = data2.reindex(x, fill_value=None)
data3 = data3.reindex(x, fill_value=None)

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(x, data1, label='test 1', color='#1f77b4', linestyle='-', linewidth=1.5)
plt.plot(x, data2, label='test 2', color='#ff7f0e', linestyle='--', linewidth=1.5)
plt.plot(x, data3, label='test 3', color='#2ca02c', linestyle=':', linewidth=1.5)

plt.xlabel('Index', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.title('Three Datasets Comparison', fontsize=14)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.savefig('result.png', dpi=300)
plt.show()