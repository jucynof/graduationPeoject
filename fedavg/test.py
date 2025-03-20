# import numpy as np
# import torch
# from torch import softmax
# import matplotlib.pyplot as plt
# # a=np.array([1,2,3,4,6],dtype=np.float32)
# # y=softmax(torch.tensor(a),dim=0)
# # y=y.numpy()
# # x=np.arange(len(y))+1
# # plt.plot(x, y)  # 默认生成蓝色实线
# # plt.show()
# # print(y)
# def knapsack_01(weights, values, capacity):
#     n = len(weights)
#     dp = [0] * (capacity + 1)
#
#     for i in range(n):
#         weight = weights[i]
#         value = values[i]
#         # 逆序遍历，防止重复选择
#         for w in range(capacity, weight - 1, -1):
#             dp[w] = max(dp[w], dp[w - weight] + value)
#
#     return dp[capacity]
# if __name__== "__main__":
#     a=np.array([1,2,3,4,5])
#     b=np.array([2,3,7,10,20])
#     c=knapsack_01(a, b, 10)
#     print(c)