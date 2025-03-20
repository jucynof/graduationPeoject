import numpy as np
import torch
from torch import softmax

def Evaluate1(acc,loss,w,w_attenaution):
    scores=np.zeros(len(acc))
    maxLoss=np.max(loss)
    maxAcc=np.max(acc)
    for i in range(len(acc)):
        scores[i]=(w*acc[i]/maxAcc)+((1-w)*(1-loss[i]/maxLoss))
        for i in range(len(acc)):
            scores[i]=scores[i]*w_attenaution[i]
    scores=softmax(torch.Tensor(scores),dim=0)
    return scores.cpu().numpy()


def knapsack_01(weights, values, capacity):
    n = len(weights)
    dp = [0] * (capacity + 1)
    # 使用二维数组记录是否选择物品（True/False）
    selected = [[False] * (capacity + 1) for _ in range(n)]
    for i in range(n):
        weight = weights[i]
        value = values[i]
        # 逆序遍历容量
        for w in range(capacity, weight - 1, -1):
            if dp[w - weight] + value > dp[w]:
                dp[w] = dp[w - weight] + value
                selected[i][w] = True  # 标记当前容量下选择了物品i
    # 逆向回溯找出被选中的物品
    selected_indices = []
    remaining_capacity = capacity
    # 从最后一个物品开始检查
    for i in range(n - 1, -1, -1):
        if selected[i][remaining_capacity]:
            selected_indices.append(i)
            remaining_capacity -= weights[i]
    selected_indices.reverse() # 因为是从后往前添加，需要反转结果
    return dp[capacity], selected_indices


