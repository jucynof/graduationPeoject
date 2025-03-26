import numpy as np
import torch
from torch import softmax

#只考虑准确率和损失进行打分
def Evaluate1(acc,loss,w,w_attenaution):
    scores=np.zeros(len(acc))
    maxLoss=np.max(loss)
    maxAcc=np.max(acc)
    for i in range(len(acc)):
        scores[i]=(w*acc[i]/maxAcc)+((1-w)*(1-loss[i]/maxLoss))
    for i in range(len(acc)):
        scores[i] = scores[i] * w_attenaution[i]
    # scores=softmax(torch.Tensor(scores),dim=0)
    # scores=scores.cpu().numpy()
    scoreMax=np.max(scores)
    for i in range(len(scores)):
        scores[i]=scores[i]/scoreMax
    return scores
#01背包
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
#获取每个客户端的耗时
def getTime(config):
    np.random.seed(config['random_seed']+10)
    times = [np.random.uniform(10, 100) for _ in range(config['num_clients'])]
    timeMax=np.max(times)
    for i in range(len(times)):
        times[i] = times[i]/timeMax
    return times
#获取每个客户端的消费和总的消费预算
def getcost(config):
    np.random.seed(config['random_seed']+20)
    costs=[np.random.randint(10, 100) for _ in range(config['num_clients'])]
    costThreshold=int((np.average(costs)))*int((config['num_clients']*config['client_rate']))
    return costs,costThreshold

def getClientsForTrain(scores,times,costs,costThreshold,config):
    times=np.array(times)
    indices = np.argsort(-times)[:config["num_clients"]]  # 降序排序后取前config["num_clients"]个索引
    scoresFinal=[]
    idChoosed=[]
    for i in range(len(indices)):
        costsCur=costs
        costThresholdCur=costThreshold
    #保证该轮一定选中这个客户端，且比这个客户端耗时更多的客户端一定不选
        #保存该客户端的得分
        scoreCur=scores[indices[i]]
        #把该客户端的消费减去
        costCur=costsCur[indices[i]]
        costThresholdCur=costThresholdCur-costCur
        #把耗时高于该客户端的消费全部暂时拔高以保证不会被选择
        for j in range(0,i+1):
            costsCur[indices[j]]=costThresholdCur+1
        #价值和时间的统筹估算

        valuesum,idChoosedCur=knapsack_01(costsCur,scores,costThresholdCur)
        timeSum = times[indices[i]]*len(idChoosedCur)
        scoresFinalCur=valuesum+scoreCur-timeSum
        scoresFinal.append(scoresFinalCur)
        idChoosed.append(idChoosedCur)
    scoresFinal=np.array(scoresFinal)
    i = (np.argsort(-scoresFinal)[:1])[0]  # 输出得分最大的是哪一轮
    idChoosed.append(indices[i])
    return idChoosed[i]

























