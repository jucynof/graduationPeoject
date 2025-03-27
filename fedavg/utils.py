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
    timeStart=np.random.uniform(1,10)
    times = [np.random.uniform(timeStart ,timeStart*config['timesMax']) for _ in range(config['num_clients'])]
    timeMax=np.max(times)
    for i in range(len(times)):
        times[i] = times[i]/timeMax
    return times
def getFinalTime(config,times,curr_round):
    np.random.seed(config['random_seed']+10+curr_round)
    timeStart=np.random.uniform(1,10)
    times_ = [np.random.uniform(timeStart, timeStart * config['timesMax']) for _ in range(config['num_clients'])]
    timeMax = np.max(times_)
    for i in range(len(times)):
        times_[i] = times_[i] / timeMax
        times[i]=(times[i]+times_[i])/2
    return times
#获取每个客户端的消费和总的消费预算
def getcost(config):
    np.random.seed(config['random_seed']+20)
    costStart=np.random.randint(10,20)
    costs=[np.random.randint(costStart, int(costStart*config['costsMax']))for _ in range(config['num_clients'])]
    costThreshold=int((np.average(costs)))*int((config['num_clients']*config['costThreshold']))
    return costs,costThreshold

def getClientsForTrain(scores,times,costs,costThreshold,config):
    times=np.array(times)
    indices = np.argsort(-times)[:config["num_clients"]]  # 降序排序后取前config["num_clients"]个索引
    scoresFinal=[]
    idChoosed=[]
    for i in range(len(indices)):
        costThresholdCur=costThreshold

        #保存该客户端的得分
        scoreCur=scores[indices[i]]
        #把该客户端的消费减去
        costCur=costs[indices[i]]
        costThresholdCur=costThresholdCur-costCur
        # 保证该轮一定选中这个客户端，且比这个客户端耗时更多的客户端一定不选
        costs[indices[i]]=costThresholdCur+1
        #价值和时间的统筹估算
        valuesum,idChoosedCur=knapsack_01(costs[:],scores[:],costThresholdCur)
        idChoosedCur.append(i)
        # for k in range(len(idChoosedCur)):
        #     print(costs[idChoosedCur[k]]/np.average(costs))
        timeSum = times[indices[i]]*len(idChoosedCur)
        scoresFinalCur=(config['a']*(valuesum+scoreCur))-(config['b']*timeSum)
        scoresFinal.append(scoresFinalCur)
        idChoosed.append(idChoosedCur)
    scoresFinal=np.array(scoresFinal)
    i = np.argsort(-scoresFinal)[:config["num_clients"]] # 输出得分最大的是哪一轮
    # for j in range(10):
    #     print(i[j*10], scoresFinal[i[j*10]])
    return idChoosed[i[0]]

























