import csv
import numpy as np
import torch
from torch import softmax


#只考虑准确率和损失进行打分
def Evaluate1(acc,loss,w,w_attenaution):
    accmin=np.min(acc)
    accmax=np.max(acc)
    for i in range(len(acc)):
        acc[i]-=accmin
        acc[i]+=(accmax/3)
    scores=np.zeros(len(acc))
    maxLoss=np.max(loss)
    maxAcc=np.max(acc)
    for i in range(len(acc)):
        scores[i]=(w*acc[i]/maxAcc)+((1-w)*(1-loss[i]/maxLoss))
    for i in range(len(acc)):
        scores[i] = scores[i] * w_attenaution[i]
    # print(scores)
    for i in range(len(scores)):
        scores[i]=scores[i]**4
   # scores=softmax(torch.Tensor(scores),dim=0)
    # scores=scores.cpu().numpy()
   # scoreMax=np.max(scores)
   # for i in range(len(scores)):
    #    scores[i]=scores[i]/scoreMax
    # print(scores)
    return scores
#01背包
def knapsack_01(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    # 填充动态规划表
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]

    # 回溯找到所选物品的编号
    selected_items = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected_items.append(i-1)
            w -= weights[i - 1]

    selected_items.reverse()  # 因为是从后往前添加的，需反转
    return dp[n][capacity], selected_items

# def knapsack_01(weights, values, capacity):
#     n = len(weights)
#     dp = [0] * (capacity + 1)
#     # 使用二维数组记录是否选择物品（True/False）
#     selected = [[False] * (capacity + 1) for _ in range(n)]
#     for i in range(n):
#         weight = weights[i]
#         value = values[i]
#         # 逆序遍历容量
#         for w in range(capacity, weight - 1, -1):
#             if dp[w - weight] + value > dp[w]:
#                 dp[w] = dp[w - weight] + value
#                 selected[i][w] = True  # 标记当前容量下选择了物品i
#     # 逆向回溯找出被选中的物品
#     selected_indices = []
#     remaining_capacity = capacity
#     # 从最后一个物品开始检查
#     for i in range(n - 1, -1, -1):
#         if selected[i][remaining_capacity]:
#             selected_indices.append(i)
#             remaining_capacity -= weights[i]
#     selected_indices.reverse() # 因为是从后往前添加，需要反转结果
#     return dp[capacity], selected_indices
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
        times[i]=times[i]*0.7+times_[i]*0.3
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
    timesFinal=[]
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
        scoresSum=valuesum+scoreCur
        timesFinal.append(timeSum)
        scoresFinal.append(scoresSum)
        # print(scoressum,timeSum)
        # scoresFinalCur=(config['a']*scoressum)-(config['b']*timeSum)
        # scoresFinal.append(scoresFinalCur)
        idChoosed.append(idChoosedCur)
    scoresFinalMax=np.max(scoresFinal)
    timesMax=np.max(timesFinal)
    for i in range(len(scoresFinal)):
        scoresFinal[i]=scoresFinal[i]/scoresFinalMax
        timesFinal[i]=timesFinal[i]/timesMax
        scoresFinal[i]=(config['a']*scoresFinal[i])-(config['b']*timesFinal[i])
    temp_costs=np.argsort(costs)[:config["num_clients"]]
    temp_scores = np.argsort(-scores)[:config["num_clients"]]
    temp_times = np.argsort(times)[:config["num_clients"]]
    scoresFinal=np.array(scoresFinal)
    i = np.argsort(-scoresFinal)[:config["num_clients"]] # 输出得分最大的是哪一轮
    # print(temp_scores[0:30])
    # print(temp_times[0:30])
    # print(idChoosed[i[0]])
    return idChoosed[i[0]]

def connectLastTrain(config):
    print("衔接上次继续训练")
    global_parameters = torch.load(config["glocalParameterPath"])
    accuracyGlobal = []
    lossGlobal = []
    timeGlobal = []
    costGlobal = []
    with open('./test_accuracy.csv', 'r') as file:  # 读取acc表方便继续更新
        csv_reader = csv.reader(file)
        for row in csv_reader:
            try:
                # 转换为浮点数并保留精度
                accuracyGlobal.append(float(row[0].strip()))
            except (ValueError, IndexError) as e:
                print(f"第{len(accuracyGlobal) + 1}行转换失败：{e}")
                accuracyGlobal.append(None)  # 保留空值标记异常数据
    lastRound = config["lastRound"]
    with open("./test_loss.csv", 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            try:
                # 转换为浮点数并保留精度
                lossGlobal.append(float(row[0].strip()))
            except (ValueError, IndexError) as e:
                print(f"第{len(lossGlobal) + 1}行转换失败：{e}")
                lossGlobal.append(None)  # 保留空值标记异常数据
    with open("./test_time.csv", 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            try:
                # 转换为浮点数并保留精度
                timeGlobal.append(float(row[0].strip()))
            except (ValueError, IndexError) as e:
                print(f"第{len(timeGlobal) + 1}行转换失败：{e}")
                timeGlobal.append(None)  # 保留空值标记异常数据
    with open("./test_cost.csv", 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            try:
                # 转换为浮点数并保留精度
                costGlobal.append(int(row[0].strip()))
            except (ValueError, IndexError) as e:
                print(f"第{len(costGlobal) + 1}行转换失败：{e}")
                costGlobal.append(None)  # 保留空值标记异常数据
    return  global_parameters,accuracyGlobal,lossGlobal,timeGlobal,costGlobal,lastRound





















