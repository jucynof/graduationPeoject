import csv
import os
import numpy as np
import torch
from torch import nn, optim
import json
from torch.utils.data import DataLoader, Subset,TensorDataset
from models import SimpleCNN,EnhancedCNN,CNNforFashionMinst
from fed import client,server
from getData import getData, getNoIIDData
from testaccuracy import test_accuracy
from utils import *
with open("config.json") as f:
    config = json.load(f)
f.close()
if __name__ == "__main__":
    torch.manual_seed(config['random_seed'])  # 设置PyTorch种子
    np.random.seed(config['random_seed'])
    #获取数据
    if config["isIID"]==1:
        print("IID")
        data = getData(config)
    else:
        print("Not IID")
        data = getNoIIDData(config)
    # ----------------------------------设置参数----------------------------------
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(dev)
    # config["dev"]=dev
    rounds=config["rounds"]
    numClients=config["num_clients"]
    numbers = np.arange(numClients)  # 创建一个包含numClients个数字的数组
    clientRate=config["client_rate"]
    # numClientsChoosed=int(numClients*clientRate)
    # ----------------------------------初始化模型----------------------------------
    #模型实例化
    if config['dataName']=="fashionMinst":
        # net = SimpleCNN(config)
        print("FashionMinst")
        net =CNNforFashionMinst(config,in_channels=1)
    elif config['dataName']=="cifar10":
        print("CIFAR10")
        net = EnhancedCNN(config,in_channels=3)
    net = net.to(dev)
    # 定义损失函数
    loss_func = nn.CrossEntropyLoss()
    loss_func = loss_func.to(dev)
    # 定义优化器
    if config["isIID"]==1:
        lr=config["lrIID"]
    else:
        lr=config["lrNoIID"]
    opti = optim.Adam(net.parameters(),lr=lr)
    # 定义变量global_parameters
    global_parameters = net.state_dict()
    # --# --------------------------------通信----------------------------------
    #储存每次通信accuracy
    accuracyGlobal=[]
    lossGlobal=[]
    timeGlobal=[]
    costGlobal=[]
    lastRound=0#上次关闭程序时通信轮次
    if os.path.exists(config["glocalParameterPath"]):#存在说明可以直接继续训练
        global_parameters,accuracyGlobal,lossGlobal,timeGlobal,costGlobal,lastRound=connectLastTrain(config)
        opti.load_state_dict(torch.load(config['optimizerPath']))
    else:
        # 通信前的accuracy
        print("重新训练")
        #未进行训练是的acc和loss
        accuracy = test_accuracy()
        global_loss, global_acc = accuracy.test_accuracy(net, global_parameters, data.getTestData(), dev, loss_func,config)
        print(
            '----------------------------------[Round: %d] accuracy: %f  loss: %f----------------------------------'
            % (0, global_acc, global_loss))
        accuracyGlobal.append(global_acc)
        lossGlobal.append(global_loss)
    # clients与server之间通信
    x=0#定期保存训练的参数和acc
    timesFixed = getTime(config)#每个客户端的固定时间开销
    costs, costthreshold = getcost(config)

    for curr_round in range(lastRound+1, lastRound+rounds + 1):
        cur_time=[]
        cur_cost=0
        timesFinal = getFinalTime(config, timesFixed[:], curr_round)  # 添加每个客户每轮通信的随机开销
        local_loss = []
        #timesFinal=timesFixed
        client_params = {}
        #遍历所有模型，给所有客户端打分
        local_parameters={}
        indices = np.random.choice(numbers, int(config["num_clients"] * config["client_rate"]),
                                       replace=True)  # 随机抽取
        print("第%d轮次通信中选中的客户端为:"%(curr_round),indices)
        idChoosed=indices#根据得分抽取
        for k in range(len(idChoosed)):
            cur_client = client(config)
            subTrainDateset = Subset(data.getTrainData(), data.getDataIndices()[idChoosed[k]])
            # 每个client训练得到的权重
            local_parameters = cur_client.localUpdate(localEpoch=config["localEpoch"], localBatchSize=config["batchSize"], Net=net,
                                                     lossFun=loss_func,
                                                     opti=opti,
                                                     global_parameters=global_parameters,
                                                     trainDataSet=subTrainDateset, dev=dev)
            client_params[k] = local_parameters
            cur_time.append(timesFinal[idChoosed[k]])
            cur_cost+=costs[idChoosed[k]]
        # 取平均值，得到本次通信中server得到的更新后的模型参数
        if curr_round==100:
            for param_group in opti.param_groups:
                param_group['lr'] = 0.001
        elif curr_round==200:
            for param_group in opti.param_groups:
                param_group['lr'] = 0.0001
        elif curr_round==300:
            for param_group in opti.param_groups:
                param_group['lr'] = 0.00001
        s = server(client_params,config=config)
        global_parameters = s.agg_average()
        net.load_state_dict(global_parameters, strict=True)
        accuracy = test_accuracy()
        global_loss, global_acc = accuracy.test_accuracy(net, global_parameters, data.getTestData(), dev, loss_func,config)
        print(
            '----------------------------------[Round: %d] accuracy: %f  loss: %f time :%.16f cost:%d----------------------------------'
             % (curr_round, global_acc, global_loss, np.max(cur_time), cur_cost))
        accuracyGlobal.append(global_acc)
        lossGlobal.append(global_loss)
        timeGlobal.append(np.max(cur_time))
        costGlobal.append(cur_cost)
        #定期将参数和acc保存
        x=x+1
        if x % 10 == 0:
            torch.save(net.state_dict(), config["glocalParameterPath"])
            torch.save(opti.state_dict(), config["optimizerPath"])
            listaccuracy = np.array(accuracyGlobal)
            listloss = np.array(lossGlobal)
            listtime = np.array(timeGlobal)
            listcost = np.array(costGlobal)
            np.savetxt('./test_accuracy.csv', listaccuracy.reshape(-1, 1), delimiter=',', fmt='%.4f')
            np.savetxt('./test_loss.csv', listloss.reshape(-1, 1), delimiter=',', fmt='%.6f')
            np.savetxt('./test_time.csv', listtime.reshape(-1, 1), delimiter=',', fmt='%.16f')
            np.savetxt('./test_cost.csv', listcost.reshape(-1, 1), delimiter=',', fmt='%d')
            config["lastRound"] = curr_round
            with open('config.json', 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            f.close()
