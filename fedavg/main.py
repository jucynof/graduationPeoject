import csv
import os
import numpy as np
import torch
from torch import nn, optim
import json
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split
from models import SimpleCNN,EnhancedCNN,CNNforFashionMinst
from fed import client,server
from getData import getData, getNoIIDData
from utils import Evaluate1, getTime, getClientsForTrain, getcost, getFinalTime,connectLastTrain
from testaccuracy import test_accuracy
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
    net=None
    if config['dataName']=="fashionMinst":
        # net = SimpleCNN(config)
        print("FashionMinst")
        net =CNNforFashionMinst(config,in_channels=1).to(dev)
    elif config['dataName']=="cifar10":
        print("CIFAR10")
        net = EnhancedCNN(config,in_channels=3).to(dev)
    # 定义损失函数
    loss_func = nn.CrossEntropyLoss().to(dev)
    # 定义优化器
    if config["isIID"]==1:
        lr=config["lrIID"]
    else:
        lr=config["lrNoIID"]
    opti = optim.Adam(net.parameters(), lr=lr)

    # 定义变量global_parameters
    global_parameters = net.state_dict()
    # --# --------------------------------通信----------------------------------
    #储存每次通信accuracy
    accuracyGlobal=[]
    lossGlobal=[]
    timeGlobal = []
    costGlobal = []
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
    w_attenaution=np.array([1 for i in range(config["num_clients"])],dtype=np.float32)#定义每个客户端的打分的衰减率
    x=0#定期保存训练的参数和acc
    timesFixed = getTime(config)#每个客户端的固定时间开销
    costs, costthreshold = getcost(config)
    for curr_round in range(lastRound+1, lastRound+rounds + 1):
        timesFinal=getFinalTime(config,timesFixed[:],curr_round)#添加每个客户每轮通信的随机开销
        #timesFinal=timesFixed
        scores=None
        cur_time=[]
        cur_cost=0
        local_loss = []
        client_params = {}
        acc = np.zeros(config["num_clients"])#用于打分保存的acc
        loss = np.zeros(config["num_clients"])#用于打分保存的loss
        #遍历所有模型，给所有客户端打分
        local_parameters = {}
        for k in range(config["num_clients"]):
            cur_client = client(config)
            subTrainDateset = Subset(data.getTrainData(), data.getDataIndices()[k])
            # 每个client训练得到的权重
            local_parameters[k] = cur_client.localUpdate(localEpoch=3,
                                                      localBatchSize=config["batchSize"],
                                                        Net=net,
                                                      lossFun=loss_func,
                                                      opti=opti,
                                                      global_parameters=global_parameters,
                                                      #global_parameters=local_parameters[k],
                                                      trainDataSet=subTrainDateset, dev=dev)
            accuracy = test_accuracy()
            testDataset=data.getTestData()
            # 计算拆分比例（10%作为子集）
            subset_size = int(len(testDataset) * 0.1)
            remainder_size = len(testDataset) - subset_size
            # 随机分割数据集
            test_subset, _ = random_split(
                testDataset,
                [subset_size, remainder_size],
                generator=torch.Generator().manual_seed(config["random_seed"]+curr_round)  # 固定随机种子
            )
            loss[k], acc[k] = accuracy.test_accuracy(net, local_parameters[k], test_subset, dev, loss_func,config)
        # print(acc)
        scores = Evaluate1(acc[:], loss[:], config["w"], w_attenaution[:])
        indices = getClientsForTrain(scores[:], timesFinal[:], costs[:], costthreshold, config)
        print("第%d轮次通信中选中的客户端为:"%(curr_round),indices)
        # print("他们的得分如下：")
        # for ind in indices:
        #     print(scores[ind])
            #重新计算衰减率
        for i in range(config["num_clients"]):
            if i in indices:
                w_attenaution[i]=w_attenaution[i]*config["attenuationRate"]
            else:
                w_attenaution[i]=1
        idChoosed=indices#根据得分抽取
        # 将选择出的客户端训练参数用于更新全局参数
        # j=0
        # for i in idChoosed:
        #     cur_time.append(timesFinal[i])
        #     cur_cost += costs[i]
        #     client_params[j]=local_parameters[i]
        #     j=j+1
        # for k in range(int(config["num_clients"]*config["client_rate"])):
        j = 0
        for i in idChoosed:
            cur_time.append(timesFinal[i])
            cur_cost += costs[i]
            cur_client = client(config)
            subTrainDateset = Subset(data.getTrainData(), data.getDataIndices()[i])
            # 每个client训练得到的权重
            local_parameters = cur_client.localUpdate(localEpoch=config["localEpoch"],
                                                      localBatchSize=config["batchSize"],
                                                      Net=net,
                                                     lossFun=loss_func,
                                                     opti=opti,
                                                     global_parameters=global_parameters,
                                                     trainDataSet=subTrainDateset, dev=dev)
            client_params[j] = local_parameters
            j+=1
        # 取平均值，得到本次通信中server得到的更新后的模型参数
        s = server(client_params, config=config)
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
        #更新学习率
        if curr_round==100:
            for param_group in opti.param_groups:
                param_group['lr'] = 0.001
        elif curr_round==200:
            for param_group in opti.param_groups:
                param_group['lr'] = 0.0001
        elif curr_round==300:
            for param_group in opti.param_groups:
                param_group['lr'] = 0.00001

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
