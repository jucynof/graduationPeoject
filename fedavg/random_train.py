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
    opti = optim.Adam(net.parameters(), lr=lr)
    # 定义变量global_parameters
    global_parameters = net.state_dict()
    # --# --------------------------------通信----------------------------------
    #储存每次通信accuracy
    accuracyGlobal=[]
    lossGlobal=[]
    lastRound=0#上次关闭程序时通信轮次
    if os.path.exists(config["glocalParameterPath"]):#存在说明可以直接继续训练
        print("衔接上次继续训练")
        global_parameters = torch.load(config["glocalParameterPath"])
        with open('./test_accuracy.csv', 'r') as file:#读取acc表方便继续更新
            csv_reader = csv.reader(file)
            for row in csv_reader:
                try:
                    # 转换为浮点数并保留精度
                    accuracyGlobal.append(float(row[0].strip()))
                except (ValueError, IndexError) as e:
                    print(f"第{len(accuracyGlobal) + 1}行转换失败：{e}")
                    accuracyGlobal.append(None)  # 保留空值标记异常数据
        lastRound=config["lastRound"]
        with open("./test_loss.csv",'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                try:
                    # 转换为浮点数并保留精度
                    lossGlobal.append(float(row[0].strip()))
                except (ValueError, IndexError) as e:
                    print(f"第{len(lossGlobal) + 1}行转换失败：{e}")
                    lossGlobal.append(None)  # 保留空值标记异常数据
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
    for curr_round in range(lastRound+1, lastRound+rounds + 1):
        local_loss = []
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


            # # ###########################
            # train_loader = DataLoader(subTrainDateset, batch_size=64, shuffle=True)
            # for epoch in range(100):
            #     net.train()
            #     running_loss = 0.0
            #     sumcorrect = 0
            #     total = 0
            #     for images, labels in train_loader:
            #         # 将数据移动到GPU
            #         images, labels = images.to(dev), labels.to(dev)
            #         # 前向传播
            #         outputs,_ = net(images)
            #         loss = loss_func(outputs, labels)
            #         print("loss:", loss.item())
            #         # 反向传播和优化
            #         opti.zero_grad()
            #         loss.backward()
            #         opti.step()
            #         running_loss += loss.item()
            #         preds = torch.argmax(outputs, dim=1)
            #         correct = (preds == labels).sum().item()
            #         sumcorrect += correct
            #         total += labels.size(0)
            #     print(
            #         f'Epoch [{epoch + 1}/{100}], Loss: {running_loss / total:.4f}, Accuracy: {sumcorrect / total:.4f}')
            #
            #
            # ############################
            # 每个client训练得到的权重
            local_parameters = cur_client.localUpdate(localEpoch=config["localEpoch"], localBatchSize=config["batchSize"], Net=net,
                                                     lossFun=loss_func,
                                                     opti=opti,
                                                     global_parameters=global_parameters,
                                                     trainDataSet=subTrainDateset, dev=dev)
            client_params[k] = local_parameters
        # 取平均值，得到本次通信中server得到的更新后的模型参数
        s = server(client_params,config=config)
        global_parameters = s.agg_average()
        net.load_state_dict(global_parameters, strict=True)
        accuracy = test_accuracy()
        global_loss, global_acc = accuracy.test_accuracy(net, global_parameters, data.getTestData(), dev, loss_func,config)
        print(
            '----------------------------------[Round: %d] accuracy: %f  loss: %f----------------------------------'
             % (curr_round, global_acc, global_loss))
        accuracyGlobal.append(global_acc)
        lossGlobal.append(global_loss)
        #定期将参数和acc保存
        x=x+1
        if x % 10 == 0:
            torch.save(net.state_dict(), config["glocalParameterPath"])
            listaccuracy = np.array(accuracyGlobal)
            listloss = np.array(lossGlobal)
            np.savetxt('./test_accuracy.csv', listaccuracy.reshape(-1, 1), delimiter=',', fmt='%.4f')
            np.savetxt('./test_loss.csv', listloss.reshape(-1, 1), delimiter=',', fmt='%.6f')
            config["lastRound"] = curr_round
            with open('config.json', 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            f.close()
