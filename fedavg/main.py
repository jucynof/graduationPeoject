import csv
import os
import numpy as np
import torch
from torch import nn, optim
import json
from torch.utils.data import DataLoader, Subset,TensorDataset
from models import SimpleCNN,EnhancedCNN
from fed import client,server
from getData import getData, getNoIIDData
from utils import Evaluate1
with open("config.json") as f:
    config = json.load(f)
f.close()
# ----------------------------------在测试集上评估模型的性能, 计算准确率和平均损失----------------------------------
class test_accuracy:

    def test_accuracy(self, net, parameters, testDataset, dev, lossFun):
        # 存储损失

        testDataLoader=DataLoader(testDataset, batch_size=config["batchSize"], shuffle=False)
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        net.load_state_dict(parameters, strict=True)
        net.eval()  # 确保模型处于评估模式
        for data, label in testDataLoader:
            data, label = data.to(dev), label.to(dev)
            output = net(data)
            # 计算损失
            loss = lossFun(output, label)
            batch_loss = loss.item()
            batch_size = label.size(0)
            total_loss += batch_loss * batch_size  # 累加加权损失
            # 计算正确数
            preds = torch.argmax(output, dim=1)
            correct = (preds == label).sum().item()
            total_correct += correct
            total_samples += batch_size
        correctRate = total_correct / total_samples
        lossAvg = total_loss / total_samples
        return lossAvg,correctRate


        # loss_collector = []
        # with torch.no_grad():
        #     net.load_state_dict(parameters, strict=True)
        #     sum_accu = 0
        #     num = 0
        #     loss_collector.clear()
        #     # 载入测试集
        #     for data, label in testDataLoader:
        #         data, label = data.to(dev), label.to(dev)
        #         output = net(data)
        #         print(output)
        #         loss = lossFun(output, label)
        #         # loss = 1
        #         loss_collector.append(loss.item())
        #         output = torch.argmax(output, dim=1)
        #         sum_accu += (output == label).float().mean()
        #         print(sum_accu)
        #         num += 1
        #
        #     accuracy = sum_accu / num
        #     avg_loss = sum(loss_collector) / len(loss_collector)
        # return avg_loss, accuracy

if __name__ == "__main__":
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
    numClientsChoosed=int(numClients*clientRate)
    # ----------------------------------初始化模型----------------------------------
    #模型实例化
    net = SimpleCNN()
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
    # ----------------------------------通信----------------------------------
    #储存每次通信accuracy
    accuracyGlobal=[]
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
    else:
        # 通信前的accuracy
        print("重新训练")
        #未进行训练是的acc和loss
        accuracy = test_accuracy()
        global_loss, global_acc = accuracy.test_accuracy(net, global_parameters, data.getTestData(), dev, loss_func)
        print(
            '----------------------------------[Round: %d] accuracy: %f  loss: %f----------------------------------'
            % (0, global_acc, global_loss))
        accuracyGlobal.append(global_acc)
    # clients与server之间通信
    w_attenaution=np.array([1 for i in range(config["num_clients"])],dtype=np.float32)#定义每个客户端的打分的衰减率
    x=0#定期保存训练的参数和acc
    for curr_round in range(lastRound+1, lastRound+rounds + 1):
        local_loss = []
        client_params = {}
        acc = np.zeros(config["num_clients"])#用于打分保存的acc
        loss = np.zeros(config["num_clients"])#用于打分保存的loss
        #遍历所有模型，找到当前参数训练后得分最高的客户端
        for k in range(config["num_clients"]):
            cur_client = client(config)
            subTrainDateset = Subset(data.getTrainData(), data.getDataIndices()[k])
            # 每个client训练得到的权重
            local_parameters = cur_client.localUpdate(localEpoch=config["localEpoch"],
                                                      localBatchSize=config["batchSize"], Net=net,
                                                      lossFun=loss_func,
                                                      opti=opti,
                                                      global_parameters=global_parameters,
                                                      trainDataSet=subTrainDateset, dev=dev)
            accuracy = test_accuracy()
            loss[k], acc[k] = accuracy.test_accuracy(net, local_parameters, data.getTestData(), dev, loss_func)
        scores = Evaluate1(acc, loss, config["w"],w_attenaution)
        indices = np.argsort(-scores)[:int(config["num_clients"] * config["client_rate"])]  # 降序排序后取前n个索引
        print("第%d轮次通信中得分最高的客户端为:"%(curr_round),indices)
        # print("他们的得分如下：")
        # for ind in indices:
        #     print(scores[ind])
            #重新计算衰减率
        for i in range(config["num_clients"]):
            if i in indices:
                w_attenaution[i]=w_attenaution[i]*config["attenuationRate"]
            else:
                w_attenaution[i]=1
        #idChoosed = np.random.choice(numbers, numClientsChoosed, replace=True)#随机抽取
        idChoosed=indices#根据得分抽取
        #选择得分高的客户端进行训练
        for k in range(numClientsChoosed):
            cur_client = client(config)
            subTrainDateset = Subset(data.getTrainData(), data.getDataIndices()[idChoosed[k]])
            # 每个client训练得到的权重
            local_parameters = cur_client.localUpdate(localEpoch=config["localEpoch"], localBatchSize=config["batchSize"], Net=net,
                                                     lossFun=loss_func,
                                                     opti=opti,
                                                     global_parameters=global_parameters,
                                                     trainDataSet=subTrainDateset, dev=dev)
            client_params[k] = local_parameters
            # accuracy = test_accuracy()
            # local_loss, local_acc = accuracy.test_accuracy(net, local_parameters, data.getTestData(), dev, loss_func)
            #if curr_round % 10 == 0:
           # print('[Round: %d Client: %d] accuracy: %f  loss: %f ' % (curr_round, idChoosed[k], local_acc, local_loss))
        # 取平均值，得到本次通信中server得到的更新后的模型参数
        s = server(client_params,config=config)
        global_parameters = s.agg_average()
        net.load_state_dict(global_parameters, strict=True)
        accuracy = test_accuracy()
        global_loss, global_acc = accuracy.test_accuracy(net, global_parameters, data.getTestData(), dev, loss_func)
        print(
            '----------------------------------[Round: %d] accuracy: %f  loss: %f----------------------------------'
             % (curr_round, global_acc, global_loss))
        accuracyGlobal.append(global_acc)
        #定期将参数和acc保存
        x=x+1
        if x % 10 == 0:
            torch.save(net.state_dict(), config["glocalParameterPath"])
            listaccuracy = np.array(accuracyGlobal)
            np.savetxt('./test_accuracy.csv', listaccuracy.reshape(-1, 1), delimiter=',', fmt='%.6f')
            config["lastRound"] = curr_round
            with open('config.json', 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            f.close()
