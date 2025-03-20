import numpy as np
import torch
from torch import nn, optim
import json
from torch.utils.data import DataLoader, Subset,TensorDataset
from models import Mnist_CNN
from fed import client,server
from getData import getData, getNoIIDData
from utils import Evaluate1
with open("config.json") as f:
    config = json.load(f)

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
        return correctRate,lossAvg


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
    # 定义优化器
    if config["isIID"]:
        print("IID")
        data = getData(config)
    else:
        print("Not IID")
        data = getNoIIDData(config)

    # ----------------------------------设置参数----------------------------------
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(dev)
    config["dev"]=dev
    rounds=config["rounds"]
    numClients=config["num_clients"]
    numbers = np.arange(numClients)  # 创建一个包含numClients个数字的数组

    # scores=[0 for i in range(numClients)]#创建数组保存每个客户端的得分
    # indices = np.argsort(-scores)[:config["num_Clients"]*config["client_rate"]]  # 降序排序后取前n个索引


    clientRate=config["client_rate"]
    numClientsChoosed=int(numClients*clientRate)
    # ----------------------------------初始化模型----------------------------------
    # 模型实例化
    net = Mnist_CNN()
    net = net.to(dev)
    # 定义损失函数
    loss_func = nn.CrossEntropyLoss()
    loss_func = loss_func.to(dev)
    # 定义优化器
    if config["isIID"]:
        lr=config["lrIID"]
    else:
        lr=config["lrNoIID"]
    opti = optim.Adam(net.parameters(), lr=lr)

    # 定义变量global_parameters
    global_parameters = net.state_dict()
    #储存每次通信accuracy
    accuracyGlobal=np.array([0 for i in range(config["rounds"]+1)],dtype=np.float32)
    #通信前的accuracy
    accuracy = test_accuracy()
    global_loss, global_acc = accuracy.test_accuracy(net, global_parameters, data.getTestData(), dev, loss_func)
    print(
        '----------------------------------[Round: %d] accuracy: %f  loss: %f----------------------------------'
        % (0, global_acc, global_loss))
    accuracyGlobal[0]=global_acc
    # clients与server之间通信
    w_attenaution=np.array([1 for i in range(config["num_clients"])],dtype=np.float32)#定义每个客户端的打分的衰减率
    x=0
    for curr_round in range(1, rounds + 1):
        local_loss = []
        client_params = {}
        acc = np.zeros(config["num_clients"])
        loss = np.zeros(config["num_clients"])
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
            acc[k], loss[k] = accuracy.test_accuracy(net, local_parameters, data.getTestData(), dev, loss_func)
        scores = Evaluate1(acc, loss, config["w"],w_attenaution)
        indices = np.argsort(-scores)[:int(config["num_clients"] * config["client_rate"])]  # 降序排序后取前n个索引
        print("第%d轮次通信中得分最高的客户端为:"%(curr_round),indices)
        print("他们的得分如下：")
        for ind in indices:
            print(scores[ind])
            #重新计算衰减率
        for i in range(config["num_clients"]):
            if i in indices:
                w_attenaution[i]=w_attenaution[i]*config["attenuationRate"]
            else:
                w_attenaution[i]=1
        #idChoosed = np.random.choice(numbers, numClientsChoosed, replace=True)#随机抽取
        idChoosed=indices#根据得分抽取
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
            accuracy = test_accuracy()
            local_loss, local_acc = accuracy.test_accuracy(net, local_parameters, data.getTestData(), dev, loss_func)
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
             % (curr_round+1, global_acc, global_loss))
        accuracyGlobal[curr_round] = global_acc
        x=x+1
        if x % 10 == 0:
            np.savetxt('./Zero.csv', accuracyGlobal.reshape(-1, 1), delimiter=',', fmt='%.6f')
