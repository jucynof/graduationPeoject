import torch
from torch.utils.data import DataLoader
# ----------------------------------客户端----------------------------------
class client:

    def __init__(self,config):
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.train_DataLoader = None
        self.local_parameters = None
        self.localEpoch=None
        self.localBatchSize=None
        self.score=None
    # 模型训练
    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters, trainDataSet, dev):
        # localEpoch: 当前Client的迭代次数
        # localBatchSize: 当前Client的batchsize大小
        # Net: Server共享的模型
        # LossFun: 损失函数
        # opti: 优化函数
        # global_parmeters: 当前通讯中最全局参数
        # return: 返回当前Client基于自己的数据训练得到的新的模型参数
        # 加载当前通信中最新全局参数
        # 并将global_parameters传入网络模型
        Net.load_state_dict(global_parameters, strict=True)
        # 加载本地数据, client自己的数据集

        self.train_DataLoader = DataLoader(trainDataSet, batch_size=int(localBatchSize), shuffle=False)
        self.dev = dev

        # 设置迭代次数
        for epoch in range(localEpoch):
            for data, label in self.train_DataLoader:
                # 加载到GPU上
                data, label = data.to(dev), label.to(dev)
                # 模型上传入数据
                output = Net(data)
                # 计算损失函数
                loss = lossFun(output, label)
                # 将梯度归零，初始化梯度
                opti.zero_grad()
                # 反向传播
                loss.backward()
                # 计算梯度，并更新梯度
                opti.step()

        # 返回当前Client基于自己的数据训练得到的新的模型参数
        return Net.state_dict()

    def local_val(self):
        pass
# ----------------------------------服务器----------------------------------
class server:
    def __init__(self, client_params,config):
        self.client_params = client_params

    def agg_average(self):
        w = self.client_params
        weights_avg = w[0]
        for k in weights_avg.keys():
            for i in range(0, len(w)):
                weights_avg[k] = weights_avg[k] + w[i][k]
            weights_avg[k] = weights_avg[k] / len(w)
        return weights_avg
