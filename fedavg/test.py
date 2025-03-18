import numpy as np
import torch
import torchvision
from torchvision import transforms
import json
from torch.utils.data import DataLoader
with open("config.json") as f:
    config = json.load(f)
train_dataset = None
test_dataset = None
split_indices = None
#加载数据集
# 在加载数据集之前，先计算数据集的均值和标准差
        # 检查config中是否存在标准化参数
if "normalizeMeanTrain" in config and "normalizeStdTrain" in config and "normalizeMeanTest" in config and "normalizeStdTest" in config:
    print("从config.json加载标准化参数")
    meanTrain = torch.tensor(config["normalizeMeanTrain"])
    stdTrain = torch.tensor(config["normalizeStdTrain"])
    meanTest = torch.tensor(config["normalizeMeanTest"])
    stdTest = torch.tensor(config["normalizeStdTest"])
else:
    transform_raw = transforms.ToTensor()
    train_dataset_raw = torchvision.datasets.FashionMNIST("./dataset", train=True, download=True,
                                                          transform=transform_raw)
    test_dataset_raw = torchvision.datasets.FashionMNIST("./dataset", train=False, download=True,
                                                         transform=transform_raw)
    # train_dataset_raw = torchvision.datasets.MNIST("./dataset", train=True, download=True, transform=transform_raw)
    # test_dataset_raw = torchvision.datasets.MNIST("./dataset", train=False, download=True, transform=transform_raw)
    train_loader_raw = DataLoader(train_dataset_raw, batch_size=len(train_dataset_raw))
    test_loader_raw = DataLoader(test_dataset_raw, batch_size=len(test_dataset_raw))
    # 计算训练集和测试集的均值和标准差
    meanTrain = train_dataset_raw.data.float().mean() / 255.0
    stdTrain = train_dataset_raw.data.float().std() / 255.0
    meanTest = test_dataset_raw.data.float().mean() / 255.0
    stdTest = test_dataset_raw.data.float().std() / 255.0
    config["normalizeMeanTrain"] = meanTrain.item()
    config["normalizeStdTrain"] = stdTrain.item()
    config["normalizeMeanTest"] = meanTest.item()
    config["normalizeStdTest"] = stdTest.item()
    # 将更新后的配置写回文件
    with open("config.json", "w") as f:
        json.dump(config, f, indent=4)
    print(f"标准化参数已保存到config.json")
    print(f"训练集 - 均值: {meanTrain:.4f}, 标准差: {stdTrain:.4f}")
    print(f"测试集 - 均值: {meanTest:.4f}, 标准差: {stdTest:.4f}")
# 使用计算得到的值进行标准化
transformTrain = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((meanTrain.item(),), (stdTrain.item(),))
])
transformTest = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((meanTest.item(),), (stdTest.item(),))
])
train_dataset = torchvision.datasets.FashionMNIST("./dataset", train=True, download=True,
                                                  transform=transformTrain)
test_dataset = torchvision.datasets.FashionMNIST("./dataset", train=False, download=True,
                                                 transform=transformTest)
# 将数据进行非独立分布分配给客户端
# 获取数据标签，并统计类别数量
targets = np.array(train_dataset.targets)
num_classes = np.unique(targets).size
# 初始化每个客户端的数据索引字典
client_indices = {i: [] for i in range(config["num_clients"])}

# 针对每个类别进行数据分配
for cls in range(num_classes):
    # 获取该类别所有样本的索引，并打乱顺序
    cls_idx = np.where(targets == cls)[0]
    np.random.shuffle(cls_idx)

    # 对于当前类别，根据Dirichlet分布采样每个客户端分配比例

    proportions = np.random.dirichlet(alpha=[config["alpha"]] * config["num_clients"])
    # 根据比例计算各客户端分得的样本数量
    proportions = (proportions * len(cls_idx)).astype(int)
  #  print(proportions)
    # 由于取整可能导致总数不匹配，需要修正分配数量
    diff = len(cls_idx) - np.sum(proportions)
    # 将剩余的样本随机分配给一些客户端
    for i in range(diff):
        proportions[i % config["num_clients"]] += 1

    # 按照计算好的数量划分数据，并添加到对应客户端
    start = 0
    for client in range(config["num_clients"]):
        num_samples = proportions[client]
        client_indices[client].extend(cls_idx[start:start + num_samples])
        start += num_samples
print(len(client_indices))
for i in range(100):
    print(i)
    print(len(client_indices[i]))
    print(client_indices[i])


