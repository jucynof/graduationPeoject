import numpy as np
import torch
import torchvision
from torchvision import transforms
import json
from torch.utils.data import DataLoader
class getNoIIDData:
    train_dataset = None
    test_dataset = None
    split_indices = None
    def __init__(self,config):
        torch.manual_seed(config['random_seed'])  # 设置PyTorch种子
        np.random.seed(config['random_seed'])
        if config['dataName'] == 'fashionMinst':
            # 加载数据集
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

            self.train_dataset = torchvision.datasets.FashionMNIST("./dataset", train=True, download=True,
                                                              transform=transformTrain)
            self.test_dataset = torchvision.datasets.FashionMNIST("./dataset", train=False, download=True,
                                                             transform=transformTest)

            # 将数据进行非独立分布分配给客户端
            # 获取数据标签，并统计类别数量
            targets = np.array(self.train_dataset.targets)
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
            self.split_indices = client_indices
        elif config['dataName'] == 'cifar10':
            # 加载数据集
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
                train_dataset_raw = torchvision.datasets.CIFAR10("./dataset", train=True, download=True,
                                                                 transform=transform_raw)
                test_dataset_raw = torchvision.datasets.CIFAR10("./dataset", train=False, download=True,
                                                                transform=transform_raw)
                # 计算训练集和测试集的均值和标准差
                train_dataset_raw = (torch.tensor(train_dataset_raw.data).float())
                train_dataset_raw=train_dataset_raw.permute(0, 3, 1, 2) # (50000, 3, 32, 32)
                meanTrain = train_dataset_raw.mean(dim=(0, 2, 3)) / 255.0  # 三通道均值 [R_mean, G_mean, B_mean]
                stdTrain = train_dataset_raw.std(dim=(0, 2, 3)) / 255.0  # 三通道标准差 [R_std, G_std, B_std]

                test_dataset_raw = (torch.tensor(test_dataset_raw.data).float())
                test_dataset_raw = test_dataset_raw.permute(0, 3, 1, 2)  # (50000, 3, 32, 32)
                meanTest = test_dataset_raw.mean(dim=(0, 2, 3)) / 255.0  # 三通道均值 [R_mean, G_mean, B_mean]
                stdTest = test_dataset_raw.std(dim=(0, 2, 3)) / 255.0  # 三通道标准差 [R_std, G_std, B_std]

                config["normalizeMeanTrain"] = meanTrain.tolist()
                config["normalizeStdTrain"] = stdTrain.tolist()
                config["normalizeMeanTest"] = meanTest.tolist()
                config["normalizeStdTest"] = stdTest.tolist()
                # 将更新后的配置写回文件
                with open("config.json", "w") as f:
                    json.dump(config, f, indent=4)
                print(f"标准化参数已保存到config.json")
                print(f"训练集 - 均值: {meanTrain}, 标准差: {stdTrain}")
                print(f"测试集 - 均值: {meanTest}, 标准差: {stdTest}")
            # 使用计算得到的值进行标准化
            transformTrain = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=meanTrain, std=stdTrain)
            ])
            transformTest = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=meanTest, std=stdTest)
            ])

            self.train_dataset = torchvision.datasets.CIFAR10("./dataset", train=True, download=True,
                                                              transform=transformTrain)
            self.test_dataset = torchvision.datasets.CIFAR10("./dataset", train=False, download=True,
                                                             transform=transformTest)

            # 将数据进行非独立分布分配给客户端
            # 获取数据标签，并统计类别数量
            targets = np.array(self.train_dataset.targets)
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
            self.split_indices = client_indices
    def getTrainData(self):
        return self.train_dataset
    def getTestData(self):
        return self.test_dataset
    def getDataIndices(self):
        return self.split_indices


class getData:
    train_dataset = None
    test_dataset = None
    split_indices = None
    def __init__(self, config):
        torch.manual_seed(config["random_seed"])  # 设置PyTorch种子
        np.random.seed(config['random_seed'])
        if config["dataName"] == "fashionMinst":
            # 加载MNIST数据集
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
            self.train_dataset = torchvision.datasets.FashionMNIST("./dataset", train=True, download=True,
                                                                   transform=transformTrain)
            self.test_dataset = torchvision.datasets.FashionMNIST("./dataset", train=False, download=True,
                                                                  transform=transformTest)
            self.data_indices = np.arange(len(self.train_dataset))
            np.random.seed(config["random_seed"])
            np.random.shuffle(self.data_indices)
            self.split_indices = np.array_split(self.data_indices, config["num_clients"])
        elif config["dataName"] == "cifar10":
            # 加载MNIST数据集
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
                # 计算训练集和测试集的均值和标准差
                train_dataset_raw = (torch.tensor(train_dataset_raw.data).float())
                train_dataset_raw = train_dataset_raw.permute(0, 3, 1, 2)  # (50000, 3, 32, 32)
                meanTrain = train_dataset_raw.mean(dim=(0, 2, 3)) / 255.0  # 三通道均值 [R_mean, G_mean, B_mean]
                stdTrain = train_dataset_raw.std(dim=(0, 2, 3)) / 255.0  # 三通道标准差 [R_std, G_std, B_std]

                test_dataset_raw = (torch.tensor(test_dataset_raw.data).float())
                test_dataset_raw = test_dataset_raw.permute(0, 3, 1, 2)  # (50000, 3, 32, 32)
                meanTest = test_dataset_raw.mean(dim=(0, 2, 3)) / 255.0  # 三通道均值 [R_mean, G_mean, B_mean]
                stdTest = test_dataset_raw.std(dim=(0, 2, 3)) / 255.0  # 三通道标准差 [R_std, G_std, B_std]

                config["normalizeMeanTrain"] = meanTrain.tolist()
                config["normalizeStdTrain"] = stdTrain.tolist()
                config["normalizeMeanTest"] = meanTest.tolist()
                config["normalizeStdTest"] = stdTest.tolist()
                # 将更新后的配置写回文件
                with open("config.json", "w") as f:
                    json.dump(config, f, indent=4)
                print(f"标准化参数已保存到config.json")
                print(f"训练集 - 均值: {meanTrain}, 标准差: {stdTrain}")
                print(f"测试集 - 均值: {meanTest}, 标准差: {stdTest}")
                # 使用计算得到的值进行标准化
            transformTrain = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=meanTrain, std=stdTrain)
            ])
            transformTest = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=meanTest, std=stdTest)
            ])
            self.train_dataset = torchvision.datasets.FashionMNIST("./dataset", train=True, download=True,
                                                                   transform=transformTrain)
            self.test_dataset = torchvision.datasets.FashionMNIST("./dataset", train=False, download=True,
                                                                  transform=transformTest)
            self.data_indices = np.arange(len(self.train_dataset))
            np.random.seed(config["random_seed"])
            np.random.shuffle(self.data_indices)
            self.split_indices = np.array_split(self.data_indices, config["num_clients"])
    def getTrainData(self):
        return self.train_dataset
    def getTestData(self):
        return self.test_dataset
    def getDataIndices(self):
        return self.split_indices