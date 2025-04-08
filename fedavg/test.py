import json

import torch
import torch.nn as nn
import torch.optim as optim
from sympy import subsets
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from getData import getData
# 检查CUDA是否可用，并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("config.json") as f:
    config = json.load(f)
f.close()

# 定义数据转换操作
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize((0.5,), (0.5,))  # 标准化处理
])

# 下载并加载训练数据
# train_dataset = torchvision.datasets.FashionMNIST(
#     root='./data', train=True, download=True, transform=transform
# )
data=getData(config)

# 下载并加载测试数据
# test_dataset = torchvision.datasets.FashionMNIST(
#     root='./data', train=False, download=True, transform=transform
# )

test_loader = DataLoader(data.getTestData(), batch_size=64, shuffle=False)


# 定义CNN模型
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(128 * 3 * 3, 512)
#         self.fc2 = nn.Linear(512, 10)
#
#     def forward(self, x):
#         x = self.pool(torch.relu(self.conv1(x)))
#         x = self.pool(torch.relu(self.conv2(x)))
#         x = self.pool(torch.relu(self.conv3(x)))
#         x = x.view(-1, 128 * 3 * 3)  # 展平
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
class CNNforFashionMinst(nn.Module):#增强cnn适用于cifar10
    def __init__(self,in_channels=1, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),  # 64x28x28
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x16x16
            nn.Conv2d(64, 128, 3, padding=1),  # 128x14x14
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128x8x8
            nn.Conv2d(128, 256, 3, padding=1),  # 256x7x7
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)# 256x4x4
            # nn.AdaptiveAvgPool2d((1,1))  # 256x1x1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*4*4, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x=  self.classifier(x)
        return x

# 初始化模型并移动到GPU
# model = CNN().to(device)
model = CNNforFashionMinst().to(device)
# 定义损失函数并移动到GPU
criterion = nn.CrossEntropyLoss().to(device)

# 初始化优化器
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 训练模型
num_epochs = 1000
for i in range(100):
    datatrain=Subset(data.getTrainData(), data.getDataIndices()[i])
    train_loader = DataLoader(datatrain, batch_size=64, shuffle=True)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        sumcorrect = 0
        total = 0
        for images, labels in train_loader:
            # 将数据移动到GPU
            images, labels = images.to(device), labels.to(device)
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            print("loss:", loss.item())
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct = (preds == labels).sum().item()
            sumcorrect += correct
            total += labels.size(0)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / total:.4f}, Accuracy: {sumcorrect / total:.4f}')
    # 测试模型
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    sumcorrect = 0
    with torch.no_grad():
        for images, labels in test_loader:
            # 将数据移动到GPU
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)

            correct = (preds == labels).sum().item()
            sumcorrect += correct
            total += labels.size(0)
    print(f' [{epoch + 1}/{num_epochs}], Loss: {running_loss / total:.4f}, Accuracy: {sumcorrect / total:.4f}')

