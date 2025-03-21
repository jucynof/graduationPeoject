import torch.nn.functional as F
from torch import nn
# ----------------------------------定义CNN神经网络模型----------------------------------
# class Mnist_CNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         self.fc1 = nn.Linear(7 * 7 * 64, 512)
#         self.fc2 = nn.Linear(512, 10)
#
#     def forward(self, inputs):
#         tensor = inputs.view(-1, 1, 28, 28)
#         tensor = F.relu(self.conv1(tensor))
#         tensor = self.pool1(tensor)
#         tensor = F.relu(self.conv2(tensor))
#         tensor = self.pool2(tensor)
#         tensor = tensor.view(-1, 7 * 7 * 64)
#         tensor = F.relu(self.fc1(tensor))
#         tensor = self.fc2(tensor)
#         return tensor
class SimpleCNN(nn.Module):#适用于fashionminst
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),  # 输出32x28x28
            nn.ReLU(),
            nn.MaxPool2d(2),  # 输出32x14x14
            nn.Conv2d(32, 64, 3, padding=1),  # 输出64x14x14
            nn.ReLU(),
            nn.MaxPool2d(2)   # 输出64x7x7
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x)
class EnhancedCNN(nn.Module):#增强cnn适用于cifar10
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),  # 64x32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x16x16
            nn.Conv2d(64, 128, 3, padding=1),  # 128x16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128x8x8
            nn.Conv2d(128, 256, 3, padding=1),  # 256x8x8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))  # 256x1x1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)