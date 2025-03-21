import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
# ----------------------------------定义CNN神经网络模型----------------------------------

class SimpleCNN(nn.Module):#适用于fashionminst
    def __init__(self,config, in_channels=1, num_classes=10):
        super().__init__()
        torch.manual_seed(config['random_seed'])  # 设置PyTorch种子
        np.random.seed(config['random_seed'])
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, padding=2),  # 输出32x28x28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # 输出32x14x14
            nn.Conv2d(32, 64, 5, padding=2),  # 输出64x14x14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)   # 输出64x7x7
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )
        self.softmaxed_ = nn.Sequential(
            nn.Softmax(dim=1)  # 新增Softmax层
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        x_softmaxed = self.softmaxed_(x)
        return x, x_softmaxed
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
            nn.Linear(512, num_classes),
        )
        self.softmaxed_ = nn.Sequential(
            nn.Softmax(dim=1)  # 新增Softmax层
        )

    def forward(self, x):
        x = self.features(x)
        x=  self.classifier(x)
        x_softmax = self.softmaxed_(x)
        return x,x_softmax