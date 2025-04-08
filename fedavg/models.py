import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init


# ----------------------------------定义CNN神经网络模型----------------------------------
# 残差块
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.LeakyReLU(negative_slope=0.01)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels)
#         )
#         # 维度匹配的Shortcut连接
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.conv2(out)
#         out += self.shortcut(x)  # 残差连接
#         return out


class SimpleCNN(nn.Module):#适用于fashionminst
    def __init__(self,config, in_channels=1, num_classes=10):
        super().__init__()
        torch.manual_seed(config['random_seed'])  # 设置PyTorch种子
        np.random.seed(config['random_seed'])
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # 输出32x14x14
            nn.Conv2d(32, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)   # 输出64x7x7
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        self.softmaxed_ = nn.Sequential(
            nn.Softmax(dim=1)  # 新增Softmax层
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        x_softmaxed = self.softmaxed_(x)
        return x, x_softmaxed
class CNNforFashionMinst(nn.Module):#增强cnn适用于cifar10
    def __init__(self, config,in_channels=1, num_classes=10):
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
        self.softmaxed_ = nn.Sequential(
            nn.Softmax(dim=1)  # 新增Softmax层
        )

    def forward(self, x):
        x = self.features(x)
        x=  self.classifier(x)
        x_softmax = self.softmaxed_(x)
        return x,x_softmax
class EnhancedCNN(nn.Module):#增强cnn适用于cifar10
    def __init__(self, config,in_channels=3, num_classes=10):
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
            nn.MaxPool2d(2)# 256x4x4
            # nn.AdaptiveAvgPool2d((1,1))  # 256x1x1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*4*4, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
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
# class CNNforFashionMinst(nn.Module):
#     def __init__(self, config, in_channels=1, num_classes=10):
#         super().__init__()
#         self.features = nn.Sequential(
#             # 初始卷积层
#             nn.Conv2d(in_channels, 64, 3, padding=1),  # 64x28x28
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(negative_slope=0.01),
#             nn.MaxPool2d(2),  # 64x14x14
#
#             # 残差模块组
#             ResidualBlock(64, 128, stride=1),  # 通道扩展残差块
#             nn.MaxPool2d(2),  # 128x7x7
#             ResidualBlock(128, 256, stride=1),  # 通道扩展残差块
#             nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)# 256x4x4  # 256x4x4
#         )
#         # 分类器
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(256 * 4 * 4, 512),
#             nn.LeakyReLU(negative_slope=0.01),
#             nn.Dropout(0.1),
#             nn.Linear(512, num_classes),
#         )
#         self.softmaxed_ = nn.Sequential(
#             nn.Softmax(dim=1)
#         )
#         # 初始化权重
#         self.apply(self._initialize_weights)
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         x_softmax = self.softmaxed_(x)
#         return x, x_softmax
#
#     def _initialize_weights(self, m):
#         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#             init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             if m.bias is not None:
#                 init.zeros_(m.bias)
#         elif isinstance(m, nn.BatchNorm2d):
#             init.ones_(m.weight)
#             init.zeros_(m.bias)
#
#
# class EnhancedCNN(nn.Module):
#     def __init__(self, config, in_channels=3, num_classes=10):
#         super().__init__()
#         self.features = nn.Sequential(
#             # 初始卷积层
#             nn.Conv2d(in_channels, 64, 3, padding=1),  # 64x32x32
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(negative_slope=0.01),
#             nn.MaxPool2d(2),  # 64x16x16
#
#             # 残差模块组
#             ResidualBlock(64, 128, stride=1),  # 通道扩展残差块
#             nn.MaxPool2d(2),  # 128x8x8
#             ResidualBlock(128, 256, stride=1),  # 通道扩展残差块
#             nn.MaxPool2d(2)  # 256x4x4
#         )
#         # 分类器
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(256 * 4 * 4, 512),
#             nn.LeakyReLU(negative_slope=0.01),
#             nn.Dropout(0.1),
#             nn.Linear(512, num_classes),
#         )
#         self.softmaxed_ = nn.Sequential(
#             nn.Softmax(dim=1)
#         )
#         # 初始化权重
#         self.apply(self._initialize_weights)
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         x_softmax = self.softmaxed_(x)
#         return x, x_softmax
#
#     def _initialize_weights(self, m):
#         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#             init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             if m.bias is not None:
#                 init.zeros_(m.bias)
#         elif isinstance(m, nn.BatchNorm2d):
#             init.ones_(m.weight)
#             init.zeros_(m.bias)