import json
import numpy as np
import torch
import torch.nn as nn
with open("config.json") as f:
    config = json.load(f)
a=[0,1,2,3]
b=a
b[0]=4
print(a)
# for i in range(0,10):
#     print(i)
# # 模型输出（假设batch_size=2）
# logits = torch.tensor([1.0, 4])  # shape: [2, 10]
# print(torch.softmax(logits, dim=0))
# labels = torch.tensor([1,0])  # shape: [2]
#
# # 计算损失
# criterion = nn.CrossEntropyLoss()
# loss = criterion(logits, labels)  # 自动完成Softmax和One-Hot逻辑
# print(loss)

# np.random.seed(config['random_seed']+10)
# times = [np.random.randint(10, 100) for _ in range(config['num_clients'])]
# np.random.seed(config['random_seed']+20)
# costs=[np.random.randint(10, 100) for _ in range(config['num_clients'])]
# costThreshold=int((np.average(costs)))*int((config['num_clients']*config['client_rate']))
#
# print(times)
# print(costs)
# print(costThreshold)