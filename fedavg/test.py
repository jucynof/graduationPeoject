import torch
import torch.nn as nn

# 模型输出（假设batch_size=2）
logits = torch.tensor([9999.0, 1])  # shape: [2, 10]
labels = torch.tensor([1,0])  # shape: [2]

# 计算损失
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, labels)  # 自动完成Softmax和One-Hot逻辑
print(loss)