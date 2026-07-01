---
title: "chap5 CNN手写数字识别"
tags: [神经网络, nndl]
---

# CNN 手写数字识别

用 PyTorch 实现一个两层卷积网络在 MNIST 上分类。需要补全 `conv1`、`conv2` 中卷积层的参数，以及 `forward` 中展平张量的形状。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

learning_rate = 1e-4
dropout_rate = 0.3
max_epoch = 3
batch_size = 50
```

## 准备数据

用 `torchvision.datasets.MNIST` 自动下载并加载数据；`DataLoader` 负责分批和打乱。

```python
transform = transforms.ToTensor()
train_set = datasets.MNIST(root='./mnist/', train=True,  download=True, transform=transform)
test_set  = datasets.MNIST(root='./mnist/', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=500,       shuffle=False)
```

## 构建模型

网络结构：
- conv1：7x7 卷积，1 通道→32 通道，stride=1，padding 使输入输出空间尺寸不变 → ReLU → 2x2 MaxPool
- conv2：5x5 卷积，32 通道→64 通道，stride=1，padding 使输入输出空间尺寸不变 → ReLU → 2x2 MaxPool
- fc1：`7*7*64 → 1024`，ReLU + Dropout
- fc2：`1024 → 10`，输出 logits（不要在模型里加 softmax，`CrossEntropyLoss` 内部会做）

**填空 1**：`conv1` 的 5 个参数。
**填空 2**：`conv2` 的三行（卷积、激活、池化）。
**填空 3**：`forward` 中把 conv2 输出展平到 `(batch_size, 7*7*64)` 的 `view`。

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=    ,  # ???
                out_channels=   ,  # ???
                kernel_size=    ,  # ???
                stride=         ,  # ???
                padding=        ,  # ???
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            # ??? 5x5 卷积，32->64，padding 保持尺寸，stride=1
            # ??? 激活函数
            # ??? 2x2 最大池化
        )
        self.fc1 = nn.Linear(7 * 7 * 64, 1024)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(      )  # ??? 展平到 (batch_size, 7*7*64)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)  # 返回 logits
```

## 训练与评估

```python
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
    return correct / total

def train(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn   = nn.CrossEntropyLoss()
    for epoch in range(max_epoch):
        model.train()
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                acc = evaluate(model, test_loader)
                print(f'epoch {epoch} step {step:4d} | loss {loss.item():.4f} | test acc {acc:.4f}')
                model.train()
```

```python
model = CNN().to(device)
train(model)
print('final test accuracy:', evaluate(model, test_loader))
```
