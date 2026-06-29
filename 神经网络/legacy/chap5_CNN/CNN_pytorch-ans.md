---
title: "chap5 CNN手写数字识别参考答案"
tags: [神经网络, nndl]
---

# CNN 手写数字识别（参考答案）

两层卷积 + 全连接，MNIST 分类。

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

## 数据

```python
transform = transforms.ToTensor()
train_set = datasets.MNIST(root='./mnist/', train=True,  download=True, transform=transform)
test_set  = datasets.MNIST(root='./mnist/', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=500,       shuffle=False)
```

## 模型

关键点：
- 7x7 卷积 padding=3 保持 28x28 不变；5x5 卷积 padding=2 同理。
- 两次 2x2 池化把空间维降到 7x7。
- 展平用 `x.view(x.size(0), -1)`。
- 最后一层输出原始 logits，不要叠 softmax；`CrossEntropyLoss` 内部会算。

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc1 = nn.Linear(7 * 7 * 64, 1024)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
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
            loss = loss_fn(model(x), y)
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
