---
module: "hello-agents"
title: "PyTorch模型构建：nn.Module"
description: "学习如何用nn.Module搭建神经网络模型"
tags: [PyTorch, nn.Module, Neural-Network, Model]
---

# 🏗️ PyTorch模型构建：nn.Module

> **nn.Module是PyTorch的灵魂，所有神经网络都继承它**

---

## 📝 前言

### 什么是nn.Module？

**nn.Module** 是PyTorch中所有神经网络的**基类**。

它提供了：
- 参数管理（自动追踪权重和偏置）
- 设备移动（.to(device)）
- 保存/加载（state_dict）
- 训练/评估模式切换（train()/eval()）

### 类比理解

```
nn.Module = 乐高积木的底座
nn.Linear = 长条积木
nn.Conv2d = 方块积木
nn.ReLU = 连接器

用nn.Module把各种积木拼起来 → 搭出神经网络
```

---

## 🔰 1. nn.Module基础

### 1.1 最简单的模型

```python
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()  # 必须调用父类初始化
        # 定义层
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        # 定义前向传播
        return self.linear(x)

# 使用
model = SimpleModel()
print(model)
# SimpleModel(
#   (linear): Linear(in_features=10, out_features=2, bias=True)
# )

# 前向传播
x = torch.randn(32, 10)  # batch_size=32, features=10
y = model(x)
print(y.shape)  # torch.Size([32, 2])
```

### 1.2 forward方法的重要性

```python
# forward方法定义了数据如何流过网络

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 64)
        self.linear2 = nn.Linear(64, 2)

    def forward(self, x):
        # 数据流：input → linear1 → relu → linear2 → output
        x = self.linear1(x)
        x = torch.relu(x)  # 激活函数
        x = self.linear2(x)
        return x

# 注意：不要直接调用model.forward(x)
# 应该用model(x)或model.forward(x)
# 因为model(x)会触发钩子等额外功能
```

---

## 🔰 2. 常用神经网络层

### 2.1 全连接层（Linear）

```python
import torch.nn as nn

# Linear: y = x @ W^T + b
linear = nn.Linear(
    in_features=10,   # 输入特征数
    out_features=5,   # 输出特征数
    bias=True         # 是否有偏置
)

print(linear.weight.shape)  # torch.Size([5, 10])
print(linear.bias.shape)    # torch.Size([5])

# 使用
x = torch.randn(32, 10)
y = linear(x)
print(y.shape)  # torch.Size([32, 5])
```

### 2.2 卷积层（Conv2d）

```python
# Conv2d: 2D卷积，用于图片
conv = nn.Conv2d(
    in_channels=3,     # 输入通道（RGB=3）
    out_channels=16,   # 输出通道（卷积核数量）
    kernel_size=3,     # 卷积核大小
    stride=1,          # 步长
    padding=1          # 填充
)

print(conv.weight.shape)  # torch.Size([16, 3, 3, 3])
# (out_channels, in_channels, kernel_h, kernel_w)

# 使用
x = torch.randn(1, 3, 32, 32)  # (batch, channels, height, width)
y = conv(x)
print(y.shape)  # torch.Size([1, 16, 32, 32])
```

### 2.3 池化层（MaxPool2d）

```python
# MaxPool2d: 最大池化，降低特征图大小
pool = nn.MaxPool2d(
    kernel_size=2,  # 池化窗口
    stride=2        # 步长
)

# 使用
x = torch.randn(1, 16, 32, 32)
y = pool(x)
print(y.shape)  # torch.Size([1, 16, 16, 16])
# 尺寸减半
```

### 2.4 批归一化（BatchNorm）

```python
# BatchNorm: 稳定训练，加速收敛
batch_norm = nn.BatchNorm2d(16)  # 对16个通道归一化

# 使用
x = torch.randn(1, 16, 32, 32)
y = batch_norm(x)
print(y.shape)  # torch.Size([1, 16, 32, 32])
# 形状不变，但数据被归一化
```

### 2.5 Dropout

```python
# Dropout: 随机丢弃神经元，防止过拟合
dropout = nn.Dropout(p=0.5)  # 50%概率丢弃

# 使用
x = torch.ones(1, 10)
y = dropout(x)
print(y)  # 有些位置变成0
```

---

## 🔰 3. 激活函数

### 3.1 常用激活函数

```python
import torch
import torch.nn as nn

x = torch.randn(5)

# ReLU: 最常用
relu = nn.ReLU()
y_relu = relu(x)
print(f"ReLU: {y_relu}")

# Sigmoid: 输出0~1
sigmoid = nn.Sigmoid()
y_sigmoid = sigmoid(x)
print(f"Sigmoid: {y_sigmoid}")

# Tanh: 输出-1~1
tanh = nn.Tanh()
y_tanh = tanh(x)
print(f"Tanh: {y_tanh}")

# Softmax: 多分类输出
softmax = nn.Softmax(dim=0)
y_softmax = softmax(x)
print(f"Softmax: {y_softmax}")
print(f"Softmax和: {y_softmax.sum()}")  # =1
```

### 3.2 激活函数选择

```python
# 二分类输出：Sigmoid
self.output = nn.Linear(64, 1)
# 输出用sigmoid或BCEWithLogitsLoss

# 多分类输出：Softmax（或不加，用CrossEntropyLoss）
self.output = nn.Linear(64, num_classes)
# 输出直接传给CrossEntropyLoss

# 隐藏层：ReLU（推荐）
self.hidden = nn.Linear(10, 64)
# 用ReLU激活
```

---

## 🔰 4. 构建完整模型

### 4.1 全连接网络（MLP）

```python
import torch.nn as nn

class MLP(nn.Module):
    """多层感知机"""

    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.network(x)
        return x

# 使用
model = MLP()
print(model)

x = torch.randn(32, 1, 28, 28)  # MNIST图片
y = model(x)
print(y.shape)  # torch.Size([32, 10])
```

### 4.2 卷积神经网络（CNN）

```python
import torch.nn as nn

class CNN(nn.Module):
    """简单CNN"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 第三个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)
        return x

# 使用
model = CNN()
print(model)

x = torch.randn(32, 3, 32, 32)  # CIFAR10图片
y = model(x)
print(y.shape)  # torch.Size([32, 10])
```

### 4.3 使用预定义层

```python
import torch.nn as nn

# PyTorch内置了很多经典模型
from torchvision import models

# ResNet18
model = models.resnet18(pretrained=True)

# 修改最后一层（迁移学习）
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)  # 10类分类

# 冻结前面的层
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad_(True)  # 只训练最后一层
```

---

## 🔰 5. 模型参数管理

### 5.1 查看参数

```python
model = MLP()

# 查看所有参数
print("参数形状:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}")

# 参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n总参数: {total_params:,}")
print(f"可训练参数: {trainable_params:,}")
```

### 5.2 冻结参数

```python
# 冻结所有参数
for param in model.parameters():
    param.requires_grad = False

# 只解冻特定层
for param in model.network[-1].parameters():
    param.requires_grad = True

# 或者用requires_grad_()
model.network[-1].weight.requires_grad_(True)
model.network[-1].bias.requires_grad_(True)
```

### 5.3 参数初始化

```python
def init_weights(m):
    """自定义权重初始化"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

# 应用初始化
model = MLP()
model.apply(init_weights)
```

---

## 🔰 6. 训练模式与评估模式

### 6.1 train() vs eval()

```python
model = CNN()

# 训练模式（默认）
model.train()
# Dropout和BatchNorm会工作
# Dropout随机丢弃神经元
# BatchNorm用当前batch的统计量

# 评估模式
model.eval()
# Dropout不丢弃
# BatchNorm用训练时的统计量
```

### 6.2 为什么需要切换？

```python
# 训练时
model.train()
for batch_x, batch_y in train_loader:
    output = model(batch_x)
    loss = criterion(output, batch_y)
    # ...

# 评估时
model.eval()
with torch.no_grad():  # 不计算梯度
    for batch_x, batch_y in val_loader:
        output = model(batch_x)
        # ...
```

### 6.3 完整训练/评估循环

```python
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # 前向传播
        output = model(batch_x)
        loss = criterion(output, batch_y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            output = model(batch_x)
            loss = criterion(output, batch_y)

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy
```

---

## 🔰 7. 模型设备管理

### 7.1 移动模型到GPU

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 创建模型
model = CNN()

# 移动模型到GPU
model = model.to(device)

# 确保数据也在GPU上
for batch_x, batch_y in train_loader:
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    output = model(batch_x)
    # ...
```

### 7.2 多GPU训练（DataParallel）

```python
import torch.nn as nn

model = CNN()

if torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 个GPU")
    model = nn.DataParallel(model)

model = model.to(device)
```

---

## 🔰 8. 模型摘要

### 8.1 打印模型结构

```python
model = CNN()
print(model)
```

### 8.2 使用torchsummary

```python
from torchsummary import summary

model = CNN().to(device)
summary(model, input_size=(3, 32, 32))
# 输出：
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1            [-1, 32, 32, 32]             896
#        BatchNorm2d-2            [-1, 32, 32, 32]              64
#             ReLU-3            [-1, 32, 32, 32]               0
#        MaxPool2d-4            [-1, 32, 16, 16]               0
#             ...
# ================================================================
# Total params: 309,770
# Trainable params: 309,770
# Non-trainable params: 0
# ----------------------------------------------------------------
```

---

## 📚 本节小结

| 概念 | 说明 |
|------|------|
| nn.Module | 所有神经网络的基类 |
| forward() | 定义前向传播 |
| 常用层 | Linear, Conv2d, MaxPool2d, BatchNorm, Dropout |
| 激活函数 | ReLU, Sigmoid, Tanh, Softmax |
| Sequential | 顺序容器，简化代码 |
| 参数管理 | named_parameters(), requires_grad |
| 训练/评估 | train(), eval() |
| 设备管理 | .to(device) |

---

## 🎯 下一步

- [[05-PyTorch损失函数大全]] - 选择合适的损失函数
- [[06-PyTorch优化器详解]] - 理解优化器如何工作

---

> 💡 **实践建议**：
> 1. 尝试用不同的层搭建自己的网络
> 2. 打印模型结构，理解数据流动
> 3. 练习冻结和解冻参数
> 4. 分别用CPU和GPU训练，比较速度
