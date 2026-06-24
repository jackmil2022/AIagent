---
title: "深度学习：PyTorch实战"
description: "使用PyTorch构建神经网络"
tags: [Deep-Learning, PyTorch, Tensor, Autograd]
---

# 🔥 深度学习：PyTorch实战

> **用PyTorch实现深度学习**

---

## 📝 前言

PyTorch是最流行的深度学习框架之一。

本章将带你用PyTorch构建神经网络。

---

## 🔰 1. PyTorch基础

### 1.1 Tensor

**Tensor = 多维数组（类似NumPy）**

```python
import torch

# 创建Tensor
x = torch.tensor([1, 2, 3])
print(x)

# 随机Tensor
x = torch.randn(3, 4)
print(x.shape)  # torch.Size([3, 4])

# GPU加速
if torch.cuda.is_available():
    x = x.cuda()
```

### 1.2 自动微分

**PyTorch的核心能力**

```python
import torch

# 创建需要梯度的Tensor
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2

# 反向传播
y.backward()

# 查看梯度
print(f"dy/dx = {x.grad}")  # tensor([4.])
```

---

## 🔰 2. 构建神经网络

### 2.1 使用nn.Module

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(5, 1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# 创建模型
model = SimpleNet()
print(model)
```

### 2.2 完整训练流程

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 1. 准备数据
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).reshape(-1, 1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)

# 2. 定义模型
model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid()
)

# 3. 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练
for epoch in range(100):
    # 前向传播
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

# 5. 评估
with torch.no_grad():
    outputs = model(X_test)
    predicted = (outputs > 0.5).float()
    accuracy = (predicted == y_test).float().mean()
    print(f"准确率：{accuracy:.2%}")
```

---

## 🔰 3. 常用层

### 3.1 线性层

```python
linear = nn.Linear(in_features=10, out_features=5)
x = torch.randn(1, 10)
output = linear(x)
print(output.shape)  # torch.Size([1, 5])
```

### 3.2 卷积层

```python
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
x = torch.randn(1, 3, 32, 32)  # batch, channels, height, width
output = conv(x)
print(output.shape)  # torch.Size([1, 16, 32, 32])
```

### 3.3 循环层

```python
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
x = torch.randn(1, 5, 10)  # batch, seq_len, features
output, (h_n, c_n) = lstm(x)
print(output.shape)  # torch.Size([1, 5, 20])
```

---

## 🔰 4. 保存和加载模型

### 4.1 保存模型

```python
# 保存整个模型
torch.save(model, 'model.pth')

# 保存参数（推荐）
torch.save(model.state_dict(), 'model_weights.pth')
```

### 4.2 加载模型

```python
# 加载参数
model = SimpleNet()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()  # 设置为评估模式
```

---

## 🔰 5. GPU加速

### 5.1 使用GPU

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 移动模型到GPU
model = model.to(device)

# 移动数据到GPU
X_train = X_train.to(device)
y_train = y_train.to(device)
```

---

## 📚 本节小结

| 概念 | 说明 |
|------|------|
| Tensor | 多维数组 |
| autograd | 自动微分 |
| nn.Module | 网络模块 |
| 优化器 | 参数更新 |

---

## 🎯 下一步

- **04d - 深度学习：CNN卷积神经网络** - 图像处理
- **05c - NLP基础：序列模型** - RNN、LSTM

---

> 💡 **学习建议**：用PyTorch实现一个简单的分类任务，体验完整流程。
