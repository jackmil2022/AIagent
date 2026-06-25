---
module: "hello-agents"
title: "PyTorch损失函数大全"
description: "理解各种损失函数的用途和选择"
tags: [PyTorch, Loss-Function, Training]
---

# 📊 PyTorch损失函数大全

> **损失函数告诉模型：你错得有多离谱，应该往哪个方向改**

---

## 📝 前言

### 什么是损失函数？

**损失函数（Loss Function）** 衡量模型预测值和真实值之间的差距。

用生活例子理解：

```
考试成绩：
- 预测：90分
- 真实：85分
- 差距：5分（这就是损失）

损失函数就是量化这个"差距"的公式
```

### 为什么需要不同的损失函数？

不同任务需要不同的"评判标准"：

| 任务类型 | 损失函数 | 评判方式 |
|----------|----------|----------|
| 回归 | MSE | 预测值和真实值的差距 |
| 二分类 | BCE | 预测概率和真实标签的差距 |
| 多分类 | CrossEntropy | 预测分布和真实分布的差距 |

---

## 🔰 1. 回归损失函数

### 1.1 L1Loss（MAE，平均绝对误差）

```python
import torch
import torch.nn as nn

# L1Loss: |预测值 - 真实值|
loss_fn = nn.L1Loss()

pred = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([1.5, 2.5, 3.5])

loss = loss_fn(pred, target)
print(f"L1 Loss: {loss}")  # tensor(0.5000)
# 计算: (0.5 + 0.5 + 0.5) / 3 = 0.5
```

**特点**：
- 对异常值鲁棒（不像MSE那样放大异常值）
- 梯度恒定（要么+1要么-1）
- 在零点不可导（实际训练可能有小问题）

### 1.2 MSELoss（均方误差）

```python
# MSELoss: (预测值 - 真实值)²
loss_fn = nn.MSELoss()

pred = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([1.5, 2.5, 3.5])

loss = loss_fn(pred, target)
print(f"MSE Loss: {loss}")  # tensor(0.2500)
# 计算: (0.25 + 0.25 + 0.25) / 3 = 0.25
```

**特点**：
- 最常用的回归损失
- 对异常值敏感（大误差会被放大）
- 梯度随误差增大而增大（学习更快）

### 1.3 HuberLoss（平滑L1损失）

```python
# HuberLoss: 结合L1和MSE的优点
loss_fn = nn.HuberLoss(delta=1.0)  # delta是阈值

pred = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([1.5, 2.5, 3.5])

loss = loss_fn(pred, target)
print(f"Huber Loss: {loss}")
```

**特点**：
- 小误差时像MSE（平滑）
- 大误差时像L1（鲁棒）
- 比MSE更稳定

### 1.4 回归损失对比

```python
import torch
import torch.nn as nn

pred = torch.tensor([1.0, 2.0, 3.0, 100.0])  # 最后一个是异常值
target = torch.tensor([1.5, 2.5, 3.5, 2.0])

l1_loss = nn.L1Loss()(pred, target)
mse_loss = nn.MSELoss()(pred, target)
huber_loss = nn.HuberLoss()(pred, target)

print(f"L1 Loss:    {l1_loss:.4f}")    # 对异常值鲁棒
print(f"MSE Loss:   {mse_loss:.4f}")   # 被异常值主导
print(f"Huber Loss: {huber_loss:.4f}") # 平衡两者
```

---

## 🔰 2. 分类损失函数

### 2.1 BCELoss（二元交叉熵）

```python
# BCELoss: 用于二分类
loss_fn = nn.BCELoss()

# 预测概率（必须在0~1之间）
pred = torch.tensor([0.9, 0.1, 0.8, 0.3])
target = torch.tensor([1.0, 0.0, 1.0, 0.0])  # 真实标签

loss = loss_fn(pred, target)
print(f"BCE Loss: {loss}")
```

**注意**：BCELoss要求输入是概率（0~1），所以模型输出要加Sigmoid。

### 2.2 BCEWithLogitsLoss（推荐！）

```python
# BCEWithLogitsLoss: 内置Sigmoid，更稳定
loss_fn = nn.BCEWithLogitsLoss()

# 直接输入logits（不需要Sigmoid）
logits = torch.tensor([2.0, -2.0, 1.5, -1.0])
target = torch.tensor([1.0, 0.0, 1.0, 0.0])

loss = loss_fn(logits, target)
print(f"BCE with Logits: {loss}")
```

**为什么推荐**：
- 数值更稳定（避免Sigmoid的数值问题）
- 不需要手动加Sigmoid

### 2.3 CrossEntropyLoss（多分类交叉熵）

```python
# CrossEntropyLoss: 用于多分类
loss_fn = nn.CrossEntropyLoss()

# logits（不需要Softmax）
logits = torch.tensor([
    [2.0, 1.0, 0.1],   # 第1个样本
    [0.5, 2.0, 0.3],   # 第2个样本
    [0.3, 0.2, 2.5]    # 第3个样本
])
target = torch.tensor([0, 1, 2])  # 真实类别（0, 1, 2）

loss = loss_fn(logits, target)
print(f"CrossEntropy Loss: {loss}")
```

**注意**：
- 不需要Softmax（内部会做）
- logits形状：(batch_size, num_classes)
- target形状：(batch_size,)，是类别索引

### 2.4 NLLLoss（负对数似然损失）

```python
# NLLLoss: 需要先Softmax
loss_fn = nn.NLLLoss()

# 输入需要是log概率（先Softmax再log）
log_probs = torch.tensor([
    [-0.5, -1.0, -2.0],
    [-1.5, -0.3, -1.2],
    [-1.8, -2.0, -0.2]
])
target = torch.tensor([0, 1, 2])

loss = loss_fn(log_probs, target)
print(f"NLL Loss: {loss}")
```

### 2.5 分类损失对比

```python
import torch
import torch.nn as nn

logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]])
target = torch.tensor([0, 1])

# CrossEntropyLoss（推荐）
ce_loss = nn.CrossEntropyLoss()(logits, target)

# NLLLoss（需要先log_softmax）
log_probs = torch.log_softmax(logits, dim=1)
nll_loss = nn.NLLLoss()(log_probs, target)

print(f"CrossEntropy: {ce_loss:.4f}")
print(f"NLLLoss:      {nll_loss:.4f}")
# 两者结果相同！
```

---

## 🔰 3. 其他常用损失函数

### 3.1 KL散度损失

```python
# KLDivLoss: 衡量两个概率分布的差异
loss_fn = nn.KLDivLoss(reduction='batchmean')

# 输入是log概率，目标是概率
log_pred = torch.log_softmax(torch.randn(3, 5), dim=1)
target = torch.softmax(torch.randn(3, 5), dim=1)

loss = loss_fn(log_pred, target)
print(f"KL Div Loss: {loss}")
```

**用途**：
- 知识蒸馏
- VAE（变分自编码器）
- 分布对齐

### 3.2 Margin Ranking Loss

```python
# MarginRankingLoss: 用于排序任务
loss_fn = nn.MarginRankingLoss(margin=0.5)

x1 = torch.tensor([0.8])
x2 = torch.tensor([0.4])
target = torch.tensor([1])  # x1应该比x2高

loss = loss_fn(x1, x2, target)
print(f"Ranking Loss: {loss}")
```

### 3.3 Triplet Loss

```python
# TripletLoss: 用于度量学习
loss_fn = nn.TripletMarginLoss(margin=1.0)

anchor = torch.randn(1, 128)    # 锚点
positive = torch.randn(1, 128)  # 正样本（同类）
negative = torch.randn(1, 128)  # 负样本（异类）

loss = loss_fn(anchor, positive, negative)
print(f"Triplet Loss: {loss}")
```

---

## 🔰 4. 如何选择损失函数？

### 4.1 选择流程

```
你的任务是什么？
    ↓
├── 回归（预测连续值）
│   ├── 无异常值 → MSELoss
│   ├── 有异常值 → HuberLoss 或 L1Loss
│   └── 需要鲁棒性 → HuberLoss
│
├── 二分类（是/否）
│   └── BCEWithLogitsLoss（推荐）
│
├── 多分类（多个类别）
│   └── CrossEntropyLoss（推荐）
│
└── 特殊任务
    ├── 排序 → MarginRankingLoss
    ├── 度量学习 → TripletLoss
    └── 分布对齐 → KLDivLoss
```

### 4.2 代码示例

```python
import torch
import torch.nn as nn

# 回归任务
class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

model = RegressionModel()
criterion = nn.MSELoss()  # 回归用MSE

# 二分类任务
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)  # 输出1个logit

    def forward(self, x):
        return self.fc(x)  # 不加sigmoid

model = BinaryClassifier()
criterion = nn.BCEWithLogitsLoss()  # 二分类用BCE

# 多分类任务
class MultiClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(10, num_classes)

    def forward(self, x):
        return self.fc(x)  # 不加softmax

model = MultiClassifier(num_classes=10)
criterion = nn.CrossEntropyLoss()  # 多分类用CE
```

---

## 🔰 5. 损失函数实战

### 5.1 完整训练示例

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 生成数据
torch.manual_seed(42)
X = torch.randn(1000, 10)
y = (X[:, 0] + X[:, 1] > 0).long()  # 二分类标签

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 模型
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 2)  # 2类
)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练
for epoch in range(10):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in dataloader:
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
    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")
```

### 5.2 处理类别不平衡

```python
import torch
import torch.nn as nn

# 方法1：加权损失
# 假设类别0有1000个样本，类别1有100个样本
class_weights = torch.tensor([1.0, 10.0])  # 少数类权重更高
criterion = nn.CrossEntropyLoss(weight=class_weights)

# 方法2：Focal Loss（处理极端不平衡）
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# 使用
criterion = FocalLoss()
```

---

## 📚 本节小结

| 任务类型 | 推荐损失函数 | 输入要求 |
|----------|--------------|----------|
| 回归 | MSELoss / HuberLoss | 预测值和真实值 |
| 二分类 | BCEWithLogitsLoss | logits和标签 |
| 多分类 | CrossEntropyLoss | logits和类别索引 |
| 排序 | MarginRankingLoss | 两两比较 |
| 度量学习 | TripletLoss | 锚点、正、负 |

---

## 🎯 下一步

- [[06-PyTorch优化器详解]] - 学习如何用优化器更新参数
- [[07-PyTorch完整训练流程]] - 完整的训练流程

---

> 💡 **实践建议**：
> 1. 分别用不同的损失函数训练，比较结果
> 2. 理解CrossEntropyLoss内部做了什么
> 3. 尝试处理一个类别不平衡的数据集
