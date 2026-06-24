---
title: "PyTorch优化器详解"
description: "理解各种优化器的原理和选择"
tags: [PyTorch, Optimizer, SGD, Adam, Training]
---

# ⚡ PyTorch优化器详解

> **优化器是模型的"学习方法"，决定参数如何更新**

---

## 📝 前言

### 什么是优化器？

**优化器（Optimizer）** 根据梯度更新模型参数，让损失变小。

用生活例子理解：

```
下山找路：
- 损失函数 = 山的高度（目标：找到最低点）
- 梯度 = 坡度（告诉你哪个方向更陡）
- 优化器 = 下山策略（怎么走、走多快）

不同的下山策略：
- SGD：一步一步走，每步看脚下坡度
- Adam：走得多就大步走，走得少就小步走
```

### 为什么需要不同的优化器？

| 优化器 | 特点 | 适用场景 |
|--------|------|----------|
| SGD | 简单、慢、稳定 | 简单问题、调试 |
| Momentum | 加速收敛 | 大多数问题 |
| Adam | 自适应学习率、快 | 复杂问题、默认选择 |
| AdamW | Adam+权重衰减 | 推荐替代Adam |

---

## 🔰 1. 优化器基础

### 1.1 基本使用

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 创建模型
model = nn.Linear(10, 2)

# 创建优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练循环
for epoch in range(10):
    # 前向传播
    x = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, y)

    # 反向传播
    optimizer.zero_grad()  # 清除旧梯度
    loss.backward()        # 计算新梯度
    optimizer.step()       # 更新参数
```

### 1.2 优化器参数

```python
# 所有优化器都有这些参数
optimizer = optim.SGD(
    model.parameters(),   # 要优化的参数
    lr=0.01,              # 学习率（最重要的参数！）
    momentum=0.9,         # 动量（SGD专用）
    weight_decay=1e-4,    # 权重衰减（正则化）
)
```

---

## 🔰 2. 常用优化器

### 2.1 SGD（随机梯度下降）

```python
# 最基础的优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 参数更新公式：
# p = p - lr * gradient
```

**特点**：
- 简单、易于理解
- 收敛慢、可能陷入局部最优
- 需要仔细调学习率

### 2.2 SGD + Momentum（动量）

```python
# 加入动量，加速收敛
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 参数更新公式：
# v = momentum * v + gradient
# p = p - lr * v
```

**类比**：
```
想象一个球滚下山：
- 没有Momentum：球走走停停，容易卡住
- 有Momentum：球有惯性，能冲过小坑
```

### 2.3 Adam（自适应矩估计）

```python
# 最常用的优化器
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,      # Adam通常用更小的学习率
    betas=(0.9, 0.999),  # 动量参数
    eps=1e-8,      # 数值稳定性
    weight_decay=0  # 权重衰减
)

# 简化版（推荐）
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

**特点**：
- 自适应调整每个参数的学习率
- 收敛快、调参简单
- 默认选择（大多数情况）

**原理**：
```
Adam结合了两种技术：
1. Momentum（一阶矩）：用过去梯度的均值
2. RMSprop（二阶矩）：用过去梯度平方的均值

效果：
- 梯度大的参数 → 学习率变小
- 梯度小的参数 → 学习率变大
```

### 2.4 AdamW（解耦权重衰减）

```python
# Adam的改进版（推荐！）
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01  # 权重衰减
)
```

**为什么更好**：
- Adam的weight_decay实现有问题
- AdamW修正了这个问题
- 正则化效果更好

### 2.5 RMSprop

```python
# 自适应学习率
optimizer = optim.RMSprop(
    model.parameters(),
    lr=0.001,
    alpha=0.99,      # 平滑系数
    eps=1e-8,
    momentum=0,
    weight_decay=0
)
```

### 2.6 Adagrad

```python
# 适合稀疏数据
optimizer = optim.Adagrad(
    model.parameters(),
    lr=0.01,
    lr_decay=0,      # 学习率衰减
    weight_decay=0
)
```

---

## 🔰 3. 优化器对比

### 3.1 收敛速度对比

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 创建简单问题
torch.manual_seed(42)
X = torch.randn(1000, 10)
y = (X[:, 0] * 2 + X[:, 1] > 0).long()

# 测试不同优化器
optimizers = {
    'SGD': lambda p: optim.SGD(p, lr=0.01),
    'SGD+Momentum': lambda p: optim.SGD(p, lr=0.01, momentum=0.9),
    'Adam': lambda p: optim.Adam(p, lr=0.001),
    'AdamW': lambda p: optim.AdamW(p, lr=0.001, weight_decay=0.01),
}

for name, opt_fn in optimizers.items():
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2))
    optimizer = opt_fn(model.parameters())
    criterion = nn.CrossEntropyLoss()

    losses = []
    for epoch in range(100):
        output = model(X)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"{name:15s}: 最终loss={losses[-1]:.4f}")
```

### 3.2 选择建议

```python
# 选择流程
if 问题简单 or 调试:
    optimizer = 'SGD'
elif 不确定 or 默认:
    optimizer = 'Adam'  # 或 'AdamW'
elif 需要最好泛化:
    optimizer = 'SGD+Momentum'  # 很多论文用这个
elif 有预训练模型:
    optimizer = 'AdamW'  # 微调推荐
```

---

## 🔰 4. 学习率调度器

### 4.1 为什么需要学习率调度？

```
训练初期：学习率大 → 快速接近最优解
训练后期：学习率小 → 精细调整
```

### 4.2 StepLR（阶梯衰减）

```python
import torch.optim.lr_scheduler as lr_scheduler

optimizer = optim.Adam(model.parameters(), lr=0.01)

# 每30个epoch，学习率乘以0.1
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(100):
    train(...)
    validate(...)

    scheduler.step()  # 更新学习率
    print(f"Epoch {epoch}: lr={scheduler.get_last_lr()[0]}")
```

### 4.3 CosineAnnealingLR（余弦退火）

```python
# 学习率按余弦曲线变化
scheduler = lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,    # 周期
    eta_min=1e-6  # 最小学习率
)
```

### 4.4 ReduceLROnPlateau（自适应衰减）

```python
# 当指标停止改善时降低学习率
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',      # 监控指标变小
    factor=0.1,      # 衰减系数
    patience=10,     # 等待多少个epoch
    verbose=True
)

# 使用
for epoch in range(100):
    train_loss = train(...)
    val_loss = validate(...)

    scheduler.step(val_loss)  # 根据验证损失调整
```

### 4.5 OneCycleLR（单周期）

```python
# 一次训练周期内完成学习率变化
scheduler = lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.01,
    steps_per_epoch=len(train_loader),
    epochs=100
)

# 每个batch后更新
for batch in train_loader:
    ...
    scheduler.step()
```

---

## 🔰 5. 优化器实战

### 5.1 完整训练模板

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def train():
    # 1. 准备数据
    torch.manual_seed(42)
    X_train = torch.randn(1000, 10)
    y_train = torch.randint(0, 2, (1000,))
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 2. 创建模型
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 2)
    )

    # 3. 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # 4. 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # 5. 训练循环
    for epoch in range(50):
        model.train()
        total_loss = 0

        for batch_x, batch_y in train_loader:
            output = model(batch_x)
            loss = criterion(output, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")

    return model

model = train()
```

### 5.2 不同层用不同学习率

```python
# 微调预训练模型时常用
model = nn.Sequential(
    nn.Linear(10, 64),  # 特征提取层
    nn.ReLU(),
    nn.Linear(64, 2)    # 分类层
)

# 特征层用小学习率，分类层用大学习率
optimizer = optim.AdamW([
    {'params': model[0].parameters(), 'lr': 1e-5},  # 特征层
    {'params': model[2].parameters(), 'lr': 1e-3},  # 分类层
])
```

### 5.3 梯度裁剪

```python
# 防止梯度爆炸
model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2))
optimizer = optim.Adam(model.parameters())

for batch_x, batch_y in train_loader:
    output = model(batch_x)
    loss = criterion(output, batch_y)

    optimizer.zero_grad()
    loss.backward()

    # 梯度裁剪：最大范数为1.0
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
```

---

## 🔰 6. 超参数调优

### 6.1 学习率查找器

```python
def lr_finder(model, train_loader, criterion, lr_min=1e-7, lr_max=10, num_iter=100):
    """找到最佳学习率"""

    # 保存初始权重
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}

    optimizer = optim.SGD(model.parameters(), lr=lr_min)
    lr_schedule = np.geomspace(lr_min, lr_max, num_iter)

    losses = []
    lrs = []

    for i, lr in enumerate(lr_schedule):
        # 设置学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 训练一步
        for batch_x, batch_y in train_loader:
            output = model(batch_x)
            loss = criterion(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            break  # 只用一个batch

        losses.append(loss.item())
        lrs.append(lr)

        # 如果loss爆炸，停止
        if loss.item() > 4 * losses[0]:
            break

    # 恢复初始权重
    model.load_state_dict(initial_state)

    # 画图
    import matplotlib.pyplot as plt
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.show()

    # 找到loss最小的学习率
    min_idx = np.argmin(losses)
    return lrs[min_idx]
```

### 6.2 常见超参数

```python
# 学习率（最重要的超参数）
lr = 1e-3  # Adam默认
lr = 1e-2  # SGD常用

# 批大小
batch_size = 32   # 常用
batch_size = 64   # GPU显存足够时
batch_size = 128  # 大batch可能需要调大lr

# 权重衰减
weight_decay = 1e-4  # 轻度正则
weight_decay = 1e-2  # 中度正则
```

---

## 📚 本节小结

| 优化器 | 特点 | 推荐场景 |
|--------|------|----------|
| SGD | 简单、慢 | 调试、简单问题 |
| SGD+Momentum | 加速收敛 | 大多数问题 |
| Adam | 自适应、快 | 默认选择 |
| AdamW | 更好的正则化 | 微调、推荐替代Adam |
| RMSprop | 自适应 | RNN |

**学习率调度**：
- CosineAnnealingLR：通用
- ReduceLROnPlateau：自适应
- OneCycleLR：快速训练

---

## 🎯 下一步

- [[07-PyTorch完整训练流程]] - 完整的训练流程
- [[08-PyTorch模型保存与加载]] - 保存和加载模型

---

> 💡 **实践建议**：
> 1. 用不同优化器训练同一个模型，比较收敛速度
> 2. 尝试学习率调度器
> 3. 用lr_finder找到最佳学习率
> 4. 练习不同层用不同学习率
