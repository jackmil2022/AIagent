---
title: "PyTorch完整训练流程"
description: "从数据到模型，完整的训练流程模板"
tags: [PyTorch, Training, Workflow, Complete]
---

# 🎯 PyTorch完整训练流程

> **把前面学的串起来，完成一个完整的训练任务**

---

## 📝 前言

### 训练神经网络的步骤

```
1. 准备数据
   ↓
2. 创建模型
   ↓
3. 定义损失函数
   ↓
4. 定义优化器
   ↓
5. 训练循环
   ↓
6. 评估模型
   ↓
7. 保存/部署
```

### 类比做菜

```
做菜 = 训练模型
食材 = 数据
菜谱 = 模型架构
评分标准 = 损失函数
厨师 = 优化器
做菜过程 = 训练循环
```

---

## 🔰 1. 完整代码模板

### 1.1 分类任务模板

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# ==================== 1. 准备数据 ====================
def prepare_data():
    """准备数据"""
    # 生成模拟数据
    np.random.seed(42)
    X = np.random.randn(2000, 10)
    y = (X[:, 0] * 2 + X[:, 1] - X[:, 2] > 0).astype(int)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 转为Tensor
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)

    # 创建DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader

# ==================== 2. 创建模型 ====================
def create_model():
    """创建模型"""
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(32, 2)
    )
    return model

# ==================== 3. 训练函数 ====================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
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
        total_loss += loss.item() * batch_x.size(0)
        _, predicted = output.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()

    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# ==================== 4. 评估函数 ====================
@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        output = model(batch_x)
        loss = criterion(output, batch_y)

        total_loss += loss.item() * batch_x.size(0)
        _, predicted = output.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()

    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# ==================== 5. 主函数 ====================
def main():
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据
    train_loader, test_loader = prepare_data()

    # 模型
    model = create_model().to(device)
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # 学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # 训练循环
    best_acc = 0
    for epoch in range(50):
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 评估
        val_loss, val_acc = evaluate(
            model, test_loader, criterion, device
        )

        # 更新学习率
        scheduler.step()

        # 打印
        print(f"Epoch {epoch+1:2d}/50 | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

    print(f"\n最佳验证准确率: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
```

---

## 🔰 2. 完整代码解析

### 2.1 数据准备

```python
# 关键点：
# 1. 划分训练集/验证集/测试集
# 2. 数据标准化
# 3. 创建DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 标准化（必须！）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # 只在训练集上fit
X_test = scaler.transform(X_test)        # 用同样的参数transform测试集

# 转为Tensor
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
```

### 2.2 模型创建

```python
# 关键点：
# 1. 选择合适的架构
# 2. 输出层要匹配任务
# 3. 使用BatchNorm和Dropout

model = nn.Sequential(
    nn.Linear(10, 64),      # 输入层
    nn.BatchNorm1d(64),     # 批归一化
    nn.ReLU(),              # 激活函数
    nn.Dropout(0.3),        # 防止过拟合
    nn.Linear(64, 32),      # 隐藏层
    nn.BatchNorm1d(32),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(32, 2)        # 输出层（2类）
)
```

### 2.3 训练循环

```python
# 关键点：
# 1. model.train() 和 model.eval()
# 2. optimizer.zero_grad()
# 3. loss.backward()
# 4. optimizer.step()
# 5. torch.no_grad() 评估时

for epoch in range(num_epochs):
    # 训练
    model.train()
    for batch_x, batch_y in train_loader:
        output = model(batch_x)
        loss = criterion(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 评估
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            output = model(batch_x)
            loss = criterion(output, batch_y)
```

### 2.4 保存和加载

```python
# 保存模型
torch.save(model.state_dict(), "model.pth")

# 加载模型
model = create_model()
model.load_state_dict(torch.load("model.pth"))
model.eval()
```

---

## 🔰 3. 常见问题解决

### 3.1 Loss不下降

```python
# 可能原因和解决方案：
# 1. 学习率太大或太小
#    → 用lr_finder找最佳学习率
#    → Adam用0.001，SGD用0.01

# 2. 数据有问题
#    → 检查数据是否正确
#    → 检查标签是否对齐

# 3. 模型太简单
#    → 增加层数或神经元

# 4. 梯度问题
#    → 检查梯度是否消失/爆炸
#    → 用torch.nn.utils.clip_grad_norm_
```

### 3.2 过拟合

```python
# 现象：训练集准确率高，验证集准确率低

# 解决方案：
# 1. 增加数据
#    → 数据增强
#    → 收集更多数据

# 2. 正则化
#    → Dropout
#    → Weight Decay
#    → BatchNorm

# 3. 早停
class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return False  # 继续训练
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # 停止训练
            return False
```

### 3.3 欠拟合

```python
# 现象：训练集和验证集准确率都低

# 解决方案：
# 1. 增加模型复杂度
#    → 更多层
#    → 更多神经元

# 2. 训练更久
#    → 增加epoch

# 3. 减少正则化
#    → 减小Dropout
#    → 减小Weight Decay
```

---

## 🔰 4. TensorBoard可视化

```python
from torch.utils.tensorboard import SummaryWriter

# 创建writer
writer = SummaryWriter("runs/experiment_1")

# 训练循环
for epoch in range(100):
    train_loss, train_acc = train_one_epoch(...)
    val_loss, val_acc = evaluate(...)

    # 记录指标
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch)
    writer.add_scalar("Accuracy/val", val_acc, epoch)
    writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)

    # 记录模型图
    if epoch == 0:
        writer.add_graph(model, torch.randn(1, 10).to(device))

    # 记录权重直方图
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)
        writer.add_histogram(f"{name}.grad", param.grad, epoch)

writer.close()

# 启动TensorBoard
# tensorboard --logdir=runs
```

---

## 🔰 5. 完整项目结构

```
my_project/
├── data/                    # 数据目录
│   ├── train/
│   └── test/
├── models/                  # 模型定义
│   ├── __init__.py
│   └── my_model.py
├── utils/                   # 工具函数
│   ├── __init__.py
│   └── data_utils.py
├── train.py                 # 训练脚本
├── evaluate.py              # 评估脚本
├── predict.py               # 预测脚本
├── config.py                # 配置文件
├── requirements.txt         # 依赖
└── README.md
```

### 5.1 config.py

```python
# 配置文件
class Config:
    # 数据
    data_path = "./data"
    train_ratio = 0.8
    batch_size = 32

    # 模型
    input_size = 10
    hidden_size = 64
    num_classes = 2

    # 训练
    num_epochs = 100
    learning_rate = 0.001
    weight_decay = 1e-4
    device = "cuda"  # or "cpu"

    # 保存
    save_path = "./checkpoints"
    log_path = "./logs"
```

### 5.2 train.py

```python
from config import Config
from models import MyModel
from utils import prepare_data
import torch
import torch.nn as nn
import torch.optim as optim

def main():
    config = Config()

    # 准备数据
    train_loader, val_loader = prepare_data(config)

    # 创建模型
    model = MyModel(config).to(config.device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # 训练
    for epoch in range(config.num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, config.device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, config.device
        )

        print(f"Epoch {epoch+1}: "
              f"Train Loss={train_loss:.4f} Acc={train_acc:.2f}% | "
              f"Val Loss={val_loss:.4f} Acc={val_acc:.2f}%")

if __name__ == "__main__":
    main()
```

---

## 📚 本节小结

| 步骤 | 关键点 |
|------|--------|
| 数据准备 | 划分数据集、标准化、DataLoader |
| 模型创建 | 合适的架构、BatchNorm、Dropout |
| 损失函数 | 根据任务选择 |
| 优化器 | Adam/AdamW、学习率调度 |
| 训练循环 | train/eval模式、zero_grad |
| 评估 | torch.no_grad()、指标统计 |
| 保存 | state_dict()、最佳模型 |

---

## 🎯 下一步

- [[08-PyTorch模型保存与加载]] - 详细讲解保存和加载
- [[10-PyTorch实战-CIFAR10图像分类]] - 完整项目实战

---

> 💡 **实践建议**：
> 1. 把上面的模板保存下来，以后直接复用
> 2. 用这个模板训练一个自己的任务
> 3. 学会用TensorBoard看训练过程
> 4. 练习解决过拟合/欠拟合问题
