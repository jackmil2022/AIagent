---
title: "PyTorch可视化与调试"
description: "用TensorBoard等工具可视化训练过程和模型"
tags: [PyTorch, Visualization, TensorBoard, Debug]
---

# 📊 PyTorch可视化与调试

> **可视化让模型训练过程一目了然**

---

## 📝 前言

### 为什么需要可视化？

```
训练神经网络时，我们需要知道：
1. Loss在下降吗？
2. 准确率在提升吗？
3. 学习率合适吗？
4. 梯度正常吗？
5. 模型学到了什么特征？

可视化 = 用图表回答这些问题
```

### 工具选择

| 工具 | 特点 | 推荐度 |
|------|------|--------|
| TensorBoard | 官方、功能全 | ⭐⭐⭐⭐⭐ |
| matplotlib | 简单、灵活 | ⭐⭐⭐⭐ |
| wandb | 在线、协作 | ⭐⭐⭐⭐ |

---

## 🔰 1. TensorBoard基础

### 1.1 安装和启动

```bash
# 安装
pip install tensorboard

# 启动（在项目目录下）
tensorboard --logdir=runs

# 浏览器打开
# http://localhost:6006
```

### 1.2 基本使用

```python
from torch.utils.tensorboard import SummaryWriter

# 创建writer
writer = SummaryWriter("runs/experiment_1")

# 记录标量
for epoch in range(100):
    train_loss = 1.0 / (epoch + 1)
    val_loss = 1.2 / (epoch + 1)

    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)

# 关闭writer
writer.close()
```

### 1.3 记录多种指标

```python
writer = SummaryWriter("runs/experiment_1")

for epoch in range(100):
    # 训练指标
    train_loss = train_one_epoch(model, ...)
    train_acc = ...

    # 验证指标
    val_loss = evaluate(model, ...)
    val_acc = ...

    # 记录
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch)
    writer.add_scalar("Accuracy/val", val_acc, epoch)
    writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)

writer.close()
```

---

## 🔰 2. 高级可视化

### 2.1 记录图像

```python
import torchvision

# 记录训练图片
for batch_idx, (images, labels) in enumerate(train_loader):
    # 取前8张图片
    if batch_idx == 0:
        img_grid = torchvision.utils.make_grid(images[:8], nrow=4, normalize=True)
        writer.add_image("Training Images", img_grid, 0)
        break

# 记录模型预测的图片
model.eval()
with torch.no_grad():
    images, labels = next(iter(test_loader))
    outputs = model(images[:8])
    _, preds = torch.max(outputs, 1)

    # 可视化预测结果
    for i in range(8):
        writer.add_image(
            f"Prediction/{i}",
            images[i],
            caption=f"True: {labels[i]}, Pred: {preds[i]}"
        )
```

### 2.2 记录模型图

```python
# 记录模型结构
model = MyModel()
dummy_input = torch.randn(1, 10)
writer.add_graph(model, dummy_input)
# 在TensorBoard的Graphs标签页查看
```

### 2.3 记录权重直方图

```python
# 记录权重分布
for name, param in model.named_parameters():
    writer.add_histogram(f"weights/{name}", param, epoch)
    if param.grad is not None:
        writer.add_histogram(f"gradients/{name}", param.grad, epoch)
```

### 2.4 记录嵌入向量

```python
# 用于可视化高维数据
features = torch.randn(100, 128)  # 100个样本，128维
labels = torch.randint(0, 10, (100,))  # 10个类别

writer.add_embedding(
    features,
    metadata=labels.tolist(),
    label_img=torch.randn(100, 3, 32, 32),  # 可选：对应图片
    tag="Features"
)
# 在Projector标签页查看
```

---

## 🔰 3. 训练过程可视化

### 3.1 完整训练可视化

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset

def train_with_logging():
    # 创建数据
    torch.manual_seed(42)
    X = torch.randn(1000, 10)
    y = (X[:, 0] + X[:, 1] > 0).long()
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 创建模型
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 2)
    )

    # 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # TensorBoard
    writer = SummaryWriter("runs/training")

    # 训练
    for epoch in range(50):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in dataloader:
            output = model(batch_x)
            loss = criterion(output, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

        # 计算指标
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total

        # 记录
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Accuracy/train", accuracy, epoch)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)

        # 记录权重
        for name, param in model.named_parameters():
            writer.add_histogram(f"weights/{name}", param, epoch)

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")

    writer.close()

train_with_logging()
```

---

## 🔰 4. Matplotlib可视化

### 4.1 训练曲线

```python
import matplotlib.pyplot as plt

def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss曲线
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True)

    # 准确率曲线
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Curves')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()

# 使用
train_losses = [1.0, 0.8, 0.6, 0.5, 0.4]
val_losses = [1.1, 0.9, 0.7, 0.6, 0.55]
train_accs = [60, 70, 75, 80, 85]
val_accs = [58, 68, 73, 78, 82]

plot_training_curves(train_losses, val_losses, train_accs, val_accs)
```

### 4.2 混淆矩阵

```python
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, classes):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

# 使用
y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
y_pred = [0, 1, 2, 0, 2, 1, 0, 1, 2]
classes = ['cat', 'dog', 'bird']

plot_confusion_matrix(y_true, y_pred, classes)
```

### 4.3 特征可视化

```python
def visualize_filters(model):
    """可视化卷积层的滤波器"""
    # 获取第一个卷积层的权重
    first_conv = None
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            first_conv = layer
            break

    if first_conv is None:
        print("没有找到卷积层")
        return

    weights = first_conv.weight.data.cpu()

    # 绘制滤波器
    num_filters = min(32, weights.shape[0])
    fig, axes = plt.subplots(8, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            ax.imshow(weights[i, 0], cmap='gray')
        ax.axis('off')

    plt.suptitle('Convolutional Filters')
    plt.tight_layout()
    plt.savefig("filters.png")
    plt.show()
```

---

## 🔰 5. 模型调试技巧

### 5.1 梯度监控

```python
def monitor_gradients(model, writer, epoch):
    """监控梯度"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            # 梯度均值
            writer.add_scalar(f"gradients/{name}/mean",
                            param.grad.mean(), epoch)
            # 梯度标准差
            writer.add_scalar(f"gradients/{name}/std",
                            param.grad.std(), epoch)
            # 梯度最大值
            writer.add_scalar(f"gradients/{name}/max",
                            param.grad.abs().max(), epoch)

    # 梯度消失检测
    for name, param in model.named_parameters():
        if param.grad is not None:
            if param.grad.abs().mean() < 1e-7:
                print(f"⚠️ 警告：{name} 梯度可能消失")
```

### 5.2 模型诊断

```python
def diagnose_model(model, dataloader, criterion, device):
    """模型诊断"""
    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)

            _, preds = output.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_logits.extend(output.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)

    # 准确率
    accuracy = (all_preds == all_labels).mean()
    print(f"准确率: {accuracy:.4f}")

    # 每个类别的准确率
    for i in range(num_classes):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == i).mean()
            print(f"类别 {i}: {class_acc:.4f} ({mask.sum()} 个样本)")

    # 预测分布
    print(f"\n预测分布:")
    for i in range(num_classes):
        count = (all_preds == i).sum()
        print(f"  类别 {i}: {count} 个")

    return all_preds, all_labels, all_logits
```

### 5.3 过拟合检测

```python
def detect_overfitting(train_losses, val_losses, threshold=0.1):
    """检测过拟合"""
    if len(train_losses) < 5 or len(val_losses) < 5:
        return False

    # 计算最近5个epoch的gap
    train_gap = train_losses[-5:]
    val_gap = val_losses[-5:]
    gap = np.mean(val_gap) - np.mean(train_gap)

    if gap > threshold:
        print(f"⚠️ 可能过拟合！Val比Train高 {gap:.4f}")
        return True
    return False
```

---

## 🔰 6. Wandb（可选）

### 6.1 基本使用

```python
import wandb

# 初始化
wandb.init(project="my-project", name="experiment-1")

# 记录指标
for epoch in range(100):
    train_loss = train_one_epoch(model, ...)
    val_loss = evaluate(model, ...)

    wandb.log({
        "train/loss": train_loss,
        "val/loss": val_loss,
        "epoch": epoch
    })

# 记录模型
wandb.save("model.pth")

# 结束
wandb.finish()
```

### 6.2 对比TensorBoard

```
TensorBoard：
- 本地运行
- 免费
- 功能全面

wandb：
- 在线协作
- 实验对比
- 更好的可视化
- 免费版有限制
```

---

## 📚 本节小结

| 工具 | 用途 | 使用场景 |
|------|------|----------|
| TensorBoard | 训练可视化 | 记录loss、acc、图像 |
| matplotlib | 自定义图表 | 绘制混淆矩阵、曲线 |
| wandb | 在线协作 | 多人协作、实验对比 |

**调试技巧**：
1. 监控梯度防止消失/爆炸
2. 检测过拟合
3. 分析每个类别的表现
4. 可视化模型学习的特征

---

## 🎯 下一步

- [[10-PyTorch实战-CIFAR10图像分类]] - 完整项目实战
- [[11-PyTorch实战-迁移学习]] - 站在巨人肩膀上

---

> 💡 **实践建议**：
> 1. 在训练中加入TensorBoard日志
> 2. 用matplotlib绘制混淆矩阵
> 3. 练习监控梯度
> 4. 尝试用wandb做实验对比
