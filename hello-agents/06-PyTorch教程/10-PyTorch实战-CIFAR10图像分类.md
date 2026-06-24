---
title: "PyTorch实战：CIFAR10图像分类"
description: "用CNN完成一个完整的图像分类项目"
tags: [PyTorch, CIFAR10, CNN, Image-Classification, Practice]
---

# 🖼️ PyTorch实战：CIFAR10图像分类

> **把前面学的全部用上，完成一个真实项目！**

---

## 📝 前言

### 项目目标

```
任务：识别图片中的物体（10个类别）
数据：CIFAR10（60000张32x32彩色图片）
类别：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车

目标：测试集准确率 > 85%
```

### 项目结构

```
cifar10_project/
├── data/                  # 数据（自动下载）
├── models/
│   └── cnn.py            # 模型定义
├── utils/
│   └── data_utils.py     # 数据处理
├── train.py              # 训练脚本
├── evaluate.py           # 评估脚本
└── predict.py            # 预测脚本
```

---

## 🔰 1. 数据准备

### 1.1 下载和预处理

```python
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_cifar10_loaders(batch_size=128, num_workers=4):
    """获取CIFAR10数据加载器"""

    # 训练集预处理（包含数据增强）
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),      # 随机裁剪
        transforms.RandomHorizontalFlip(),          # 随机水平翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 颜色抖动
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],  # CIFAR10的均值
            std=[0.2023, 0.1994, 0.2010]    # CIFAR10的标准差
        )
    ])

    # 测试集预处理（不做增强）
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])

    # 加载数据集
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader, train_dataset.classes

# 类别名称
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
```

### 1.2 可视化数据

```python
import matplotlib.pyplot as plt
import numpy as np

def imshow(img, mean, std):
    """显示图片"""
    img = img.numpy().transpose(1, 2, 0)
    img = img * std + mean  # 反标准化
    img = np.clip(img, 0, 1)
    plt.imshow(img)

# 显示一批图片
train_loader, _, classes = get_cifar10_loaders(batch_size=8)
dataiter = iter(train_loader)
images, labels = next(dataiter)

mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]

plt.figure(figsize=(12, 2))
for i in range(8):
    plt.subplot(1, 8, i+1)
    imshow(images[i], mean, std)
    plt.title(classes[labels[i]])
    plt.axis('off')
plt.tight_layout()
plt.savefig("cifar10_samples.png")
plt.show()
```

---

## 🔰 2. 模型定义

### 2.1 简单CNN

```python
import torch.nn as nn

class SimpleCNN(nn.Module):
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
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 测试
model = SimpleCNN()
x = torch.randn(32, 3, 32, 32)
y = model(x)
print(f"输出形状: {y.shape}")  # torch.Size([32, 10])
print(f"参数数量: {sum(p.numel() for p in model.parameters()):,}")
```

### 2.2 更好的CNN（ResNet风格）

```python
class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # 残差连接
        out = torch.relu(out)
        return out

class BetterCNN(nn.Module):
    """带残差连接的CNN"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            ResidualBlock(128)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            ResidualBlock(256)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            ResidualBlock(512)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.classifier(x)
        return x
```

---

## 🔰 3. 训练脚本

### 3.1 完整训练代码

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time

def train():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据
    train_loader, test_loader, classes = get_cifar10_loaders(batch_size=128)
    print(f"训练集: {len(train_loader.dataset)} 张图片")
    print(f"测试集: {len(test_loader.dataset)} 张图片")
    print(f"类别: {classes}")

    # 创建模型
    model = BetterCNN(num_classes=10).to(device)
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # 学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # TensorBoard
    writer = SummaryWriter("runs/cifar10_experiment")

    # 训练循环
    num_epochs = 100
    best_acc = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100. * correct / total

        # 验证
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_loss /= len(test_loader)
        val_acc = 100. * correct / total

        # 更新学习率
        scheduler.step()

        # 记录到TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)

        # 打印
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"Time: {elapsed:.0f}s")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_cifar10.pth")
            print(f"  ✓ 保存最佳模型 (准确率: {best_acc:.2f}%)")

    writer.close()
    print(f"\n训练完成！最佳验证准确率: {best_acc:.2f}%")

if __name__ == "__main__":
    train()
```

---

## 🔰 4. 评估脚本

```python
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    _, test_loader, classes = get_cifar10_loaders(batch_size=128)

    # 加载模型
    model = BetterCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load("best_cifar10.pth"))
    model.eval()

    # 评估
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 准确率
    accuracy = (all_preds == all_labels).mean()
    print(f"测试集准确率: {accuracy:.4f}")

    # 分类报告
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    evaluate()
```

---

## 🔰 5. 预测脚本

```python
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

def predict_image(image_path, model_path="best_cifar10.pth"):
    """预测单张图片"""

    # 类别名称
    CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = BetterCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 预处理
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])

    # 加载图片
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # 预测
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = probs.max(1)

    # 显示结果
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    probs_np = probs.cpu().numpy()[0]
    bars = plt.barh(CLASSES, probs_np)
    plt.xlabel('Probability')
    plt.title(f'Prediction: {CLASSES[predicted.item()]} ({confidence.item():.2%})')
    plt.xlim(0, 1)

    # 高亮预测类别
    bars[predicted.item()].set_color('green')

    plt.tight_layout()
    plt.savefig("prediction.png")
    plt.show()

    print(f"预测类别: {CLASSES[predicted.item()]}")
    print(f"置信度: {confidence.item():.2%}")

# 使用
# predict_image("test_image.jpg")
```

---

## 🔰 6. 训练结果

### 6.1 预期性能

```
SimpleCNN:  ~80-85% 准确率
BetterCNN:  ~85-90% 准确率

训练时间（GPU）: ~30分钟
训练时间（CPU）: ~3小时
```

### 6.2 训练曲线

```
Epoch  1: Train Acc: 45% | Val Acc: 55%
Epoch 10: Train Acc: 75% | Val Acc: 78%
Epoch 20: Train Acc: 82% | Val Acc: 83%
Epoch 50: Train Acc: 88% | Val Acc: 87%
Epoch 100: Train Acc: 92% | Val Acc: 89%
```

---

## 🔰 7. 改进方向

### 7.1 数据增强

```python
# 更强的数据增强
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2023, 0.1994, 0.2010]),
    transforms.RandomErasing(p=0.25),
])
```

### 7.2 模型改进

```python
# 使用更深的网络
# 使用SE模块（通道注意力）
# 使用CBAM（注意力机制）
# 使用EfficientNet等现代架构
```

### 7.3 训练技巧

```python
# 1. 标签平滑
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# 2. Mixup
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b

# 3. Cutout
# 在训练时随机遮挡一部分图片
```

---

## 📚 本节小结

| 阶段 | 内容 |
|------|------|
| 数据 | 下载、预处理、数据增强 |
| 模型 | CNN架构设计 |
| 训练 | 完整训练循环 |
| 评估 | 混淆矩阵、分类报告 |
| 部署 | 预测脚本 |

**关键点**：
1. 数据增强很重要
2. BatchNorm + Residual Connection 有帮助
3. AdamW + CosineAnnealing 是好的组合
4. 监控训练曲线防止过拟合

---

## 🎯 下一步

- [[11-PyTorch实战-迁移学习]] - 用预训练模型
- 尝试更大的数据集（ImageNet子集）

---

> 💡 **实践建议**：
> 1. 跑通整个流程
> 2. 对比SimpleCNN和BetterCNN
> 3. 尝试不同的数据增强
> 4. 用TensorBoard可视化训练过程
