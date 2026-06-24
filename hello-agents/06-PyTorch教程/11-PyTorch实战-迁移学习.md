---
title: "PyTorch实战：迁移学习"
description: "站在巨人肩膀上，用预训练模型快速解决问题"
tags: [PyTorch, Transfer-Learning, Fine-tune, Pre-trained]
---

# 🚀 PyTorch实战：迁移学习

> **站在巨人肩膀上，用别人训练好的模型解决你的问题**

---

## 📝 前言

### 什么是迁移学习？

```
从头训练一个模型：
- 需要大量数据（几万张图片）
- 需要很长时间（几天到几周）
- 需要很多计算资源（多GPU）

迁移学习：
- 用别人在大数据集上训练好的模型
- 只训练最后几层
- 几分钟到几小时就能用
- 少量数据也能达到好效果
```

### 类比理解

```
人类学习：
- 从小学开始学字母、拼音
- 学会了基础概念
- 后来学英语就很快（因为有基础）

迁移学习：
- 模型在ImageNet（1000类，100万张图片）上学会了通用特征
- 迁移到你的任务（比如猫狗分类）
- 不需要从头学什么是"边缘"、"纹理"、"形状"
```

### 什么时候用迁移学习？

| 场景 | 是否适合 |
|------|----------|
| 数据少（< 1000张） | ✅ 非常适合 |
| 数据多（> 10000张） | ⚠️ 可以试试，但可能从头训练更好 |
| 相似任务（都是图片分类） | ✅ 非常适合 |
| 不同任务（图片→文本） | ❌ 不适合 |

---

## 🔰 1. 迁移学习原理

### 1.1 预训练模型结构

```
预训练模型（如ResNet）：
┌─────────────────────┐
│  输入层             │
├─────────────────────┤
│  特征提取层         │  ← 学到通用特征（边缘、纹理、形状）
│  (很多卷积层)       │
├─────────────────────┤
│  分类层             │  ← 原本是1000类（ImageNet）
│  (全连接层)         │
└─────────────────────┘

迁移学习：
┌─────────────────────┐
│  输入层             │
├─────────────────────┤
│  特征提取层         │  ← 冻结，不训练
│  (保持不变)         │
├─────────────────────┤
│  新的分类层         │  ← 替换成你的类别数（如2类）
│  (需要训练)         │
└─────────────────────┘
```

### 1.2 两种迁移学习方式

```
方式1：特征提取（Feature Extraction）
- 冻结所有预训练层
- 只训练新加的分类层
- 优点：快、省显存
- 缺点：效果可能不是最好

方式2：微调（Fine-tuning）
- 解冻部分或全部预训练层
- 用小学习率训练整个网络
- 优点：效果更好
- 缺点：慢、需要小心调参
```

---

## 🔰 2. 加载预训练模型

### 2.1 PyTorch内置模型

```python
import torch
import torch.nn as nn
from torchvision import models

# 加载预训练的ResNet18
model = models.resnet18(pretrained=True)

# 查看模型结构
print(model)
```

### 2.2 常用预训练模型

```python
# 轻量级（适合移动端）
model = models.mobilenet_v2(pretrained=True)  # 参数少
model = models.shufflenet_v2_x0_5(pretrained=True)

# 中等规模
model = models.resnet18(pretrained=True)    # 11M参数
model = models.resnet34(pretrained=True)    # 21M参数

# 大规模
model = models.resnet50(pretrained=True)    # 25M参数
model = models.efficientnet_b0(pretrained=True)

# 查看参数量
for name, model_fn in [
    ("MobileNetV2", models.mobilenet_v2),
    ("ResNet18", models.resnet18),
    ("ResNet50", models.resnet50),
]:
    m = model_fn(pretrained=False)
    params = sum(p.numel() for p in m.parameters())
    print(f"{name}: {params:,} 参数")
```

### 2.3 下载位置

```python
# 预训练权重默认下载到 ~/.cache/torch/
# 可以自定义位置
model = models.resnet18(pretrained=True)

# 或者手动下载
import torchvision
torchvision.datasets.utils.download_url(
    "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    root=".",
    filename="resnet18.pth"
)
```

---

## 🔰 3. 特征提取（Feature Extraction）

### 3.1 冻结所有层，只训练分类层

```python
import torch
import torch.nn as nn
from torchvision import models

def create_feature_extractor(num_classes=10):
    """创建特征提取器"""

    # 加载预训练模型
    model = models.resnet18(pretrained=True)

    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 替换分类层（原先是1000类）
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model

# 创建模型
model = create_feature_extractor(num_classes=10)

# 只有分类层有梯度
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"可训练参数: {trainable_params:,} / {total_params:,}")
print(f"冻结比例: {(1 - trainable_params/total_params)*100:.1f}%")
```

### 3.2 训练

```python
import torch.optim as optim

# 创建模型
model = create_feature_extractor(num_classes=10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 损失函数和优化器
# 只优化分类层的参数
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  # 只传分类层参数

# 训练
for epoch in range(20):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 🔰 4. 微调（Fine-tuning）

### 4.1 分阶段微调

```python
def create_finetuned_model(num_classes=10, finetune_layers=2):
    """创建微调模型"""

    model = models.resnet18(pretrained=True)

    # 冻结所有层
    for param in model.parameters():
        param.requires_grad = False

    # 解冻最后几层
    # ResNet18的结构：layer1, layer2, layer3, layer4, fc
    layers_to_finetune = []
    if finetune_layers >= 1:
        layers_to_finetune.extend(model.layer4.parameters())
    if finetune_layers >= 2:
        layers_to_finetune.extend(model.fc.parameters())

    for param in layers_to_finetune:
        param.requires_grad = True

    # 替换分类层
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model, layers_to_finetune

# 创建模型
model, trainable_params = create_finetuned_model(num_classes=10, finetune_layers=2)
```

### 4.2 差异学习率

```python
# 不同层用不同的学习率
# 特征层：小学习率（已经很好了）
# 分类层：大学习率（需要重新学）

def create_model_with_diff_lr(num_classes=10):
    model = models.resnet18(pretrained=True)

    # 分组参数
    params = [
        # 特征层：小学习率
        {'params': model.layer4.parameters(), 'lr': 1e-5},
        # 分类层：大学习率
        {'params': model.fc.parameters(), 'lr': 1e-3},
    ]

    # 冻结其他层
    for name, param in model.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False

    # 替换分类层
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model, params

# 使用
model, params = create_model_with_diff_lr(num_classes=10)
optimizer = optim.Adam(params, lr=1e-3)  # 初始学习率会被覆盖
```

### 4.3 渐进式解冻

```python
def progressive_unfreezing(model, epoch, total_epochs):
    """渐进式解冻"""
    # 前30%：只训练分类层
    # 中间40%：解冻layer4
    # 最后30%：解冻所有层

    unfreeze_epoch_1 = int(total_epochs * 0.3)
    unfreeze_epoch_2 = int(total_epochs * 0.7)

    if epoch < unfreeze_epoch_1:
        # 只训练分类层
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    elif epoch < unfreeze_epoch_2:
        # 解冻layer4
        for param in model.layer4.parameters():
            param.requires_grad = True

    else:
        # 解冻所有层
        for param in model.parameters():
            param.requires_grad = True
```

---

## 🔰 5. 完整迁移学习模板

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import time

def transfer_learning_pipeline(
    data_dir,
    num_classes=10,
    num_epochs=30,
    batch_size=32,
    learning_rate=0.001
):
    """完整的迁移学习流程"""

    # ==================== 1. 数据 ====================
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225])
        ]),
    }

    # 假设数据目录结构：
    # data_dir/
    #   train/
    #     class1/
    #     class2/
    #   val/
    #     class1/
    #     class2/

    image_datasets = {
        x: datasets.ImageFolder(f"{data_dir}/{x}", data_transforms[x])
        for x in ['train', 'val']
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
        for x in ['train', 'val']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    print(f"类别: {class_names}")
    print(f"训练集: {dataset_sizes['train']} 张")
    print(f"验证集: {dataset_sizes['val']} 张")

    # ==================== 2. 模型 ====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载预训练模型
    model = models.resnet18(pretrained=True)

    # 冻结所有层
    for param in model.parameters():
        param.requires_grad = False

    # 替换分类层
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    model = model.to(device)

    # ==================== 3. 优化器 ====================
    # 只优化分类层
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # ==================== 4. 训练 ====================
    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        best_model_wts = model.state_dict()
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # 每个epoch有训练和验证两个阶段
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()

            print()

        time_elapsed = time.time() - since
        print(f'训练完成 in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'最佳准确率: {best_acc:.4f}')

        model.load_state_dict(best_model_wts)
        return model

    model = train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs)

    return model

# 使用
# model = transfer_learning_pipeline("./data/flowers", num_classes=5)
```

---

## 🔰 6. 实际案例：猫狗分类

### 6.1 数据准备

```python
# 假设数据目录结构：
# data/
#   train/
#     dogs/
#       dog001.jpg
#       dog002.jpg
#     cats/
#       cat001.jpg
#       cat002.jpg
#   val/
#     dogs/
#     cats/
```

### 6.2 完整代码

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

# 数据
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

train_dataset = datasets.ImageFolder("data/train", data_transforms['train'])
val_dataset = datasets.ImageFolder("data/val", data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=True)

# 冻结所有层
for param in model.parameters():
    param.requires_grad = False

# 替换分类层（2类：猫和狗）
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

model = model.to(device)

# 优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# 训练
for epoch in range(20):
    # 训练
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100. * correct / total

    # 验证
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_acc = 100. * correct / total

    print(f"Epoch {epoch+1}: "
          f"Train Loss={train_loss/len(train_loader):.4f} Acc={train_acc:.2f}% | "
          f"Val Loss={val_loss/len(val_loader):.4f} Acc={val_acc:.2f}%")
```

---

## 🔰 7. 常见问题

### 7.1 过拟合

```python
# 数据增强更强
# 减少训练层
# 增加Dropout
# 使用Label Smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

### 7.2 训练太慢

```python
# 使用更小的模型
model = models.mobilenet_v2(pretrained=True)

# 减少batch_size
# 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for inputs, labels in train_loader:
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 7.3 效果不好

```python
# 1. 检查数据是否正确
# 2. 尝试不同的预训练模型
# 3. 调整学习率
# 4. 解冻更多层
# 5. 训练更多epoch
```

---

## 📚 本节小结

| 方式 | 特点 | 适用场景 |
|------|------|----------|
| 特征提取 | 只训练分类层 | 数据少、快速 |
| 微调 | 解冻部分层 | 数据中等 |
| 完全微调 | 解冻所有层 | 数据多 |

**关键点**：
1. 预训练模型学到的通用特征很有用
2. 学习率要小（1e-4 ~ 1e-5）
3. 数据增强很重要
4. 从简单开始，逐步复杂

---

## 🎯 下一步

- 尝试不同的预训练模型（ResNet、EfficientNet、MobileNet）
- 用迁移学习解决你自己的问题
- 探索目标检测、语义分割等任务

---

> 💡 **实践建议**：
> 1. 用迁移学习训练一个猫狗分类器
> 2. 对比特征提取和微调的效果
> 3. 尝试不同的预训练模型
> 4. 把训练好的模型部署成API
