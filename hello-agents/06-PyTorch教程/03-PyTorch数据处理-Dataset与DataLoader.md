---
title: "PyTorch数据处理：Dataset与DataLoader"
description: "学习如何加载、处理和批量提供数据给模型"
tags: [PyTorch, Dataset, DataLoader, Data]
---

# 📦 PyTorch数据处理：Dataset与DataLoader

> **数据是模型的"食材"，Dataset和DataLoader是你的"厨房助手"**

---

## 📝 前言

### 为什么需要Dataset和DataLoader？

训练神经网络时，我们需要：
1. **读取数据**：从文件/数据库加载数据
2. **预处理**：转换格式、增强、标准化
3. **批量加载**：每次取一小批（batch）数据
4. **打乱顺序**：避免模型学习到数据的顺序
5. **多进程加速**：用多个CPU核心并行加载

PyTorch提供了：
- **Dataset**：定义"如何读取一个样本"
- **DataLoader**：定义"如何批量加载、打乱、多进程"

### 类比理解

```
Dataset = 菜谱（告诉你如何准备一道菜）
DataLoader = 厨师（按照菜谱批量做菜，还能多人同时做）
```

---

## 🔰 1. Dataset基础

### 1.1 Dataset是什么？

Dataset是一个**抽象类**，你需要继承它并实现两个方法：

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self):
        # 初始化：加载数据、预处理
        pass

    def __getitem__(self, idx):
        # 返回第idx个样本
        pass

    def __len__(self):
        # 返回数据集大小
        pass
```

### 1.2 最简单的例子

```python
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)

# 使用
data = [1, 2, 3, 4, 5]
labels = [0, 1, 0, 1, 0]

dataset = SimpleDataset(data, labels)
print(len(dataset))  # 5
print(dataset[0])    # (1, 0)
print(dataset[2])    # (3, 0)
```

### 1.3 处理图片数据

```python
from torch.utils.data import Dataset
from PIL import Image
import os

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = os.listdir(image_dir)

    def __getitem__(self, idx):
        # 读取图片
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        # 预处理
        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.image_files)
```

### 1.4 处理带标签的数据

```python
from torch.utils.data import Dataset
import pandas as pd

class CSVDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform

    def __getitem__(self, idx):
        # 获取一行数据
        row = self.data.iloc[idx]

        # 分离特征和标签
        features = row.drop('label').values.astype('float32')
        label = row['label']

        # 转换为Tensor
        import torch
        features = torch.tensor(features)
        label = torch.tensor(label, dtype=torch.long)

        # 预处理
        if self.transform:
            features = self.transform(features)

        return features, label

    def __len__(self):
        return len(self.data)
```

---

## 🔰 2. DataLoader基础

### 2.1 DataLoader是什么？

DataLoader负责：
- **批量加载**：每次取batch_size个样本
- **打乱顺序**：shuffle=True
- **多进程加载**：num_workers > 0
- **自动整理**：把多个样本拼成一个batch

### 2.2 基本用法

```python
from torch.utils.data import DataLoader, Dataset

# 创建Dataset
class SimpleDataset(Dataset):
    def __init__(self, size=100):
        self.data = list(range(size))
        self.labels = [i % 2 for i in range(size)]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)

dataset = SimpleDataset(100)

# 创建DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=16,    # 每批16个
    shuffle=True,     # 打乱顺序
    num_workers=0     # 不用多进程（Windows建议设为0）
)

# 使用
for batch_data, batch_labels in dataloader:
    print(f"数据形状: {batch_data.shape}")   # torch.Size([16])
    print(f"标签形状: {batch_labels.shape}")  # torch.Size([16])
    break  # 只看第一个batch
```

### 2.3 DataLoader参数详解

```python
dataloader = DataLoader(
    dataset,
    batch_size=32,      # 批大小
    shuffle=True,       # 每个epoch打乱
    num_workers=4,      # 4个进程加载数据
    pin_memory=True,    # 锁页内存，GPU训练更快
    drop_last=True,     # 丢弃最后不完整的batch
    timeout=30,         # 超时时间（秒）
    prefetch_factor=2   # 预取因子
)
```

---

## 🔰 3. 数据预处理

### 3.1 transforms — 图片预处理

```python
from torchvision import transforms

# 定义预处理流程
transform = transforms.Compose([
    transforms.Resize((224, 224)),      # 缩放到224x224
    transforms.ToTensor(),              # 转为Tensor，归一化到[0,1]
    transforms.Normalize(               # 标准化
        mean=[0.485, 0.456, 0.406],     # ImageNet均值
        std=[0.229, 0.224, 0.225]       # ImageNet标准差
    )
])

# 在Dataset中使用
class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)
```

### 3.2 常用transforms

```python
from torchvision import transforms

# 图片预处理
transform = transforms.Compose([
    # 缩放
    transforms.Resize(256),
    transforms.CenterCrop(224),

    # 翻转
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.RandomVerticalFlip(p=0.5),    # 随机垂直翻转

    # 旋转
    transforms.RandomRotation(30),  # 随机旋转±30度

    # 颜色变换
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),

    # 转为Tensor
    transforms.ToTensor(),

    # 标准化
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),

    # 随机擦除（数据增强）
    transforms.RandomErasing(p=0.5),
])
```

### 3.3 数据增强的意义

```python
# 为什么要做数据增强？

# 原始数据：1000张猫的图片
# 增强后：相当于5000张（翻转、旋转、调色...）

# 好处：
# 1. 防止过拟合
# 2. 模型更鲁棒
# 3. 小数据集也能训练好
```

---

## 🔰 4. 自定义数据集实战

### 4.1 分类任务数据集

```python
from torch.utils.data import Dataset
from PIL import Image
import os

class ClassificationDataset(Dataset):
    """图片分类数据集"""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: 数据目录，格式为：
                root_dir/
                    class1/
                        img1.jpg
                        img2.jpg
                    class2/
                        img3.jpg
            transform: 预处理
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # 收集所有图片路径和标签
        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                if img_name.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(cls_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[cls]))

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # 读取图片
        image = Image.open(img_path).convert('RGB')

        # 预处理
        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.samples)

    def get_class_name(self, idx):
        """根据索引获取类名"""
        return self.classes[idx]
```

### 4.2 回归任务数据集

```python
from torch.utils.data import Dataset
import torch

class RegressionDataset(Dataset):
    """回归任务数据集"""

    def __init__(self, features, labels):
        """
        Args:
            features: 特征数据 (numpy array or list)
            labels: 标签数据 (numpy array or list)
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def __len__(self):
        return len(self.features)

# 使用
import numpy as np

# 生成模拟数据
np.random.seed(42)
X = np.random.randn(1000, 5)  # 1000个样本，5个特征
y = X @ np.array([1, 2, 3, 4, 5]) + np.random.randn(1000) * 0.1

dataset = RegressionDataset(X, y)
print(f"数据集大小: {len(dataset)}")
print(f"样本形状: {dataset[0][0].shape}")  # torch.Size([5])
```

### 4.3 文本数据集

```python
from torch.utils.data import Dataset
import torch

class TextDataset(Dataset):
    """文本分类数据集"""

    def __init__(self, texts, labels, vocab, max_len=100):
        """
        Args:
            texts: 文本列表
            labels: 标签列表
            vocab: 词表 {word: index}
            max_len: 最大序列长度
        """
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 简单的分词和编码
        tokens = text.lower().split()
        indices = [self.vocab.get(token, 0) for token in tokens]

        # 填充或截断
        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]

        return torch.tensor(indices, dtype=torch.long), label

    def __len__(self):
        return len(self.texts)

# 使用
texts = ["I love AI", "PyTorch is great", "Hello world"]
labels = [1, 1, 0]
vocab = {"i": 1, "love": 2, "ai": 3, "pytorch": 4, "is": 5, "great": 6, "hello": 7, "world": 8}

dataset = TextDataset(texts, labels, vocab, max_len=5)
print(dataset[0])  # (tensor([1, 2, 3, 0, 0]), 1)
```

---

## 🔰 5. 内置数据集

### 5.1 torchvision.datasets

```python
from torchvision import datasets, transforms

# MNIST手写数字
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

# CIFAR10图片分类
train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
)

# ImageFolder（通用图片分类）
train_dataset = datasets.ImageFolder(
    root='./data/train',
    transform=transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
)
```

### 5.2 预览数据

```python
import matplotlib.pyplot as plt

# 取一个样本
image, label = train_dataset[0]

# 显示图片
if image.dim() == 3:  # (C, H, W)
    image = image.permute(1, 2, 0)  # → (H, W, C)

plt.imshow(image)
plt.title(f"Label: {label}")
plt.show()
```

---

## 🔰 6. DataLoader实战

### 6.1 完整训练循环

```python
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn

# 1. 准备数据
class MyDataset(Dataset):
    def __init__(self, size=1000):
        self.x = torch.randn(size, 10)
        self.y = torch.randint(0, 2, (size,))

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

dataset = MyDataset(1000)

# 2. 创建DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2
)

# 3. 定义模型
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)

# 4. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. 训练循环
for epoch in range(10):
    model.train()
    total_loss = 0

    for batch_x, batch_y in train_loader:
        # 前向传播
        output = model(batch_x)
        loss = criterion(output, batch_y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/10, Loss: {avg_loss:.4f}")
```

### 6.2 训练集/验证集/测试集划分

```python
from torch.utils.data import random_split, DataLoader

# 假设我们有一个完整的数据集
full_dataset = MyDataset(1000)

# 划分：80% 训练，10% 验证，10% 测试
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size]
)

print(f"训练集: {len(train_dataset)}")
print(f"验证集: {len(val_dataset)}")
print(f"测试集: {len(test_dataset)}")

# 创建各自的DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

### 6.3 多worker注意事项

```python
# Windows用户注意！
# num_workers > 0 需要加 if __name__ == '__main__': 保护

import torch
from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    def __init__(self):
        self.data = list(range(100))

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

def main():
    dataset = MyDataset()

    # Windows建议num_workers=0
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0  # Windows设为0避免问题
    )

    for batch in dataloader:
        print(batch)
        break

if __name__ == '__main__':
    main()
```

---

## 🔰 7. 高级技巧

### 7.1 自定义collate_fn

```python
from torch.utils.data import DataLoader, Dataset
import torch

class VariableLengthDataset(Dataset):
    """变长序列数据集"""

    def __init__(self):
        self.sequences = [
            [1, 2, 3],
            [1, 2, 3, 4, 5],
            [1],
            [1, 2, 3, 4, 5, 6, 7],
        ]
        self.labels = [0, 1, 0, 1]

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

    def __len__(self):
        return len(self.sequences)

# 自定义collate函数
def collate_fn(batch):
    sequences, labels = zip(*batch)

    # 填充到相同长度
    max_len = max(len(seq) for seq in sequences)
    padded_seqs = [seq + [0] * (max_len - len(seq)) for seq in sequences]

    return torch.tensor(padded_seqs), torch.tensor(labels)

dataset = VariableLengthDataset()
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn
)

for batch_x, batch_y in dataloader:
    print(f"形状: {batch_x.shape}")
    print(batch_x)
    break
```

### 7.2 IterableDataset（流式数据）

```python
from torch.utils.data import IterableDataset, DataLoader

class StreamDataset(IterableDataset):
    """流式数据集，适合大数据"""

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        # 返回一个迭代器
        return iter(self.data_source)

# 使用
def data_generator():
    for i in range(100):
        yield i, i * 2

dataset = StreamDataset(data_generator())
dataloader = DataLoader(dataset, batch_size=16)

for batch_x, batch_y in dataloader:
    print(batch_x, batch_y)
    break
```

### 7.3 数据集信息查看

```python
def dataset_info(dataset):
    """打印数据集信息"""
    print(f"数据集大小: {len(dataset)}")
    print(f"样本类型: {type(dataset[0])}")

    if isinstance(dataset[0], tuple):
        sample = dataset[0]
        for i, item in enumerate(sample):
            if hasattr(item, 'shape'):
                print(f"  第{i}个元素形状: {item.shape}")
            elif hasattr(item, '__len__'):
                print(f"  第{i}个元素长度: {len(item)}")
            else:
                print(f"  第{i}个元素值: {item}")

# 使用
dataset = MyDataset(100)
dataset_info(dataset)
# 数据集大小: 100
# 样本类型: <class 'tuple'>
#   第0个元素形状: torch.Size([10])
#   第1个元素值: 0
```

---

## 📚 本节小结

| 概念 | 说明 |
|------|------|
| Dataset | 数据集抽象类，定义如何读取样本 |
| DataLoader | 数据加载器，批量加载、打乱、多进程 |
| __getitem__ | 返回第idx个样本 |
| __len__ | 返回数据集大小 |
| transforms | 数据预处理和增强 |
| batch_size | 批大小 |
| shuffle | 是否打乱 |
| num_workers | 多进程加载 |

---

## 🎯 下一步

- [[04-PyTorch模型构建-nn.Module]] - 学习如何搭建神经网络
- [[07-PyTorch完整训练流程]] - 完整的训练流程

---

> 💡 **实践建议**：
> 1. 创建一个自己的Dataset，加载你手头的数据
> 2. 尝试不同的数据增强方法
> 3. 练习使用DataLoader的各个参数
> 4. 运行完整训练循环，观察loss变化
