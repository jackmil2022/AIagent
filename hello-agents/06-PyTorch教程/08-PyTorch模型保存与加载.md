---
title: "PyTorch模型保存与加载"
description: "学习如何保存训练好的模型并在以后使用"
tags: [PyTorch, Save, Load, Model, Deployment]
---

# 💾 PyTorch模型保存与加载

> **训练好的模型要保存下来，以后才能用**

---

## 📝 前言

### 为什么需要保存模型？

```
训练一个模型可能需要几小时甚至几天
不可能每次都重新训练

所以需要：
1. 保存训练好的模型
2. 以后直接加载使用
3. 或者继续训练
```

### 保存什么？

```
需要保存：
1. 模型参数（权重和偏置）
2. 优化器状态（可选，用于继续训练）
3. 训练信息（epoch、loss等，可选）

不需要保存：
- 模型代码（代码要保存在版本控制中）
- 训练数据（数据要单独管理）
```

---

## 🔰 1. 保存和加载模型参数（推荐）

### 1.1 保存state_dict

```python
import torch
import torch.nn as nn

# 创建模型
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)

# 保存模型参数
torch.save(model.state_dict(), "model.pth")

# state_dict是什么？
print(model.state_dict().keys())
# odict_keys(['0.weight', '0.bias', '2.weight', '2.bias'])
```

### 1.2 加载state_dict

```python
# 加载模型参数
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)

# 加载参数
model.load_state_dict(torch.load("model.pth"))
model.eval()  # 设为评估模式

print("模型加载成功！")
```

### 1.3 完整示例

```python
import torch
import torch.nn as nn

# 定义模型结构（必须和保存时一致）
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 训练模型
model = MyModel()
# ... 训练代码 ...

# 保存
torch.save(model.state_dict(), "my_model.pth")

# 加载（在另一个文件中）
loaded_model = MyModel()
loaded_model.load_state_dict(torch.load("my_model.pth"))
loaded_model.eval()

# 使用
x = torch.randn(1, 10)
output = loaded_model(x)
print(output)
```

---

## 🔰 2. 保存和加载整个模型

### 2.1 保存整个模型

```python
# 保存整个模型（包括结构和参数）
torch.save(model, "whole_model.pth")
```

### 2.2 加载整个模型

```python
# 加载整个模型
loaded_model = torch.load("whole_model.pth")
loaded_model.eval()
```

### 2.3 对比

```python
# state_dict方式（推荐）
# 优点：
# - 文件更小
# - 更灵活（可以加载到不同结构的模型）
# - PyTorch官方推荐
# 缺点：
# - 需要定义模型结构

# 整个模型方式
# 优点：
# - 使用简单
# - 不需要定义模型结构
# 缺点：
# - 文件更大
# - 依赖代码结构
# - 不推荐用于部署
```

---

## 🔰 3. 保存训练检查点

### 3.1 保存完整检查点

```python
def save_checkpoint(model, optimizer, epoch, loss, path="checkpoint.pth"):
    """保存训练检查点"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_checkpoint(model, optimizer, path="checkpoint.pth"):
    """加载训练检查点"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss
```

### 3.2 训练中保存

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = MyModel()
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

best_loss = float('inf')

for epoch in range(100):
    # 训练...
    train_loss = train_one_epoch(model, ...)

    # 验证...
    val_loss = evaluate(model, ...)

    # 保存最佳模型
    if val_loss < best_loss:
        best_loss = val_loss
        save_checkpoint(model, optimizer, epoch, val_loss, "best_model.pth")

    # 保存最新模型
    save_checkpoint(model, optimizer, epoch, val_loss, "latest_model.pth")

    # 每10个epoch保存一次
    if (epoch + 1) % 10 == 0:
        save_checkpoint(model, optimizer, epoch, val_loss, f"checkpoint_epoch{epoch+1}.pth")
```

### 3.3 继续训练

```python
# 从检查点继续训练
model = MyModel()
optimizer = optim.Adam(model.parameters())

start_epoch, start_loss = load_checkpoint(model, optimizer, "latest_model.pth")
print(f"从 epoch {start_epoch} 继续训练")

for epoch in range(start_epoch, 100):
    # 继续训练...
    pass
```

---

## 🔰 4. GPU和CPU模型转换

### 4.1 GPU保存，CPU加载

```python
# 在GPU上保存
device = torch.device("cuda")
model = MyModel().to(device)

# 训练...
torch.save(model.state_dict(), "model.pth")

# 在CPU上加载
device = torch.device("cpu")
model = MyModel()
model.load_state_dict(torch.load("model.pth", map_location=device))
```

### 4.2 自动检测设备

```python
def load_model(model, path, device=None):
    """自动检测设备加载模型"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

# 使用
model = load_model(MyModel(), "model.pth")
```

### 4.3 映射到CPU

```python
# 无论原设备是什么，都加载到CPU
model.load_state_dict(torch.load("model.pth", map_location="cpu"))

# 或者映射到指定设备
model.load_state_dict(torch.load("model.pth", map_location="cuda:1"))
```

---

## 🔰 5. 模型导出（部署）

### 5.1 导出为ONNX

```python
# ONNX是通用模型格式，可以被多种框架使用
model = MyModel()
model.eval()

# 创建示例输入
dummy_input = torch.randn(1, 10)

# 导出ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print("ONNX模型已导出！")
```

### 5.2 TorchScript（推荐）

```python
# TorchScript是PyTorch的序列化格式
model = MyModel()
model.eval()

# 方法1：追踪（Trace）
dummy_input = torch.randn(1, 10)
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("model_traced.pt")

# 方法2：脚本（Script）
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")

# 加载TorchScript模型
loaded_model = torch.jit.load("model_traced.pt")
output = loaded_model(dummy_input)
```

### 5.3 TorchScript vs ONNX

```
TorchScript：
- PyTorch原生格式
- 只能用PyTorch加载
- 部署PyTorch服务

ONNX：
- 通用格式
- 可以被TensorRT、OpenVINO等优化
- 跨框架部署
```

---

## 🔰 6. 最佳实践

### 6.1 推荐的保存方式

```python
def save_model(model, optimizer, epoch, path):
    """推荐的保存方式"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def load_model(model, optimizer, path):
    """推荐的加载方式"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('epoch', 0)
```

### 6.2 文件组织

```python
# 推荐的文件组织
checkpoints/
├── best_model.pth          # 最佳模型
├── latest_model.pth        # 最新模型
├── checkpoint_epoch10.pth  # 定期保存
├── checkpoint_epoch20.pth
└── ...

# 文件命名
f"checkpoint_epoch{epoch:03d}_loss{loss:.4f}.pth"
```

### 6.3 版本控制

```python
import os
import time

def save_with_version(model, optimizer, epoch, loss, base_dir="checkpoints"):
    """带版本号的保存"""
    os.makedirs(base_dir, exist_ok=True)

    # 用时间戳作为版本号
    version = time.strftime("%Y%m%d_%H%M%S")
    filename = f"model_v{version}.pth"

    path = os.path.join(base_dir, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'version': version,
    }, path)

    print(f"模型已保存: {path}")
    return path
```

---

## 🔰 7. 常见问题

### 7.1 键名不匹配

```python
# 错误：键名不匹配
# RuntimeError: Error(s) in loading state_dict for MyModel:
#     size mismatch for fc1.weight: copying a param with shape torch.Size([64, 10]) from checkpoint, the shape in current model is torch.Size([64, 20]).

# 解决方案1：严格=False
model.load_state_dict(torch.load("model.pth"), strict=False)

# 解决方案2：打印键名检查
checkpoint = torch.load("model.pth")
print(checkpoint.keys())
print(model.state_dict().keys())
```

### 7.2 参数不完全匹配

```python
# 只加载匹配的参数
model_dict = model.state_dict()
checkpoint_dict = torch.load("model.pth")

# 过滤
pretrained_dict = {k: v for k, v in checkpoint_dict.items()
                   if k in model_dict and v.shape == model_dict[k].shape}

# 更新
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
```

### 7.3 pickle安全警告

```python
# 警告：torch.load uses pickle
# Warning: This may be a security risk

# 解决方案：只加载已知来源的模型
import pickle

def safe_load(path):
    """安全加载"""
    with open(path, 'rb') as f:
        return torch.load(f, weights_only=True)  # PyTorch 2.0+
```

---

## 📚 本节小结

| 方法 | 用途 | 推荐度 |
|------|------|--------|
| state_dict | 保存/加载参数 | ⭐⭐⭐⭐⭐ |
| 整个模型 | 简单保存 | ⭐⭐⭐ |
| checkpoint | 保存训练状态 | ⭐⭐⭐⭐⭐ |
| TorchScript | PyTorch部署 | ⭐⭐⭐⭐ |
| ONNX | 跨框架部署 | ⭐⭐⭐⭐ |

**最佳实践**：
1. 总是用state_dict保存
2. 保存优化器状态以便继续训练
3. 保存最佳模型（不覆盖）
4. 用map_location处理设备问题

---

## 🎯 下一步

- [[09-PyTorch可视化与调试]] - 看看模型学到了什么
- [[10-PyTorch实战-CIFAR10图像分类]] - 完整项目实战

---

> 💡 **实践建议**：
> 1. 保存一个训练好的模型
> 2. 在另一个文件中加载并使用
> 3. 尝试从检查点继续训练
> 4. 把模型导出为TorchScript
