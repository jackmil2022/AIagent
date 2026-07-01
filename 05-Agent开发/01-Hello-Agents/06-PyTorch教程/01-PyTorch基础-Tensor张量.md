---
module: "hello-agents"
title: "PyTorch基础：Tensor张量"
description: "理解Tensor是什么，掌握创建和操作Tensor的方法"
tags: [PyTorch, Tensor, Basics]
---

# 🧱 PyTorch基础：Tensor张量

> **Tensor是PyTorch的世界里的"原子"，一切数据和运算都围绕它展开**

---

## 📝 前言

### 什么是Tensor？

**Tensor（张量）** 就是一个**多维数组**，和NumPy的ndarray类似。

用生活中的例子理解：

| 维度 | 形状 | 生活例子 |
|------|------|----------|
| 0维 | `()` | 一个数字：`5` |
| 1维 | `(3,)` | 一行数字：`[1, 2, 3]` |
| 2维 | `(2, 3)` | 一个表格：2行3列 |
| 3维 | `(2, 3, 4)` | 一堆表格：2本，每本3行4列 |
| 4维 | `(N, C, H, W)` | 一批图片：N张，C通道，H高W宽 |

### Tensor和NumPy的区别

```python
# NumPy数组
import numpy as np
a = np.array([1, 2, 3])

# PyTorch Tensor
import torch
b = torch.tensor([1, 2, 3])

# 关键区别：Tensor可以在GPU上运行！
if torch.cuda.is_available():
    b_gpu = b.cuda()  # 移动到GPU
    print(b_gpu.device)  # cuda:0
```

---

## 🔰 1. 创建Tensor

### 1.1 从Python列表创建

```python
import torch

# 从列表创建
a = torch.tensor([1, 2, 3])
print(a)  # tensor([1, 2, 3])

# 二维
b = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(b)
# tensor([[1, 2, 3],
#         [4, 5, 6]])

print(b.shape)  # torch.Size([2, 3])  → 2行3列
```

### 1.2 特殊Tensor

```python
# 全零
zeros = torch.zeros(3, 4)  # 3行4列全0
print(zeros)

# 全一
ones = torch.ones(2, 3)  # 2行3列全1
print(ones)

# 随机数
rand = torch.rand(2, 3)  # 0~1之间的随机数
print(rand)

# 正态分布
randn = torch.randn(2, 3)  # 均值0，标准差1
print(randn)

# 等差序列
arange = torch.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
print(arange)

# 指定值填充
full = torch.full((3, 3), 3.14)  # 全是3.14
print(full)
```

### 1.3 指定数据类型

```python
# 默认是Float32
a = torch.tensor([1.0, 2.0, 3.0])
print(a.dtype)  # torch.float32

# 指定为Float64
b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
print(b.dtype)  # torch.float64

# 整数
c = torch.tensor([1, 2, 3], dtype=torch.int64)
print(c.dtype)  # torch.int64

# 布尔
d = torch.tensor([True, False, True])
print(d.dtype)  # torch.bool
```

### 1.4 常见创建方式对比

```python
# torch.tensor() - 从数据创建（推荐）
a = torch.tensor([1, 2, 3])

# torch.Tensor() - 注意大写T，不推荐（默认类型不同）
b = torch.Tensor([1, 2, 3])

# torch.FloatTensor() - 指定类型
c = torch.FloatTensor([1, 2, 3])

# 推荐用torch.tensor()，类型自动推断
```

---

## 🔰 2. Tensor运算

### 2.1 基本数学运算

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# 逐元素运算
print(a + b)    # tensor([5., 7., 9.])
print(a - b)    # tensor([-3., -3., -3.])
print(a * b)    # tensor([ 4., 10., 18.])
print(a / b)    # tensor([0.25, 0.40, 0.50])
print(a ** 2)   # tensor([1., 4., 9.])

# 广播机制（不同形状也能运算）
c = torch.tensor([[1.0], [2.0], [3.0]])  # (3, 1)
d = torch.tensor([10.0, 20.0, 30.0])    # (3,)
# c + d 会自动扩展为 (3, 3)
```

### 2.2 矩阵运算

```python
# 矩阵乘法
a = torch.tensor([[1, 2], [3, 4]])  # (2, 2)
b = torch.tensor([[5, 6], [7, 8]])  # (2, 2)

# 方式1：torch.mm()
c = torch.mm(a, b)
print(c)
# tensor([[19, 22],
#         [43, 50]])

# 方式2：@ 运算符（推荐）
c = a @ b
print(c)

# 方式3：torch.matmul()（支持广播）
c = torch.matmul(a, b)
```

### 2.3 聚合运算

```python
a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

print(a.sum())        # tensor(21.)  所有元素求和
print(a.mean())       # tensor(3.5000)  所有元素平均
print(a.max())        # tensor(6.)  最大值
print(a.min())        # tensor(1.)  最小值

# 沿维度运算
print(a.sum(dim=0))   # tensor([5., 7., 9.])  按列求和
print(a.sum(dim=1))   # tensor([ 6., 15.])   按行求和
print(a.mean(dim=0))  # tensor([2.5, 3.5, 4.5])  按列平均
```

---

## 🔰 3. Tensor形状操作

### 3.1 查看形状

```python
a = torch.zeros(2, 3, 4)

print(a.shape)        # torch.Size([2, 3, 4])
print(a.size())       # torch.Size([2, 3, 4])
print(a.dim())        # 3（维度数）
print(a.numel())      # 24（元素总数 = 2*3*4）
```

### 3.2 改变形状

```python
a = torch.arange(12)  # tensor([0, 1, 2, ..., 11])
print(a.shape)  # torch.Size([12])

# reshape
b = a.reshape(3, 4)
print(b.shape)  # torch.Size([3, 4])

c = a.reshape(2, 3, 2)
print(c.shape)  # torch.Size([2, 3, 2])

# view（和reshape类似，但要求连续内存）
d = a.view(4, 3)
print(d.shape)  # torch.Size([4, 3])

# flatten（展平）
e = torch.zeros(2, 3, 4)
f = e.flatten()
print(f.shape)  # torch.Size([24])
```

### 3.3 转置

```python
a = torch.tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
print(a.shape)

# 转置
b = a.t()  # 或者 a.T
print(b.shape)  # torch.Size([3, 2])

# 二维以上用transpose
c = torch.zeros(2, 3, 4)
d = c.transpose(0, 1)  # 交换第0维和第1维
print(d.shape)  # torch.Size([3, 2, 4])
```

### 3.4 拼接与拆分

```python
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

# 纵向拼接（按行）
c = torch.cat([a, b], dim=0)
print(c)
# tensor([[1, 2],
#         [3, 4],
#         [5, 6],
#         [7, 8]])

# 横向拼接（按列）
d = torch.cat([a, b], dim=1)
print(d)
# tensor([[1, 2, 5, 6],
#         [3, 4, 7, 8]])

# 拆分
e = torch.tensor([1, 2, 3, 4, 5, 6])
chunks = torch.chunk(e, 3)  # 拆成3份
print(chunks)  # [tensor([1, 2]), tensor([3, 4]), tensor([5, 6])]
```

---

## 🔰 4. 索引与切片

### 4.1 基本索引

```python
a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 取单个元素
print(a[0, 0])    # tensor(1)
print(a[1, 2])    # tensor(6)

# 取一行
print(a[0])       # tensor([1, 2, 3])

# 取一列
print(a[:, 0])    # tensor([1, 4, 7])
```

### 4.2 切片

```python
a = torch.arange(20).reshape(4, 5)
print(a)
# tensor([[ 0,  1,  2,  3,  4],
#         [ 5,  6,  7,  8,  9],
#         [10, 11, 12, 13, 14],
#         [15, 16, 17, 18, 19]])

# 取前2行
print(a[:2])
# tensor([[0, 1, 2, 3, 4],
#         [5, 6, 7, 8, 9]])

# 取后2列
print(a[:, 2:])
# tensor([[ 2,  3,  4],
#         [ 7,  8,  9],
#         [12, 13, 14],
#         [17, 18, 19]])

# 步长为2
print(a[::2, ::2])
# tensor([[ 0,  2,  4],
#         [10, 12, 14]])
```

### 4.3 布尔索引

```python
a = torch.tensor([1, 2, 3, 4, 5])

# 条件筛选
mask = a > 3
print(mask)  # tensor([False, False, False,  True,  True])

# 用mask取值
print(a[mask])  # tensor([4, 5])

# 等价写法
print(a[a > 3])  # tensor([4, 5])
```

---

## 🔰 5. CPU与GPU

### 5.1 设备检测

```python
import torch

# 检测是否有GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("使用CPU")

# 推荐写法：自动选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
```

### 5.2 移动Tensor到GPU

```python
# 创建Tensor（默认在CPU）
a = torch.randn(3, 3)
print(a.device)  # cpu

# 移动到GPU
a_gpu = a.to(device)  # 或 a.cuda()
print(a_gpu.device)  # cuda:0

# 或者直接在GPU上创建
b = torch.randn(3, 3, device=device)
print(b.device)  # cuda:0
```

### 5.3 GPU运算

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 在GPU上做运算
a = torch.randn(1000, 1000, device=device)
b = torch.randn(1000, 1000, device=device)

c = a @ b  # GPU矩阵乘法（比CPU快很多）
print(c.shape)  # torch.Size([1000, 1000])
```

### 5.4 移动回CPU

```python
# GPU Tensor转回CPU（比如要保存或用NumPy处理）
a_gpu = torch.randn(3, 3, device="cuda")
a_cpu = a_gpu.cpu()  # 移动到CPU

# 转为NumPy（必须先在CPU上）
a_numpy = a_cpu.numpy()
print(type(a_numpy))  # <class 'numpy.ndarray'>
```

---

## 🔰 6. Tensor与NumPy互转

### 6.1 Tensor → NumPy

```python
import torch
import numpy as np

# Tensor转NumPy
a = torch.tensor([1, 2, 3])
b = a.numpy()
print(type(b))  # <class 'numpy.ndarray'>
print(b)  # [1 2 3]
```

### 6.2 NumPy → Tensor

```python
# NumPy转Tensor
a = np.array([1, 2, 3])
b = torch.from_numpy(a)  # 共享内存
print(type(b))  # <class 'torch.Tensor'>
print(b)  # tensor([1, 2, 3])

# 或者用torch.tensor()（复制数据，不共享内存）
c = torch.tensor(a)
```

### 6.3 注意事项

```python
# 共享内存的情况
a = torch.tensor([1, 2, 3])
b = a.numpy()  # b和a共享内存
b[0] = 100
print(a[0])  # tensor(100) ← a也变了！

# 用clone()避免共享
a = torch.tensor([1, 2, 3])
b = a.clone().numpy()  # b是独立副本
b[0] = 100
print(a[0])  # tensor(1) ← a不受影响
```

---

## 🔰 7. 实用技巧

### 7.1 自动求导的Tensor

```python
# requires_grad=True 表示需要计算梯度
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(x.requires_grad)  # True

# 后续运算会自动记录梯度
y = x.sum()
y.backward()  # 反向传播
print(x.grad)  # tensor([1., 1., 1.])  梯度
```

### 7.2 detach() — 切断梯度

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2

# detach()创建一个不参与求导的新Tensor
z = y.detach()
print(z.requires_grad)  # False
```

### 7.3.item() — 获取标量值

```python
# 单元素Tensor取值
a = torch.tensor(5.0)
print(a.item())  # 5.0（Python数字）

# 多元素Tensor不能用item()
b = torch.tensor([1, 2, 3])
# b.item()  # 报错！
print(b[0].item())  # 1（取单个元素）
```

---

## 📚 本节小结

| 概念 | 说明 |
|------|------|
| Tensor | 多维数组，PyTorch的基本数据结构 |
| 创建方式 | `torch.tensor()`, `zeros()`, `ones()`, `rand()` |
| 运算 | `+`, `-`, `*`, `/`, `@`（矩阵乘法） |
| 形状 | `reshape()`, `view()`, `t()`, `cat()` |
| 索引 | 类似NumPy的切片和布尔索引 |
| GPU | `.to(device)` 移动，`.cpu()` 移回 |
| NumPy互转 | `.numpy()`, `torch.from_numpy()` |

---

## 🎯 下一步

- [[02-PyTorch基础-Autograd自动求导]] - 理解梯度和自动求导
- [[03-PyTorch数据处理-Dataset与DataLoader]] - 学习如何加载数据

---

> 💡 **实践建议**：
> 1. 尝试创建各种形状的Tensor
> 2. 用不同的方式创建全0、全1、随机Tensor
> 3. 练习Tensor的形状变换（reshape、transpose）
> 4. 如果有GPU，试试把Tensor移到GPU上运算
