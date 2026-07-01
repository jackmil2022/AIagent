---
module: "hello-agents"
title: "PyTorch基础：Autograd自动求导"
description: "理解梯度的概念，掌握PyTorch自动求导机制"
tags: [PyTorch, Autograd, Gradient, Basics]
---

# 🎯 PyTorch基础：Autograd自动求导

> **让PyTorch自动帮你算梯度，不用手动推导公式**

---

## 📝 前言

### 什么是梯度？

**梯度（Gradient）** 就是**函数的变化率**。

用生活例子理解：

```
你爬山时，脚下的坡度就是梯度：
- 坡度陡 → 梯度大 → 需要小心走
- 坡度缓 → 梯度小 → 可以快步走
- 平地 → 梯度为0 → 没有坡度
```

### 为什么需要梯度？

在训练神经网络时：
1. **前向传播**：输入数据 → 得到预测结果
2. **计算损失**：预测结果 vs 真实结果 → 差多少
3. **反向传播**：根据损失 → 计算每个参数的梯度
4. **更新参数**：沿着梯度方向 → 调整参数

**梯度告诉我们：参数应该往哪个方向调整、调整多少**

### Autograd是什么？

**Autograd = Automatic Gradient（自动梯度）**

PyTorch的Autograd能：
- 自动记录所有运算（构建计算图）
- 自动计算梯度（反向传播）
- 你只需要写前向传播代码！

---

## 🔰 1. requires_grad — 开启梯度追踪

### 1.1 基本用法

```python
import torch

# requires_grad=True 表示需要计算梯度
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(x.requires_grad)  # True

# 默认情况下，Tensor不需要梯度
y = torch.tensor([1.0, 2.0, 3.0])
print(y.requires_grad)  # False
```

### 1.2 运算中的梯度追踪

```python
# 创建需要梯度的Tensor
x = torch.tensor([2.0, 3.0], requires_grad=True)

# 运算会自动追踪
y = x ** 2  # y = x²
z = y.sum()  # z = x₁² + x₂²

print(f"x = {x}")
print(f"y = {y}")
print(f"z = {z}")
```

### 1.3 计算图

```python
x = torch.tensor(2.0, requires_grad=True)

# 运算构建计算图
y = x ** 2      # y = x²
z = y + 1       # z = y + 1
w = z * 2       # w = z * 2

# 查看计算图
print(f"x → {x}")
print(f"y = x² → {y}")
print(f"z = y + 1 → {z}")
print(f"w = z * 2 → {w}")
```

计算图结构：
```
x (2.0)
  ↓
  y = x² (4.0)
  ↓
  z = y + 1 (5.0)
  ↓
  w = z * 2 (10.0)
```

---

## 🔰 2. backward() — 反向传播

### 2.1 基本用法

```python
x = torch.tensor(3.0, requires_grad=True)

# 前向传播
y = x ** 2  # y = x² = 9

# 反向传播（计算梯度）
y.backward()

# 查看梯度
print(f"x的梯度: {x.grad}")  # tensor(6.)
# dy/dx = 2x = 2*3 = 6
```

### 2.2 多个变量的梯度

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 前向传播
y = x ** 2  # y = [x₁², x₂², x₃²]
z = y.sum()  # z = x₁² + x₂² + x₃²

# 反向传播
z.backward()

# 查看梯度
print(f"x的梯度: {x.grad}")
# tensor([2., 4., 6.])
# dz/dx₁ = 2x₁ = 2*1 = 2
# dz/dx₂ = 2x₂ = 2*2 = 4
# dz/dx₃ = 2x₃ = 2*3 = 6
```

### 2.3 向量对向量的梯度

```python
# 当输出是向量时，需要传入同形状的gradient
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2  # [1, 4, 9]

# 传入gradient参数（同形状的Tensor）
y.backward(torch.tensor([1.0, 1.0, 1.0]))
print(x.grad)  # tensor([2., 4., 6.])

# 等价于先sum再backward
x.grad = None  # 清除梯度
y.sum().backward()
print(x.grad)  # tensor([2., 4., 6.])
```

---

## 🔰 3. 梯度计算示例

### 3.1 简单函数

```python
# f(x) = x²
x = torch.tensor(4.0, requires_grad=True)
y = x ** 2
y.backward()
print(f"df/dx = {x.grad}")  # tensor(8.)  → 2x = 2*4 = 8
```

### 3.2 复合函数

```python
# f(x) = (x + 1)²
x = torch.tensor(3.0, requires_grad=True)
y = (x + 1) ** 2
y.backward()
print(f"df/dx = {x.grad}")  # tensor(8.)  → 2(x+1) = 2*4 = 8
```

### 3.3 链式法则

```python
# f(x) = sin(x²)
x = torch.tensor(1.0, requires_grad=True)
y = torch.sin(x ** 2)
y.backward()
# df/dx = cos(x²) * 2x = cos(1) * 2 ≈ 1.08
print(f"df/dx = {x.grad}")
```

### 3.4 多层嵌套

```python
x = torch.tensor(2.0, requires_grad=True)

# 多层运算
a = x + 1      # a = 3
b = a ** 2      # b = 9
c = b * 2       # c = 18
d = c + a       # d = 21

d.backward()
print(f"dd/dx = {x.grad}")  # tensor(14.)
# dd/dx = d(d)/d(c) * d(c)/d(b) * d(b)/d(a) * d(a)/d(x)
#        = 1 * 2 * 2 * a + 1 * 2 * 1
#        = 4a + 2 = 4*3 + 2 = 14
```

---

## 🔰 4. 梯度控制

### 4.1 torch.no_grad() — 禁用梯度

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 训练时不需要计算梯度的场景
with torch.no_grad():
    y = x * 2  # 这个运算不追踪梯度
    print(y.requires_grad)  # False

# 用于：
# 1. 推理时节省内存
# 2. 评估模型性能
# 3. 更新参数时不希望被追踪
```

### 4.2 detach() — 切断梯度

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2

# detach创建一个新Tensor，不参与求导
z = y.detach()
print(z.requires_grad)  # False

# z和y值相同，但梯度追踪被切断
print(y)  # tensor([2., 4., 6.], grad_fn=<MulBackward0>)
print(z)  # tensor([2., 4., 6.])
```

### 4.3 requires_grad_() — 原地修改

```python
x = torch.tensor([1.0, 2.0, 3.0])
print(x.requires_grad)  # False

# 原地修改requires_grad
x.requires_grad_(True)
print(x.requires_grad)  # True
```

### 4.4 清除梯度

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x.sum()
y.backward()
print(x.grad)  # tensor([1., 1.])

# 方法1：归零梯度（每次迭代必须做！）
x.grad.zero_()
print(x.grad)  # tensor([0., 0.])

# 方法2：用optimizer（推荐）
# optimizer.zero_grad()
```

---

## 🔰 5. 计算图详解

### 5.1 计算图是什么？

计算图是一张有向无环图（DAG），记录了所有运算：

```
x ──→ y = x² ──→ z = y + 1 ──→ w = z * 2
```

每个节点存储：
- **Tensor**：数据
- **grad_fn**：梯度函数（如何计算梯度）

### 5.2 grad_fn属性

```python
x = torch.tensor(2.0, requires_grad=True)

# 叶子节点（原始输入）
print(x.grad_fn)  # None

# 运算后的节点
y = x ** 2
print(y.grad_fn)  # PowBackward0

z = y + 1
print(z.grad_fn)  # AddBackward0

w = z * 2
print(w.grad_fn)  # MulBackward0
```

### 5.3 叶子节点 vs 中间节点

```python
x = torch.tensor(2.0, requires_grad=True)
y = x * 2

# is_leaf属性
print(x.is_leaf)  # True（叶子节点）
print(y.is_leaf)  # False（中间节点）

# 只有叶子节点才有梯度
print(x.grad_fn)  # None
print(y.grad_fn)  # MulBackward0
```

---

## 🔰 6. 实际应用示例

### 6.1 线性回归

```python
import torch

# 生成数据: y = 2x + 1 + 噪声
torch.manual_seed(42)
x = torch.randn(100, 1)
y = 2 * x + 1 + torch.randn(100, 1) * 0.1

# 初始化参数
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 训练循环
learning_rate = 0.01
for epoch in range(100):
    # 前向传播
    y_pred = x * w + b

    # 计算损失（MSE）
    loss = ((y_pred - y) ** 2).mean()

    # 反向传播
    loss.backward()

    # 更新参数（不用no_grad会报错！）
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # 清除梯度（必须！）
    w.grad.zero_()
    b.grad.zero_()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: loss={loss.item():.4f}")

print(f"\n训练结果: w={w.item():.4f}, b={b.item():.4f}")
# 接近 w=2, b=1
```

### 6.2 简单神经网络

```python
import torch

# 输入数据
x = torch.randn(100, 1)
y = torch.sin(x) + 0.1 * torch.randn(100, 1)

# 简单网络：1 → 10 → 1
w1 = torch.randn(1, 10, requires_grad=True)
b1 = torch.zeros(10, requires_grad=True)
w2 = torch.randn(10, 1, requires_grad=True)
b2 = torch.zeros(1, requires_grad=True)

# 训练
lr = 0.01
for epoch in range(200):
    # 前向传播
    h = torch.relu(x @ w1 + b1)  # 隐藏层
    output = h @ w2 + b2          # 输出层

    # 损失
    loss = ((output - y) ** 2).mean()

    # 反向传播
    loss.backward()

    # 更新
    with torch.no_grad():
        w2 -= lr * w2.grad
        b2 -= lr * b2.grad
        w1 -= lr * w1.grad
        b1 -= lr * b1.grad

    # 清零
    w1.grad.zero_()
    b1.grad.zero_()
    w2.grad.zero_()
    b2.grad.zero_()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}: loss={loss.item():.4f}")
```

---

## 🔰 7. 常见错误与调试

### 7.1 忘记清除梯度

```python
x = torch.tensor(2.0, requires_grad=True)

# 错误：梯度会累加！
for i in range(3):
    y = x ** 2
    y.backward()
    print(f"第{i+1}次: {x.grad}")  # 4, 8, 12 ← 累加了！
    # 必须 x.grad.zero_()

# 正确：每次清除
x = torch.tensor(2.0, requires_grad=True)
for i in range(3):
    y = x ** 2
    y.backward()
    print(f"第{i+1}次: {x.grad}")  # 4, 4, 4
    x.grad.zero_()
```

### 7.2 在no_grad中更新参数

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()

# 错误：不能在追踪梯度时修改Tensor
# x -= 0.01 * x.grad  # 报错！

# 正确：用no_grad
with torch.no_grad():
    x -= 0.01 * x.grad
```

### 7.3 中间节点求梯度

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
z = y * 2

# 只有叶子节点能获取梯度
print(x.grad)  # None（还没backward）
z.backward()
print(x.grad)  # tensor(8.)
# print(y.grad)  # 报错！y不是叶子节点
```

---

## 📚 本节小结

| 概念 | 说明 |
|------|------|
| requires_grad | 开启梯度追踪 |
| backward() | 反向传播，计算梯度 |
| .grad | 查看梯度值 |
| grad_fn | 梯度计算函数（运算记录） |
| is_leaf | 是否是叶子节点 |
| torch.no_grad() | 禁用梯度追踪 |
| detach() | 切断梯度 |
| grad.zero_() | 清除梯度（必须！） |

---

## 🎯 下一步

- [[03-PyTorch数据处理-Dataset与DataLoader]] - 学习如何加载数据
- [[04-PyTorch模型构建-nn.Module]] - 用nn.Module搭建神经网络

---

> 💡 **实践建议**：
> 1. 手动计算几个函数的梯度，然后用PyTorch验证
> 2. 理解计算图的结构
> 3. 练习使用torch.no_grad()
> 4. 运行线性回归示例，观察参数如何变化
