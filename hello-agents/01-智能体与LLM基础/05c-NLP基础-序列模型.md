---
module: "hello-agents"
title: "NLP基础：序列模型"
description: "RNN、LSTM、GRU详解"
tags: [NLP, RNN, LSTM, GRU, Sequence]
---

# 🔄 NLP基础：序列模型

> **处理序列数据的神经网络**

---

## 📝 前言

文本是序列数据。本章将带你理解如何用神经网络处理序列。

---

## 🔰 1. 为什么需要序列模型？

### 1.1 序列数据的特点

```
"我 喜欢 吃 苹果"
  ↑    ↑   ↑   ↑
顺序很重要！
```

### 1.2 普通神经网络的问题

```
普通NN：
- 固定大小输入
- 不考虑顺序
- 无法处理变长序列
```

---

## 🔰 2. RNN（循环神经网络）

### 2.1 核心思想

**维护一个隐藏状态，捕捉序列信息**

```
h_t = f(W_hh * h_{t-1} + W_xh * x_t + b)
```

### 2.2 架构图

```
x_1 → [RNN] → h_1
              ↓
x_2 → [RNN] → h_2
              ↓
x_3 → [RNN] → h_3 → 输出
```

### 2.3 代码实现

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_xh = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)
        self.W_hy = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()
    
    def forward(self, x, hidden):
        # x: (batch, input_size)
        # hidden: (batch, hidden_size)
        
        hidden = self.tanh(self.W_xh(x) + self.W_hh(hidden))
        output = self.W_hy(hidden)
        
        return output, hidden

# 使用
rnn = SimpleRNN(input_size=10, hidden_size=20, output_size=5)
x = torch.randn(1, 10)
h = torch.zeros(1, 20)

output, h_new = rnn(x, h)
print(f"输出形状：{output.shape}")
print(f"隐藏状态形状：{h_new.shape}")
```

### 2.4 问题

| 问题 | 说明 |
|------|------|
| 梯度消失 | 难以学习长距离依赖 |
| 梯度爆炸 | 训练不稳定 |

---

## 🔰 3. LSTM（长短期记忆）

### 3.1 核心思想

**引入门控机制，控制信息流动**

### 3.2 门控机制

```
遗忘门：决定丢弃什么信息
输入门：决定存储什么信息
输出门：决定输出什么信息
```

### 3.3 数学公式

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)  # 遗忘门
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)  # 输入门
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)  # 候选值
C_t = f_t * C_{t-1} + i_t * C̃_t  # 更新细胞状态
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)  # 输出门
h_t = o_t * tanh(C_t)  # 更新隐藏状态
```

### 3.4 代码实现

```python
import torch
import torch.nn as nn

# PyTorch内置LSTM
lstm = nn.LSTM(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True,
    bidirectional=True
)

# 输入：(batch, seq_len, input_size)
x = torch.randn(1, 5, 10)

# 前向传播
output, (h_n, c_n) = lstm(x)

print(f"输出形状：{output.shape}")  # (1, 5, 40) 双向所以是40
print(f"隐藏状态形状：{h_n.shape}")  # (4, 1, 20) 2层*2方向
print(f"细胞状态形状：{c_n.shape}")
```

---

## 🔰 4. GRU（门控循环单元）

### 4.1 什么是GRU？

**LSTM的简化版，只有两个门**

### 4.2 对比

| 特性 | LSTM | GRU |
|------|------|-----|
| 门数量 | 3个 | 2个 |
| 参数量 | 更多 | 更少 |
| 训练速度 | 较慢 | 较快 |
| 效果 | 相当 | 相当 |

### 4.3 代码

```python
gru = nn.GRU(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True
)

x = torch.randn(1, 5, 10)
output, h_n = gru(x)

print(f"输出形状：{output.shape}")
```

---

## 🔰 5. Seq2Seq

### 5.1 架构

```
编码器：输入序列 → 隐藏状态
解码器：隐藏状态 → 输出序列
```

### 5.2 应用

| 应用 | 说明 |
|------|------|
| 机器翻译 | 源语言 → 目标语言 |
| 文本摘要 | 长文本 → 短摘要 |
| 对话系统 | 问题 → 回答 |

### 5.3 代码示例

```python
class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(output_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, src, trg):
        # 编码
        _, (hidden, cell) = self.encoder(src)
        
        # 解码
        output, _ = self.decoder(trg, (hidden, cell))
        output = self.fc(output)
        
        return output
```

---

## 📚 本节小结

| 模型 | 特点 | 适用场景 |
|------|------|----------|
| RNN | 基础序列模型 | 简单任务 |
| LSTM | 门控机制 | 长序列 |
| GRU | 简化版LSTM | 轻量任务 |
| Seq2Seq | 编码-解码 | 序列转换 |

---

## 🎯 下一步

- **05d - NLP基础：注意力机制** - Transformer基础
- **06a - Transformer：架构详解** - 现代架构

---

> 💡 **学习建议**：用LSTM实现一个简单的文本分类任务。
