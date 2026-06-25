---
module: "hello-agents"
title: "NLP基础：注意力机制"
description: "Transformer的基础"
tags: [NLP, Attention, Self-Attention, Multi-Head]
---

# 👁️ NLP基础：注意力机制

> **让模型学会"关注"重点**

---

## 📝 前言

注意力机制是Transformer的核心。本章将带你理解注意力的原理。

---

## 🔰 1. 为什么需要注意力？

### 1.1 RNN的问题

```
"我 去 北京 天安门 看 升旗 仪式"

处理到最后时，可能已经忘了"我"
```

### 1.2 注意力的思想

**让模型知道应该关注哪些词**

```
翻译"我 喜欢 吃 苹果" → "I like eating apples"

翻译"apples"时，应该关注"苹果"
```

---

## 🔰 2. Self-Attention

### 2.1 核心思想

**每个词都关注序列中的所有词**

### 2.2 Q、K、V

```
Query（查询）：我想要什么？
Key（键）：我有什么？
Value（值）：我的内容是什么？
```

### 2.3 计算过程

```
1. 生成Q、K、V
   Q = X × Wq
   K = X × Wk
   V = X × Wv

2. 计算注意力分数
   scores = Q × K^T / √d

3. Softmax归一化
   weights = softmax(scores)

4. 加权求和
   output = weights × V
```

### 2.4 直观理解

```
句子："小明 去 银行 取钱"

计算"银行"的注意力：
- "小明" → 0.2（相关）
- "去" → 0.3（相关）
- "银行" → 0.5（非常相关）
- "取钱" → 0.8（非常相关）

加权求和得到"银行"的新表示
```

---

## 🔰 3. 多头注意力

### 3.1 为什么需要多头？

**一个头只能关注一种关系，多头可以关注多种关系**

### 3.2 生活例子

评价电影时：
- 头1：关注剧情
- 头2：关注演技
- 头3：关注特效
- 头4：关注音乐

### 3.3 代码实现

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # 线性变换
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)
        
        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)
        
        return output

# 使用
mha = MultiHeadAttention()
x = torch.randn(1, 10, 512)
output = mha(x)
print(f"输出形状：{output.shape}")
```

---

## 🔰 4. 注意力掩码

### 4.1 Padding Mask

```
处理变长序列时，填充位置不应该被关注

"我 喜欢 吃" → [1, 2, 3, 0, 0]
                 ↑       ↑  ↑
               关注    不关注
```

### 4.2 Causal Mask（因果掩码）

```
GPT中使用，防止看到未来的词

"我 喜欢 吃 苹果"

处理"喜欢"时：
- 只能看到"我"
- 不能看到"吃"、"苹果"
```

### 4.3 代码实现

```python
def create_causal_mask(seq_len):
    """创建因果掩码"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask == 0  # True表示可以关注

# 示例
mask = create_causal_mask(4)
print(mask)
# tensor([[ True, False, False, False],
#         [ True,  True, False, False],
#         [ True,  True,  True, False],
#         [ True,  True,  True,  True]])
```

---

## 🔰 5. 位置编码

### 5.1 为什么需要？

**自注意力本身不关心顺序**

```
"我吃苹果" 和 "苹果吃我"
在没有位置信息时是一样的！
```

### 5.2 正弦位置编码

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### 5.3 代码实现

```python
import torch
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# 使用
pe = PositionalEncoding(d_model=512)
x = torch.randn(1, 10, 512)
output = pe(x)
print(f"输出形状：{output.shape}")
```

---

## 📚 本节小结

| 概念 | 说明 |
|------|------|
| Self-Attention | 每个词关注所有词 |
| Q、K、V | 查询、键、值 |
| 多头注意力 | 多种关系并行 |
| 注意力掩码 | 控制关注范围 |
| 位置编码 | 注入位置信息 |

---

## 🎯 下一步

- **06a - Transformer：架构详解** - 完整架构
- **07a - 预训练模型：从BERT到GPT** - 实际应用

---

> 💡 **学习建议**：手动实现Self-Attention，理解Q、K、V的作用。
