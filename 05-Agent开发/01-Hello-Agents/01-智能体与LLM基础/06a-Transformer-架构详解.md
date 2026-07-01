---
module: "hello-agents"
title: "Transformer：架构详解"
description: "深入理解Transformer的核心架构"
tags: [Transformer, Attention, BERT, GPT]
---

# 🔄 Transformer：架构详解

> **现代大模型的基石**

---

## 📝 前言

Transformer 是2017年Google提出的模型架构，论文标题是 "Attention Is All You Need"。

它彻底改变了NLP领域，几乎所有现代大模型（GPT、BERT、LLaMA等）都基于Transformer。

---

## 🔰 1. Transformer 之前

### 1.1 RNN的局限

在Transformer之前，主流方法是RNN（循环神经网络）：

| 问题 | 说明 |
|------|------|
| 顺序计算 | 必须按顺序处理，无法并行 |
| 长距离依赖 | 难以捕捉远距离关系 |
| 梯度消失 | 长序列训练困难 |

### 1.2 Transformer 的突破

| 优势 | 说明 |
|------|------|
| 并行计算 | 可以同时处理所有位置 |
| 全局注意力 | 任意位置之间可以直接交互 |
| 可扩展 | 容易扩展到更大规模 |

---

## 🔰 2. 整体架构

### 2.1 架构图

```
┌─────────────────────────────────────────────────┐
│                  Transformer                    │
├─────────────────────────────────────────────────┤
│                                                 │
│    ┌──────────────┐      ┌──────────────┐      │
│    │   Encoder    │      │   Decoder    │      │
│    │   (编码器)    │      │   (解码器)    │      │
│    │              │      │              │      │
│    │  ┌────────┐  │      │  ┌────────┐  │      │
│    │  │Self-Attn│  │      │  │Masked  │  │      │
│    │  └────────┘  │      │  │Self-Attn│  │      │
│    │       ↓      │      │  └────────┘  │      │
│    │  ┌────────┐  │      │       ↓      │      │
│    │  │   FFN  │  │      │  ┌────────┐  │      │
│    │  └────────┘  │      │  │Cross-Attn│ │      │
│    │              │      │  └────────┘  │      │
│    └──────────────┘      │       ↓      │      │
│                          │  ┌────────┐  │      │
│                          │  │   FFN  │  │      │
│                          │  └────────┘  │      │
│                          └──────────────┘      │
└─────────────────────────────────────────────────┘
```

### 2.2 两种使用方式

| 方式 | 代表模型 | 用途 |
|------|----------|------|
| Encoder-only | BERT | 文本理解、分类 |
| Decoder-only | GPT | 文本生成 |
| Encoder-Decoder | T5, BART | 翻译、摘要 |

---

## 🔰 3. 核心组件

### 3.1 自注意力机制 (Self-Attention)

#### 什么是注意力？

想象你在读一句话：
"小明去**银行**取钱"

当你看到"银行"时，你会自动关注"取钱"这个词，因为它们相关。

这就是注意力：**让模型知道应该关注哪些词**。

#### 数学原理

```
输入：X (序列长度 × 维度)

步骤1：生成Q, K, V
Q = X × Wq  (查询)
K = X × Wk  (键)
V = X × Wv  (值)

步骤2：计算注意力分数
scores = Q × K^T / √d

步骤3：Softmax归一化
weights = softmax(scores)

步骤4：加权求和
output = weights × V
```

#### 直观理解

```
Query（我想要什么）: 去银行
Key（我有什么）: 小明、去、银行、取钱
Value（内容）: 小明、去、银行、取钱

计算相关性：
- 小明 与 去银行：0.2
- 去   与 去银行：0.3
- 银行 与 去银行：0.5
- 取钱 与 去银行：0.8

加权求和得到输出
```

### 3.2 多头注意力 (Multi-Head Attention)

#### 为什么需要多头？

一个注意力头只能捕捉一种关系。多头可以同时关注不同类型的关系。

#### 生活类比

你在评价一部电影：
- 头1：关注剧情
- 头2：关注演技
- 头3：关注特效
- 头4：关注音乐

最后综合所有方面的评价。

#### 代码实现

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
```

### 3.3 位置编码 (Positional Encoding)

#### 为什么需要位置编码？

自注意力本身不关心顺序。"我吃苹果"和"苹果吃我"在没有位置信息时是一样的。

位置编码告诉模型每个词的位置。

#### 正弦位置编码

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

#### 代码实现

```python
import torch
import torch.nn as nn
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
```

### 3.4 前馈网络 (Feed-Forward Network)

#### 结构

```
输入 → 线性变换 → ReLU → 线性变换 → 输出
```

#### 代码

```python
class FeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))
```

### 3.5 层归一化 (Layer Normalization)

```python
# PyTorch实现
self.norm = nn.LayerNorm(d_model)

# 使用
output = self.norm(x + self_attention(x))
```

---

## 🔰 4. 完整Transformer

### 4.1 编码器层

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 自注意力 + 残差连接 + 层归一化
        attn_output = self.self_attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```

### 4.2 解码器层

```python
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.cross_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 掩码自注意力
        attn_output = self.self_attention(x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 交叉注意力
        cross_output = self.cross_attention(x, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x
```

---

## 🔰 5. GPT vs BERT

### 5.1 架构对比

| 特性 | GPT | BERT |
|------|-----|------|
| 架构 | Decoder-only | Encoder-only |
| 注意力方向 | 单向（从左到右） | 双向 |
| 预训练任务 | 语言模型 | 掩码语言模型 |
| 适用场景 | 文本生成 | 文本理解 |

### 5.2 注意力掩码

```python
# GPT的因果掩码（下三角）
def create_causal_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask == 0  # True表示可以关注

# BERT的双向掩码（全1）
def create_bert_mask(seq_len):
    return torch.ones(seq_len, seq_len, dtype=torch.bool)
```

---

## 📚 本节小结

| 组件 | 功能 | 关键点 |
|------|------|--------|
| 自注意力 | 捕捉词间关系 | Q, K, V计算 |
| 多头注意力 | 多种关系并行 | 分头计算再合并 |
| 位置编码 | 注入位置信息 | 正弦函数 |
| 前馈网络 | 非线性变换 | 两层线性+ReLU |
| 残差连接 | 缓解梯度消失 | 输入直接加到输出 |
| 层归一化 | 稳定训练 | 归一化到均值0方差1 |

---

## 🎯 下一步

- **06b - Transformer：位置编码** - RoPE、ALiBi等现代位置编码
- **06c - Transformer：代码实现** - 从零实现完整Transformer

---

> 💡 **学习建议**：亲手实现一遍Transformer，理解每个组件的作用。

---

## 📚 相关笔记

### 前置知识
- [[05d-NLP基础-注意力机制]] - 注意力机制基础
- [[05c-NLP基础-序列模型]] - RNN、LSTM
- [[04a-深度学习-神经网络基础]] - 神经网络基础

### Transformer 系列
- [[06b-Transformer-位置编码]] - RoPE、ALiBi
- [[06c-Transformer-代码实现]] - 从零实现

### 基于 Transformer 的模型
- [[07a-预训练模型-从BERT到GPT]] - BERT、GPT
- [[05b-NLP基础-语言模型]] - 语言模型基础

### 应用
- [[10a-Prompt工程-基础技巧]] - Prompt工程
- [[12a-LangChain-核心概念]] - LangChain框架

---

> 🏷️ 标签：#Transformer #Attention #NLP #深度学习
