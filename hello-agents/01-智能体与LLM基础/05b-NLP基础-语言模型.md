---
module: "hello-agents"
title: "NLP基础：语言模型"
description: "理解语言模型的原理"
tags: [NLP, Language-Model, Perplexity, Neural-LM]
---

# 📖 NLP基础：语言模型

> **让计算机理解语言的概率**

---

## 📝 前言

语言模型是NLP的核心。本章将带你理解语言模型的原理。

---

## 🔰 1. 什么是语言模型？

### 1.1 核心任务

**预测下一个词的概率**

```
P("苹果" | "我喜欢吃") = 0.8
P("香蕉" | "我喜欢吃") = 0.15
P("汽车" | "我喜欢吃") = 0.001
```

### 1.2 应用场景

| 应用 | 说明 |
|------|------|
| 输入法 | 预测下一个字 |
| 机器翻译 | 选择最可能的翻译 |
| 文本生成 | 生成自然文本 |
| 语音识别 | 选择最可能的句子 |

---

## 🔰 2. N-gram 语言模型

### 2.1 基本思想

**用前N-1个词预测下一个词**

### 2.2 例子

```
Unigram (N=1): P(词)
Bigram (N=2): P(词2 | 词1)
Trigram (N=3): P(词3 | 词1, 词2)
```

### 2.3 代码示例

```python
from collections import defaultdict, Counter

class BigramModel:
    def __init__(self):
        self.bigram_counts = defaultdict(Counter)
        self.unigram_counts = Counter()
    
    def train(self, sentences):
        for sentence in sentences:
            words = sentence.split()
            for i in range(len(words) - 1):
                self.bigram_counts[words[i]][words[i+1]] += 1
                self.unigram_counts[words[i]] += 1
    
    def probability(self, word1, word2):
        """P(word2 | word1)"""
        if self.unigram_counts[word1] == 0:
            return 0
        return self.bigram_counts[word1][word2] / self.unigram_counts[word1]

# 训练
model = BigramModel()
sentences = [
    "我 喜欢 吃 苹果",
    "我 喜欢 吃 香蕉",
    "他 喜欢 吃 苹果"
]
model.train(sentences)

# 预测
print(model.probability("我", "喜欢"))  # 1.0
print(model.probability("喜欢", "吃"))  # 1.0
```

### 2.4 问题

| 问题 | 说明 |
|------|------|
| 数据稀疏 | 很多N-gram没见过 |
| 长距离依赖 | 只看前N-1个词 |

---

## 🔰 3. 神经语言模型

### 3.1 核心思想

**用神经网络预测下一个词**

```
词向量 → 隐藏层 → 输出层 → 概率分布
```

### 3.2 架构

```
输入：[w_{t-n+1}, ..., w_{t-1}] 的词向量
       ↓
    隐藏层
       ↓
输出：词表大小的概率分布
```

### 3.3 代码示例

```python
import torch
import torch.nn as nn

class NeuralLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        # x: (batch, seq_len)
        embeds = self.embedding(x)  # (batch, seq_len, embed_dim)
        rnn_out, _ = self.rnn(embeds)  # (batch, seq_len, hidden_dim)
        output = self.fc(rnn_out)  # (batch, seq_len, vocab_size)
        return output

# 创建模型
vocab_size = 10000
model = NeuralLM(vocab_size, embed_dim=128, hidden_dim=256)
```

---

## 🔰 4. 评估指标

### 4.1 困惑度（Perplexity）

```
PPL = exp(-1/N * Σ log P(w_i | w_{i-n+1:i-1}))

越低越好，表示模型越"确定"
```

### 4.2 代码计算

```python
import numpy as np

def perplexity(model, test_data):
    """计算困惑度"""
    log_prob = 0
    n_words = 0
    
    for sentence in test_data:
        words = sentence.split()
        for i in range(1, len(words)):
            prob = model.probability(words[i-1], words[i])
            if prob > 0:
                log_prob += np.log(prob)
            n_words += 1
    
    return np.exp(-log_prob / n_words)
```

---

## 🔰 5. 从语言模型到LLM

### 5.1 演进路线

```
N-gram → 神经LM → RNN-LM → Transformer-LM → GPT → LLM
```

### 5.2 关键突破

| 突破 | 说明 |
|------|------|
| 词向量 | 语义表示 |
| 注意力 | 长距离依赖 |
| 预训练 | 大规模学习 |
| 规模化 | 更多参数、更多数据 |

---

## 📚 本节小结

| 概念 | 说明 |
|------|------|
| 语言模型 | 预测下一个词 |
| N-gram | 统计方法 |
| 神经LM | 神经网络方法 |
| 困惑度 | 评估指标 |

---

## 🎯 下一步

- **05c - NLP基础：序列模型** - RNN、LSTM
- **05d - NLP基础：注意力机制** - Transformer基础

---

> 💡 **学习建议**：实现一个简单的N-gram模型，理解语言模型的基本思想。
