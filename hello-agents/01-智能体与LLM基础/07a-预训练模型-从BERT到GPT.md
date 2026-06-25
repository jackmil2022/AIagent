---
module: "hello-agents"
title: "预训练模型：从BERT到GPT"
description: "理解预训练语言模型的演进"
tags: [Pre-training, BERT, GPT, Language-Model]
---

# 📚 预训练模型：从BERT到GPT

> **现代大模型的前世今生**

---

## 📝 前言

2018年是NLP的转折点。BERT和GPT的出现，开启了预训练大模型时代。

本章将带你理解预训练模型的核心思想和演进历程。

---

## 🔰 1. 预训练的革命

### 1.1 传统方法的问题

```
传统流程：
数据少 → 从头训练 → 效果差

问题：
- 需要大量标注数据
- 每个任务都要单独训练
- 模型泛化能力弱
```

### 1.2 预训练思想

```
预训练流程：
海量无标注数据 → 预训练 → 通用语言模型
                              ↓
                        少量标注数据 → 微调 → 任务模型
```

**核心思想**：先在大规模数据上学习通用知识，再针对具体任务微调。

---

## 🔰 2. 语言模型基础

### 2.1 什么是语言模型？

语言模型的任务：**预测下一个词**

```
输入：我 喜欢 吃 ___
预测：苹果（概率最高）
```

### 2.2 两种范式

| 范式 | 代表 | 任务 |
|------|------|------|
| CLM (Causal LM) | GPT | 从左到右预测 |
| MLM (Masked LM) | BERT | 预测被遮盖的词 |

### 2.3 数学定义

```
CLM: P(x_t | x_1, x_2, ..., x_{t-1})

MLM: P(x_mask | x_1, x_2, ..., x_n)  # 除了被遮盖的词
```

---

## 🔰 3. BERT

### 3.1 核心思想

**BERT = Bidirectional Encoder Representations from Transformers**

**双向**：同时看前面和后面的词

### 3.2 预训练任务

#### 掩码语言模型（MLM）

```
原始：我 喜欢 吃 苹果
遮盖：我 [MASK] 吃 [MASK]
预测：喜欢、苹果
```

#### 下一句预测（NSP）

```
句子A：今天天气很好
句子B：适合出去玩  →  IsNext
句子B：股票大跌    →  NotNext
```

### 3.3 架构

```
输入：[CLS] 我 喜欢 吃 苹果 [SEP]
       ↓
   Transformer Encoder
       ↓
输出：[CLS]的向量 → 分类
      各词的向量 → 序列标注
```

### 3.4 代码示例

```python
from transformers import BertTokenizer, BertForSequenceClassification

# 加载模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese')

# 编码
text = "我喜欢这个产品"
inputs = tokenizer(text, return_tensors="pt")

# 推理
outputs = model(**inputs)
logits = outputs.logits
print("预测结果：", logits)
```

---

## 🔰 4. GPT

### 4.1 核心思想

**GPT = Generative Pre-trained Transformer**

**单向**：从左到右生成

### 4.2 预训练任务

#### 因果语言模型（CLM）

```
输入：我 喜欢 吃
预测：苹果

目标：最大化 P(苹果 | 我, 喜欢, 吃)
```

### 4.3 演进

| 版本 | 年份 | 参数量 | 特点 |
|------|------|--------|------|
| GPT-1 | 2018 | 117M | 首个GPT |
| GPT-2 | 2019 | 1.5B | 生成能力强 |
| GPT-3 | 2020 | 175B | 少样本学习 |
| GPT-4 | 2023 | - | 多模态 |

### 4.4 GPT-3 的突破

```
传统方法：微调才能适应任务
GPT-3：直接给示例就能完成任务（In-context Learning）

示例：
Q: 1+1=?
A: 2

Q: 2+3=?
A: 5

Q: 5+7=?
A: 12（模型直接回答）
```

---

## 🔰 5. BERT vs GPT

### 5.1 架构对比

| 特性 | BERT | GPT |
|------|------|-----|
| 架构 | Encoder-only | Decoder-only |
| 注意力 | 双向 | 单向（因果） |
| 预训练 | MLM + NSP | CLM |
| 适合任务 | 理解任务 | 生成任务 |

### 5.2 适用场景

| 任务 | 推荐模型 |
|------|----------|
| 文本分类 | BERT |
| 命名实体识别 | BERT |
| 文本生成 | GPT |
| 对话系统 | GPT |
| 问答系统 | BERT/GPT |

---

## 🔰 6. 从BERT到LLM

### 6.1 演进路线

```
BERT (2018)
  ↓
RoBERTa, ALBERT (2019)
  ↓
GPT-3 (2020)
  ↓
PaLM, LLaMA (2022)
  ↓
GPT-4, Claude 3 (2023-2024)
```

### 6.2 关键突破

| 突破 | 说明 |
|------|------|
| 规模扩大 | 从110M到175B+ |
| 指令微调 | 对齐人类偏好 |
| RLHF | 人类反馈强化学习 |
| 上下文学习 | 不需要微调 |

---

## 🔰 7. 开源模型

### 7.1 主流开源模型

| 模型 | 公司 | 参数量 | 特点 |
|------|------|--------|------|
| LLaMA 3 | Meta | 8B-405B | 开源最强 |
| Qwen 2.5 | 阿里 | 0.5B-72B | 中文最强 |
| Mistral | Mistral | 7B | 性价比高 |
| DeepSeek | DeepSeek | 7B-671B | 推理能力强 |

### 7.2 使用示例

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载LLaMA
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")

# 生成
inputs = tokenizer("Once upon a time", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

---

## 📚 本节小结

| 概念 | 说明 |
|------|------|
| 预训练 | 在大规模数据上学习通用知识 |
| BERT | 双向编码器，适合理解任务 |
| GPT | 单向解码器，适合生成任务 |
| 演进 | 从BERT到GPT，从理解到生成 |

---

## 🎯 下一步

- **07b - 预训练模型：大语言模型演进** - GPT系列、LLaMA等
- **07c - 预训练模型：Scaling Laws** - 规模定律

---

> 💡 **学习建议**：理解BERT和GPT的区别，是理解现代大模型的基础。
