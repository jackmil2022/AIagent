---
title: "NLP基础：文本表示"
description: "如何让计算机理解人类语言"
tags: [NLP, Embedding, Word2Vec, Text-Representation]
---

# 📝 NLP基础：文本表示

> **让计算机"读懂"人类语言的第一步**

---

## 📝 前言

计算机只懂数字，不懂文字。要让计算机处理文本，首先要把文本转换成数字。

这就是**文本表示**的核心任务：**把文本变成计算机能理解的数字形式**。

---

## 🔰 1. 为什么需要文本表示？

### 1.1 生活类比

想象你要给外国人介绍"火锅"：
- 直接说"火锅"，他可能不理解
- 但如果说"一种中国传统的烹饪方式，把食材放在沸腾的汤锅里煮"，他就理解了

文本表示就是找到一种方式，让计算机能"理解"文本的含义。

### 1.2 技术挑战

- 文本是**非结构化**数据
- 不同语言、不同表达方式
- 需要捕捉语义信息

---

## 🔰 2. One-Hot 编码

### 2.1 基本思想

最简单的文本表示：给每个词一个独立的维度。

### 2.2 生活类比

想象一个巨大的投票箱：
- "猫" → [1, 0, 0, 0, ...]（只有第1个位置是1）
- "狗" → [0, 1, 0, 0, ...]（只有第2个位置是1）
- "苹果" → [0, 0, 1, 0, ...]（只有第3个位置是1）

### 2.3 代码示例

```python
import numpy as np

# 假设词汇表
vocab = ["猫", "狗", "苹果", "香蕉", "跑步"]

# One-Hot编码
def one_hot(word, vocab):
    vector = [0] * len(vocab)
    if word in vocab:
        vector[vocab.index(word)] = 1
    return vector

# 测试
print(one_hot("猫", vocab))  # [1, 0, 0, 0, 0]
print(one_hot("狗", vocab))  # [0, 1, 0, 0, 0]
print(one_hot("苹果", vocab))  # [0, 0, 1, 0, 0]
```

### 2.4 问题

| 问题 | 说明 |
|------|------|
| 维度灾难 | 词汇表很大时，向量很长 |
| 稀疏性 | 大部分位置都是0 |
| 无法表示语义 | "猫"和"狗"的距离与"猫"和"苹果"相同 |

---

## 🔰 3. TF-IDF

### 3.1 基本思想

**TF-IDF = 词频 × 逆文档频率**

- **TF（词频）**：词在文档中出现的次数
- **IDF（逆文档频率）**：词在多少文档中出现（越常见，权重越低）

### 3.2 生活类比

想象你在写论文：
- "的"、"是"这些词出现很多次，但不重要
- "量子计算"出现次数少，但很关键

TF-IDF就是找出那些"在当前文档很重要，但在其他文档不常见"的词。

### 3.3 公式

```
TF(t, d) = 词t在文档d中出现的次数 / 文档d的总词数

IDF(t) = log(总文档数 / 包含词t的文档数)

TF-IDF(t, d) = TF(t, d) × IDF(t)
```

### 3.4 代码示例

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# 文档集合
documents = [
    "我喜欢吃苹果",
    "我喜欢吃香蕉",
    "苹果和香蕉都是水果"
]

# 计算TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# 查看词汇表
print("词汇表：", vectorizer.get_feature_names_out())

# 查看TF-IDF值
print("TF-IDF矩阵：")
print(tfidf_matrix.toarray())

# 结果解释：
# "喜欢"在前两个文档中出现，IDF较低
# "水果"只在第三个文档出现，IDF较高
```

### 3.5 优缺点

| 优点 | 缺点 |
|------|------|
| 简单高效 | 无法捕捉语义 |
| 无监督方法 | 词序信息丢失 |
| 可解释性强 | 对同义词不敏感 |

---

## 🔰 4. Word2Vec

### 4.1 核心思想

**"一个词的含义由它的上下文决定"**

如果"猫"经常出现在"可爱"、"喵"、"宠物"附近，那么"猫"的向量应该和这些词的向量接近。

### 4.2 两种模型

#### CBOW（连续词袋模型）
- 用**上下文**预测**中心词**
- 例子：用"我 喜欢 吃 ___ 水果"预测"苹果"

#### Skip-gram
- 用**中心词**预测**上下文**
- 例子：用"苹果"预测"我 喜欢 吃 水果"

### 4.3 训练过程

```
原始文本：我 喜欢 吃 苹果

训练样本（Skip-gram）：
输入: 苹果 → 输出: 我
输入: 苹果 → 输出: 喜欢
输入: 苹果 → 输出: 吃

通过神经网络训练，得到：
"苹果"的向量 = [0.2, -0.5, 0.8, ...]
```

### 4.4 代码示例

```python
from gensim.models import Word2Vec

# 准备训练数据
sentences = [
    ["我", "喜欢", "吃", "苹果"],
    ["我", "喜欢", "吃", "香蕉"],
    ["猫", "喜欢", "吃", "鱼"],
    ["狗", "喜欢", "吃", "骨头"]
]

# 训练Word2Vec模型
model = Word2Vec(
    sentences,
    vector_size=100,  # 向量维度
    window=5,         # 上下文窗口
    min_count=1,      # 最小词频
    workers=4         # 并行线程数
)

# 获取词向量
print("苹果的向量：", model.wv["苹果"])

# 查找相似词
print("与'苹果'相似的词：", model.wv.most_similar("苹果"))

# 计算相似度
print("苹果和香蕉的相似度：", model.wv.similarity("苹果", "香蕉"))
```

### 4.5 有趣的发现

Word2Vec学到的向量有**语义关系**！

```python
# 经典例子
king - man + woman ≈ queen

# 在中文中
# 北京 - 中国 + 日本 ≈ 东京
```

---

## 🔰 5. GloVe

### 5.1 核心思想

**结合全局统计和局部上下文**

GloVe利用整个语料库的**共现矩阵**来学习词向量。

### 5.2 与Word2Vec的区别

| 方法 | Word2Vec | GloVe |
|------|----------|-------|
| 训练方式 | 局部上下文 | 全局共现矩阵 |
| 理论基础 | 预测模型 | 计数模型 |
| 效果 | 相当 | 相当 |

### 5.3 代码示例

```python
import gensim.downloader as api

# 加载预训练的GloVe模型
model = api.load("glove-wiki-gigaword-100")

# 使用
print("国王的向量：", model["king"])

# 相似词
print("与'king'相似的词：", model.most_similar("king"))

# 类比
print("king - man + woman =", model.most_similar(
    positive=["king", "woman"],
    negative=["man"]
)[0])
```

---

## 🔰 6. FastText

### 6.1 核心思想

**考虑子词信息**

FastText不仅看整个词，还看词的**子词**（字符n-gram）。

### 6.2 优势

- 可以为**未登录词**生成向量
- 对**形态丰富**的语言（如德语、土耳其语）效果更好

### 6.3 示例

```
"unhappiness" 的子词：
un, unh, nh, hap, happ, app, ppi, pin, iness, ness

这些子词的组合决定了"unhappiness"的向量
```

---

## 🔰 7. 现代Embedding模型

### 7.1 上下文相关Embedding

传统的Word2Vec给每个词一个**固定的向量**。

现代模型（如BERT）给每个词**动态的向量**，取决于上下文。

```
"我去银行取钱" → "银行"的向量偏向"金融机构"
"我在河边散步" → "河"的向量偏向"自然水域"
```

### 7.2 常用模型

| 模型 | 维度 | 特点 |
|------|------|------|
| BERT | 768 | 双向上下文 |
| GPT | 768/1024 | 单向上下文 |
| Sentence-BERT | 384/512 | 句子级别 |
| text-embedding-ada-002 | 1536 | OpenAI |

### 7.3 代码示例

```python
from sentence_transformers import SentenceTransformer

# 加载模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 获取句子Embedding
sentences = [
    "我喜欢吃苹果",
    "苹果很好吃",
    "今天天气真好"
]

embeddings = model.encode(sentences)

print("句子向量形状：", embeddings.shape)
# (3, 384)

# 计算相似度
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
print("句子1和句子2的相似度：", similarity[0][0])
# 0.7+ (相似)
```

---

## 🔰 8. 实战对比

### 8.1 完整示例

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer

# 准备数据
documents = [
    "我喜欢吃苹果和香蕉",
    "苹果和香蕉都是水果",
    "今天天气很好",
    "适合出去玩"
]

# 1. TF-IDF
tfidf = TfidfVectorizer()
tfidf_vectors = tfidf.fit_transform(documents)
print("TF-IDF向量维度：", tfidf_vectors.shape)

# 2. Word2Vec
tokenized = [doc.split() for doc in documents]
w2v_model = Word2Vec(tokenized, vector_size=50, min_count=1)
doc_vectors_w2v = np.mean([w2v_model.wv[words] for words in tokenized], axis=0)
print("Word2Vec向量维度：", doc_vectors_w2v.shape)

# 3. Sentence-BERT
sbert = SentenceTransformer('all-MiniLM-L6-v2')
sbert_vectors = sbert.encode(documents)
print("Sentence-BERT向量维度：", sbert_vectors.shape)
```

### 8.2 对比总结

| 方法 | 维度 | 语义理解 | 计算成本 | 适用场景 |
|------|------|----------|----------|----------|
| One-Hot | 词汇表大小 | ❌ 无 | 低 | 简单任务 |
| TF-IDF | 词汇表大小 | ⚠️ 弱 | 低 | 文本分类 |
| Word2Vec | 可配置 | ✅ 中 | 中 | 词相似度 |
| GloVe | 可配置 | ✅ 中 | 中 | 词类比 |
| Sentence-BERT | 384/512 | ✅ 强 | 高 | 语义检索 |

---

## 📚 本节小结

| 方法 | 核心思想 | 优缺点 |
|------|----------|--------|
| One-Hot | 独立维度 | 简单但无法表示语义 |
| TF-IDF | 词频统计 | 高效但忽略语序 |
| Word2Vec | 上下文预测 | 能捕捉语义 |
| GloVe | 全局共现 | 结合统计和预测 |
| Sentence-BERT | 上下文Embedding | 最强大但成本高 |

---

## 🎯 下一步

- **05b - NLP基础：语言模型** - 理解语言模型的原理
- **05c - NLP基础：序列模型** - RNN、LSTM等序列处理方法
- **05d - NLP基础：注意力机制** - Transformer的基础

---

> 💡 **实践建议**：用Gensim训练自己的Word2Vec模型，观察词向量的有趣性质。
