---
title: "RAG：基础概念"
description: "检索增强生成的核心原理与架构"
tags: [RAG, LLM, Retrieval, Generation]
---

# 🔍 RAG：基础概念

> **让大模型拥有"查资料"的能力**

---

## 📝 前言：为什么需要RAG？

### 大模型的局限性

想象一下，你有一个非常聪明的朋友，但他有一个致命缺点：
- **知识截止日期**：他只记得2024年之前的事情
- **没有你的私人文档**：他不知道你的公司内部资料
- **可能编造事实**：遇到不确定的问题，他会"一本正经地胡说八道"

这就是大语言模型（LLM）的三大问题：
1. **知识过时**：训练数据有截止日期
2. **缺乏私域知识**：没有你的企业数据、个人文档
3. **幻觉问题**：可能生成看似合理但错误的内容

### RAG 的解决方案

**RAG（Retrieval-Augmented Generation，检索增强生成）** 就是给AI配上一个"资料库"：

```
用户问题 → 检索相关资料 → 将资料给AI → AI基于资料回答
```

就像你考试时可以翻书一样！

---

## 🔰 1. RAG 的核心思想

### 1.1 生活类比

想象你去图书馆查资料写论文：

1. **用户问题**：你要写关于"人工智能发展史"的论文
2. **检索**：去图书馆找相关书籍和论文
3. **阅读**：快速浏览找到的资料
4. **生成**：基于资料写出论文

RAG 就是让AI经历这个过程！

### 1.2 技术定义

```
RAG = Retrieval（检索） + Augmented（增强） + Generation（生成）
```

- **检索**：从知识库中找到与问题相关的文档
- **增强**：将检索到的文档作为上下文
- **生成**：LLM基于上下文生成答案

---

## 🔰 2. RAG 的工作流程

### 2.1 完整流程图

```
┌─────────────────────────────────────────────────────────────┐
│                      RAG 工作流程                            │
└─────────────────────────────────────────────────────────────┘

离线阶段（知识库构建）：
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ 文档收集  │ → │ 文档分块  │ → │ 向量化   │ → │ 存入向量库│
└──────────┘    └──────────┘    └──────────┘    └──────────┘

在线阶段（问答过程）：
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ 用户提问  │ → │ 问题向量化│ → │ 检索相关  │ → │ LLM生成   │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
                                   ↑               │
                                   │               ↓
                              ┌──────────┐    ┌──────────┐
                              │ 向量数据库│    │ 返回答案  │
                              └──────────┘    └──────────┘
```

### 2.2 详细步骤

#### 步骤1：文档预处理

```python
# 原始文档
documents = [
    "Python是一种解释型编程语言...",
    "机器学习是人工智能的一个分支...",
    "神经网络受生物神经元启发..."
]

# 分块（Chunking）
chunks = [
    "Python是一种解释型编程语言，由Guido van Rossum创建...",
    "Python支持多种编程范式，包括面向对象、函数式...",
    "机器学习是人工智能的一个分支，它使计算机...",
    "机器学习算法通过数据学习模式，而不是...",
    "神经网络受生物神经元启发，由多层节点组成...",
    "神经网络可以学习复杂的非线性关系..."
]
```

#### 步骤2：向量化（Embedding）

```python
from sentence_transformers import SentenceTransformer

# 加载Embedding模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 将文本转换为向量
vectors = model.encode(chunks)
# 每个chunk变成一个384维的向量
```

#### 步骤3：存入向量数据库

```python
import chromadb

# 创建向量数据库
client = chromadb.Client()
collection = client.create_collection("my_docs")

# 存储文档和向量
collection.add(
    documents=chunks,
    embeddings=vectors.tolist(),
    ids=[f"doc_{i}" for i in range(len(chunks))]
)
```

#### 步骤4：检索与生成

```python
# 用户提问
query = "什么是Python？"

# 问题向量化
query_vector = model.encode([query])

# 检索相关文档
results = collection.query(
    query_embeddings=query_vector.tolist(),
    n_results=3  # 返回最相关的3个文档
)

# 构建Prompt
context = "\n".join(results['documents'][0])
prompt = f"""
基于以下参考资料回答问题。

参考资料：
{context}

问题：{query}

回答：
"""

# 调用LLM生成答案
# response = llm.generate(prompt)
```

---

## 🔰 3. 核心组件详解

### 3.1 文档分块（Chunking）

#### 为什么需要分块？

- LLM有上下文长度限制
- 太长的文档会影响检索精度
- 分块可以让检索更精准

#### 分块策略

| 策略 | 方法 | 优点 | 缺点 |
|------|------|------|------|
| 固定长度 | 按字符数切分 | 简单快速 | 可能切断语义 |
| 句子分割 | 按句子切分 | 保持语义完整 | 块大小不均 |
| 语义分割 | 按主题切分 | 最精准 | 计算成本高 |
| 递归分割 | 多级分割 | 平衡效果 | 实现复杂 |

#### 代码示例

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 递归字符分割器
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 每块最大500字符
    chunk_overlap=50,    # 块之间重叠50字符
    separators=["\n\n", "\n", "。", "！", "？"]
)

chunks = splitter.split_documents(documents)
```

### 3.2 向量化（Embedding）

#### 什么是Embedding？

Embedding 就是把文本转换成数字向量，让计算机能"理解"文本的含义。

#### 生活类比

想象给每个词找一个"坐标"：
- "国王" → [0.8, 0.2, 0.9, ...]
- "女王" → [0.7, 0.3, 0.85, ...]
- "苹果" → [0.1, 0.9, 0.2, ...]

"国王"和"女王"的向量很接近，因为它们语义相似！

#### 常用Embedding模型

| 模型 | 维度 | 特点 | 适用场景 |
|------|------|------|----------|
| all-MiniLM-L6-v2 | 384 | 快速、轻量 | 通用场景 |
| text-embedding-ada-002 | 1536 | OpenAI模型 | 高质量需求 |
| bge-large-zh-v1.5 | 1024 | 中文优化 | 中文场景 |
| m3e-base | 768 | 中文、开源 | 中文场景 |

### 3.3 向量数据库

#### 常用向量数据库

| 数据库 | 类型 | 特点 | 适用场景 |
|--------|------|------|----------|
| Chroma | 嵌入式 | 简单易用 | 原型开发 |
| FAISS | 库 | 高性能 | 大规模数据 |
| Milvus | 服务 | 分布式 | 企业级应用 |
| Pinecone | 云服务 | 托管服务 | 快速上线 |
| Weaviate | 服务 | 功能丰富 | 复杂场景 |

#### 代码示例（Chroma）

```python
import chromadb

# 创建客户端
client = chromadb.Client()

# 创建集合
collection = client.create_collection(
    name="my_documents",
    metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
)

# 添加文档
collection.add(
    documents=["文档1内容", "文档2内容"],
    metadatas=[{"source": "web"}, {"source": "pdf"}],
    ids=["doc1", "doc2"]
)

# 查询
results = collection.query(
    query_texts=["查询内容"],
    n_results=2
)
```

---

## 🔰 4. 检索策略

### 4.1 相似度检索

最基础的检索方式，找到与查询最相似的文档。

```python
# 余弦相似度
def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))
```

### 4.2 混合检索

结合关键词检索和语义检索：

```python
# 混合检索
results = hybrid_search(
    query="Python机器学习",
    keyword_weight=0.3,    # 关键词权重
    semantic_weight=0.7    # 语义权重
)
```

### 4.3 重排序（Reranking）

对初步检索结果进行重新排序：

```python
from sentence_transformers import CrossEncoder

# 加载重排序模型
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# 重新排序
scores = reranker.predict([(query, doc) for doc in candidates])
ranked_results = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
```

---

## 🔰 5. RAG vs Fine-tuning

### 5.1 对比表

| 维度 | RAG | Fine-tuning |
|------|-----|-------------|
| 知识更新 | 实时更新 | 需要重新训练 |
| 成本 | 低 | 高 |
| 数据需求 | 少量 | 大量 |
| 适用场景 | 知识密集型 | 任务特定型 |
| 可解释性 | 高（可追溯） | 低 |

### 5.2 选择建议

```
需要最新知识？ → RAG
需要特定领域？ → Fine-tuning
预算有限？ → RAG
需要高准确率？ → RAG + Fine-tuning
```

---

## 🔰 6. 实战示例

### 6.1 完整的简单RAG系统

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

# 1. 加载文档
loader = TextLoader("my_documents.txt")
documents = loader.load()

# 2. 文档分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

# 3. 向量化并存储
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings)

# 4. 创建问答链
llm = Ollama(model="llama2")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# 5. 提问
answer = qa_chain.run("什么是机器学习？")
print(answer)
```

---

## 📚 本节小结

| 概念 | 说明 |
|------|------|
| RAG | 检索增强生成，给LLM配上知识库 |
| 分块 | 将长文档切分为小段 |
| Embedding | 将文本转换为向量 |
| 向量数据库 | 存储和检索向量 |
| 检索策略 | 如何找到最相关的文档 |

---

## 🎯 下一步

- **17b - RAG：向量数据库与Embedding** - 深入学习向量检索技术
- **17c - RAG：高级检索策略** - 混合检索、重排序等高级技巧
- **17d - RAG：实战与优化** - 构建完整的RAG系统

---

> 💡 **实践建议**：先用LangChain + Chroma搭建一个简单的RAG系统，体验整个流程。

---

## 📚 相关笔记

### RAG 系列
- [[17b-RAG-向量数据库与Embedding]] - 深入向量检索
- [[17c-RAG-高级检索策略]] - 混合检索、重排序
- [[17d-RAG-实战与优化]] - 完整系统搭建

### 相关技术
- [[05a-NLP基础-文本表示]] - 文本向量化基础
- [[08-记忆与检索]] - Agent记忆系统
- [[12c-LangChain-工具使用]] - LangChain工具调用

### 应用场景
- [[30a-案例-智能客服]] - RAG在客服中的应用
- [[13-智能旅行助手]] - RAG在旅行助手中的应用

---

> 🏷️ 标签：#RAG #向量检索 #LLM应用
