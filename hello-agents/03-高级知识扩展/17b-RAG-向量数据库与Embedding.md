---
title: "RAG：向量数据库与Embedding"
description: "深入理解向量检索技术"
tags: [Vector-Database, Embedding, RAG, Retrieval]
---

# 🗄️ RAG：向量数据库与Embedding

> **让计算机理解语义相似性**

---

## 📝 前言

在RAG系统中，向量数据库和Embedding是核心组件。

- **Embedding**：把文本变成向量
- **向量数据库**：存储和检索向量

本章将深入讲解这两个组件。

---

## 🔰 1. Embedding 深入理解

### 1.1 什么是好的Embedding？

好的Embedding应该：
- **语义相似的文本，向量也相似**
- **支持高效的相似度计算**

### 1.2 相似度度量

#### 余弦相似度

衡量两个向量的**方向**是否相似。

```python
import numpy as np

def cosine_similarity(vec1, vec2):
    """计算余弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# 示例
vec1 = np.array([1, 0, 0])
vec2 = np.array([0.5, 0.5, 0])
print(cosine_similarity(vec1, vec2))  # 0.707
```

#### 欧氏距离

衡量两个向量的**距离**。

```python
def euclidean_distance(vec1, vec2):
    """计算欧氏距离"""
    return np.linalg.norm(np.array(vec1) - np.array(vec2))
```

#### 点积

```python
def dot_product_similarity(vec1, vec2):
    """计算点积相似度"""
    return np.dot(vec1, vec2)
```

### 1.3 选择建议

| 度量方式 | 适用场景 |
|----------|----------|
| 余弦相似度 | 文本相似度（最常用） |
| 欧氏距离 | 图像相似度 |
| 点积 | 已归一化的向量 |

---

## 🔰 2. 主流Embedding模型

### 2.1 模型对比

| 模型 | 维度 | 特点 | 适用场景 |
|------|------|------|----------|
| all-MiniLM-L6-v2 | 384 | 快速、轻量 | 通用场景 |
| bge-large-zh-v1.5 | 1024 | 中文优化 | 中文场景 |
| text-embedding-ada-002 | 1536 | OpenAI | 高质量需求 |
| m3e-base | 768 | 开源中文 | 中文场景 |
| e5-large-v2 | 1024 | 多语言 | 多语言场景 |

### 2.2 使用 Hugging Face

```python
from sentence_transformers import SentenceTransformer

# 加载模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 编码文本
sentences = [
    "我喜欢吃苹果",
    "苹果很好吃",
    "今天天气真好"
]

embeddings = model.encode(sentences)
print("向量维度：", embeddings.shape)  # (3, 384)

# 计算相似度
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(embeddings)
print("相似度矩阵：")
print(sim_matrix)
```

### 2.3 使用 OpenAI Embedding

```python
from openai import OpenAI

client = OpenAI()

response = client.embeddings.create(
    model="text-embedding-ada-002",
    input=["我喜欢吃苹果"]
)

embedding = response.data[0].embedding
print("向量维度：", len(embedding))  # 1536
```

---

## 🔰 3. 向量数据库基础

### 3.1 为什么需要向量数据库？

- **高维向量检索**：传统数据库不适合
- **近似最近邻**：快速找到相似向量
- **可扩展性**：支持大规模数据

### 3.2 核心概念

| 概念 | 说明 |
|------|------|
| Collection | 类似表，存储向量 |
| Vector | 要存储的向量 |
| Metadata | 附加信息 |
| Query | 查询向量 |

### 3.3 索引算法

| 算法 | 特点 | 适用场景 |
|------|------|----------|
| HNSW | 高精度、高内存 | 精度要求高 |
| IVF | 平衡速度和精度 | 大规模数据 |
| LSH | 快速、低精度 | 快速检索 |
| PQ | 压缩向量 | 内存受限 |

---

## 🔰 4. Chroma 实战

### 4.1 基本使用

```python
import chromadb

# 创建客户端（内存模式）
client = chromadb.Client()

# 创建集合
collection = client.create_collection(
    name="my_documents",
    metadata={"hnsw:space": "cosine"}
)

# 添加文档
collection.add(
    documents=[
        "Python是一种解释型编程语言",
        "机器学习是AI的分支",
        "神经网络受生物启发"
    ],
    metadatas=[
        {"source": "python_docs"},
        {"source": "ml_books"},
        {"source": "dl_papers"}
    ],
    ids=["doc1", "doc2", "doc3"]
)

# 查询
results = collection.query(
    query_texts=["什么是深度学习？"],
    n_results=2
)
print(results)
```

### 4.2 持久化存储

```python
# 持久化到磁盘
client = chromadb.PersistentClient(path="./chroma_db")

# 或指定目录
client = chromadb.Client(chroma_settings={
    "chroma_db_impl": "duckdb+parquet",
    "persist_directory": "./chroma_db"
})
```

### 4.3 高级查询

```python
# 带过滤条件的查询
results = collection.query(
    query_texts=["Python编程"],
    n_results=5,
    where={"source": "python_docs"}  # 过滤条件
)

# 组合过滤
results = collection.query(
    query_texts=["机器学习"],
    n_results=5,
    where={
        "$and": [
            {"source": {"$eq": "ml_books"}},
            {"year": {"$gte": 2020}}
        ]
    }
)
```

---

## 🔰 5. FAISS 实战

### 5.1 基本使用

```python
import faiss
import numpy as np

# 创建索引
dimension = 384
index = faiss.IndexFlatL2(dimension)  # L2距离

# 添加向量
vectors = np.random.random((1000, dimension)).astype('float32')
index.add(vectors)

print(f"索引中共有 {index.ntotal} 个向量")

# 查询
query = np.random.random((1, dimension)).astype('float32')
k = 5  # 返回最近的5个
distances, indices = index.search(query, k)
print("距离：", distances)
print("索引：", indices)
```

### 5.2 高级索引

```python
# IVF索引（更快）
nlist = 100  # 聚类中心数
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
index.train(vectors)
index.add(vectors)

# HNSW索引（更准）
index = faiss.IndexHNSWFlat(dimension, 32)  # 32是图的度
index.add(vectors)
```

---

## 🔰 6. 实战：完整RAG系统

### 6.1 架构

```
文档 → 分块 → Embedding → 向量数据库
                                ↓
用户问题 → Embedding → 相似度检索 → 上下文 + 问题 → LLM → 答案
```

### 6.2 完整代码

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

# 1. 加载文档
loader = TextLoader("documents.txt")
documents = loader.load()

# 2. 文档分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？"]
)
chunks = text_splitter.split_documents(documents)

# 3. 向量化
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cuda'}
)

# 4. 存入向量数据库
vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./chroma_db"
)

# 5. 创建检索器
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# 6. 创建问答链
llm = Ollama(model="llama2")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 7. 提问
result = qa_chain({"query": "什么是机器学习？"})
print("答案：", result["result"])
print("来源：", [doc.metadata for doc in result["source_documents"]])
```

---

## 📚 本节小结

| 概念 | 说明 |
|------|------|
| Embedding | 文本到向量的转换 |
| 相似度 | 余弦相似度最常用 |
| 向量数据库 | 存储和检索向量 |
| Chroma | 简单易用，适合原型 |
| FAISS | 高性能，适合大规模 |

---

## 🎯 下一步

- **17c - RAG：高级检索策略** - 混合检索、重排序等
- **17d - RAG：实战与优化** - 完整系统搭建与优化

---

> 💡 **实践建议**：用Chroma搭建一个简单的向量检索系统，体验语义搜索的魅力。
