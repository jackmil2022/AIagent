---
module: "hello-agents"
title: "RAG：高级检索策略"
description: "混合检索、重排序等高级技巧"
tags: [RAG, Hybrid-Search, Reranking, Advanced]
---

# 🔍 RAG：高级检索策略

> **让检索更精准、更高效**

---

## 📝 前言

基础的向量检索可能不够用。本章将介绍更高级的检索策略。

---

## 🔰 1. 混合检索

### 1.1 什么是混合检索？

结合**关键词检索**和**语义检索**的优点。

```
关键词检索：精确匹配，适合专有名词
语义检索：理解含义，适合同义词
混合检索：两者结合，效果更好
```

### 1.2 实现方式

```python
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import BM25Retriever
from langchain.vectorstores import Chroma

# BM25检索器（关键词）
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 3

# 向量检索器（语义）
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 混合检索
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7]  # 关键词30%，语义70%
)

# 检索
results = ensemble_retriever.get_relevant_documents("机器学习")
```

### 1.3 权重调整

| 场景 | 关键词权重 | 语义权重 |
|------|------------|----------|
| 通用查询 | 0.3 | 0.7 |
| 专有名词 | 0.7 | 0.3 |
| 模糊查询 | 0.2 | 0.8 |

---

## 🔰 2. 重排序（Reranking）

### 2.1 什么是重排序？

对初步检索结果进行**重新排序**，提高准确性。

### 2.2 工作流程

```
查询 → 初步检索(100个) → 重排序 → 最终结果(10个)
```

### 2.3 使用Cohere重排序

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

# Cohere重排序
compressor = CohereRerank(model="rerank-multilingual-v3.0", top_n=3)

# 压缩检索器
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_retriever
)

# 检索
results = compression_retriever.get_relevant_documents("什么是深度学习")
```

### 2.4 使用Cross-Encoder

```python
from sentence_transformers import CrossEncoder
import numpy as np

# 加载重排序模型
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query, documents, top_k=3):
    """重排序"""
    # 构建查询-文档对
    pairs = [(query, doc.page_content) for doc in documents]
    
    # 计算分数
    scores = reranker.predict(pairs)
    
    # 排序
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    
    return [doc for doc, score in ranked[:top_k]]

# 使用
reranked_results = rerank("什么是机器学习", initial_results)
```

---

## 🔰 3. 查询改写

### 3.1 为什么需要查询改写？

用户的问题可能：
- 不够清晰
- 缺少上下文
- 与文档表述不同

### 3.2 HyDE（假设文档嵌入）

```python
from langchain.prompts import ChatPromptTemplate

# 步骤1：生成假设文档
hyde_prompt = ChatPromptTemplate.from_template(
    "请写一段关于以下问题的详细回答：\n{question}"
)

# 步骤2：用假设文档检索
def hyde_retrieval(question):
    # 生成假设文档
    hypothetical_doc = llm.invoke(hyde_prompt.format(question=question))
    
    # 用假设文档检索
    results = vectorstore.similarity_search(hypothetical_doc.content)
    return results
```

### 3.3 多查询检索

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

# 生成多个查询
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)

# 检索
results = multi_query_retriever.get_relevant_documents("什么是AI")
```

---

## 🔰 4. 父文档检索

### 4.1 什么是父文档检索？

检索小块，但返回大块，保持上下文完整。

```
小块：用于精准匹配
大块：用于提供完整上下文
```

### 4.2 实现

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore

# 分块器
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)

# 存储
vectorstore = Chroma(collection_name="split_parents")
store = InMemoryStore()

# 检索器
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)

# 添加文档
retriever.add_documents(documents)

# 检索
results = retriever.get_relevant_documents("机器学习")
```

---

## 🔰 5. 自查询检索

### 5.1 什么是自查询？

LLM自动将自然语言查询转换为结构化查询。

### 5.2 实现

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

# 定义元数据字段
metadata_field_info = [
    AttributeInfo(
        name="source",
        description="文档来源，如'website'或'book'",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="文档发布年份",
        type="integer",
    ),
]

document_content_description = "技术文档"

# 创建自查询检索器
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose=True
)

# 使用
results = retriever.get_relevant_documents(
    "2023年之后的关于机器学习的文章"
)
```

---

## 🔰 6. 评估检索质量

### 6.1 评估指标

| 指标 | 说明 |
|------|------|
| 精确率 | 检索结果中相关文档的比例 |
| 召回率 | 所有相关文档中被检索到的比例 |
| MRR | 平均倒数排名 |
| NDCG | 归一化折损累积增益 |

### 6.2 评估代码

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

# 准备评估数据
eval_dataset = {
    "question": ["什么是机器学习？"],
    "answer": ["机器学习是AI的一个分支..."],
    "contexts": [["机器学习是人工智能的一个分支，它使计算机..."]],
    "ground_truth": ["机器学习是人工智能的分支"]
}

# 评估
result = evaluate(
    dataset=eval_dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
)

print(result)
```

---

## 📚 本节小结

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| 混合检索 | 关键词+语义 | 通用场景 |
| 重排序 | 重新排序 | 提高精度 |
| 查询改写 | 优化查询 | 查询质量差 |
| 父文档 | 小块检索大块 | 保持上下文 |

---

## 🎯 下一步

- **17d - RAG：实战与优化** - 完整系统搭建
- **18 - 记忆与检索** - Agent记忆系统

---

> 💡 **实践建议**：尝试混合检索和重排序，对比效果差异。
