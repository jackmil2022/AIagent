---
title: "RAG：实战与优化"
description: "构建完整的RAG系统并优化"
tags: [RAG,实战, Optimization, Production]
---

# ⚡ RAG：实战与优化

> **从原型到生产级RAG系统**

---

## 📝 前言

前面学习了RAG的基础和高级策略。本章将带你构建一个完整的RAG系统，并进行优化。

---

## 🔰 1. 完整RAG系统

### 1.1 系统架构

```
┌─────────────────────────────────────────────────┐
│                   RAG 系统                       │
├─────────────────────────────────────────────────┤
│                                                 │
│  文档处理层                                      │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
│  │ 文档加载 │→│ 文档分块 │→│ 向量化  │         │
│  └─────────┘  └─────────┘  └─────────┘         │
│       ↓                              ↓          │
│  检索层                        向量数据库        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
│  │ 查询处理 │→│ 混合检索 │→│ 重排序  │         │
│  └─────────┘  └─────────┘  └─────────┘         │
│       ↓                              ↓          │
│  生成层                        上下文组装        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
│  │ Prompt   │→│  LLM    │→│ 后处理  │         │
│  └─────────┘  └─────────┘  └─────────┘         │
│                                                 │
└─────────────────────────────────────────────────┘
```

### 1.2 完整代码

```python
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama
import os

class RAGSystem:
    def __init__(self, data_dir, persist_dir="./chroma_db"):
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        
        # 初始化组件
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.llm = Ollama(model="llama2")
        self.vectorstore = None
        self.qa_chain = None
    
    def load_documents(self):
        """加载文档"""
        loader = DirectoryLoader(
            self.data_dir,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        documents = loader.load()
        print(f"加载了 {len(documents)} 个文档")
        return documents
    
    def split_documents(self, documents, chunk_size=500, chunk_overlap=50):
        """分块"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？"]
        )
        chunks = splitter.split_documents(documents)
        print(f"分成 {len(chunks)} 个块")
        return chunks
    
    def build_index(self, chunks):
        """构建索引"""
        self.vectorstore = Chroma.from_documents(
            chunks,
            self.embeddings,
            persist_directory=self.persist_dir
        )
        print(f"索引构建完成，共 {len(chunks)} 个文档")
    
    def create_qa_chain(self):
        """创建问答链"""
        prompt_template = """
你是一个专业的助手。请根据以下参考资料回答问题。

参考资料：
{context}

问题：{question}

要求：
1. 只基于参考资料回答
2. 如果资料中没有相关信息，说明无法回答
3. 回答要准确、简洁
"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
    
    def query(self, question):
        """查询"""
        if not self.qa_chain:
            raise ValueError("请先构建索引")
        
        result = self.qa_chain({"query": question})
        return {
            "answer": result["result"],
            "sources": [doc.metadata for doc in result["source_documents"]]
        }
    
    def build(self):
        """完整构建流程"""
        documents = self.load_documents()
        chunks = self.split_documents(documents)
        self.build_index(chunks)
        self.create_qa_chain()
        print("RAG系统构建完成！")

# 使用
rag = RAGSystem("./documents")
rag.build()

# 查询
result = rag.query("什么是机器学习？")
print("答案：", result["answer"])
print("来源：", result["sources"])
```

---

## 🔰 2. 优化策略

### 2.1 分块优化

```python
# 策略1：语义分块
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers = [
    ("#", "标题1"),
    ("##", "标题2"),
    ("###", "标题3")
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
chunks = splitter.split_text(document)

# 策略2：递归分块
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", "。", "！", "？", " "]
)
```

### 2.2 检索优化

```python
# 策略1：混合检索
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import BM25Retriever

bm25 = BM25Retriever.from_documents(documents)
bm25.k = 5

vector = vectorstore.as_retriever(search_kwargs={"k": 5})

ensemble = EnsembleRetriever(
    retrievers=[bm25, vector],
    weights=[0.3, 0.7]
)

# 策略2：查询扩展
from langchain.retrievers.multi_query import MultiQueryRetriever

multi_query = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)
```

### 2.3 生成优化

```python
# 策略1：更好的Prompt
prompt_template = """
你是一个专业的助手。请严格按照以下要求回答：

1. 只使用提供的参考资料
2. 如果参考资料不足以回答，明确说明
3. 引用具体的来源
4. 保持回答简洁准确

参考资料：
{context}

问题：{question}

回答：
"""

# 策略2：引用来源
prompt_with_citation = """
请回答问题，并引用来源。

参考资料：
{context}

问题：{question}

回答格式：
[回答内容]

引用：
- [来源1]
- [来源2]
"""
```

---

## 🔰 3. 评估与监控

### 3.1 评估指标

```python
# RAGAS评估
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

# 准备数据
eval_data = {
    "question": ["什么是机器学习？", "Python的优势是什么？"],
    "answer": ["...", "..."],
    "contexts": [["..."], ["..."]],
    "ground_truth": ["...", "..."]
}

# 评估
result = evaluate(
    dataset=eval_data,
    metrics=[faithfulness, answer_relevancy]
)
```

### 3.2 监控日志

```python
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGMonitor:
    def __init__(self):
        self.query_count = 0
        self.total_latency = 0
    
    def log_query(self, question, answer, latency, sources):
        """记录查询日志"""
        self.query_count += 1
        self.total_latency += latency
        
        logger.info(f"""
查询时间: {datetime.now()}
问题: {question}
答案: {answer[:100]}...
延迟: {latency:.2f}秒
来源数量: {len(sources)}
""")
    
    def get_stats(self):
        """获取统计信息"""
        return {
            "total_queries": self.query_count,
            "avg_latency": self.total_latency / max(self.query_count, 1)
        }
```

---

## 🔰 4. 生产部署

### 4.1 API服务

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
rag = RAGSystem("./documents")
rag.build()

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

class QueryResponse(BaseModel):
    answer: str
    sources: list

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    result = rag.query(request.question)
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"]
    )

@app.get("/health")
async def health():
    return {"status": "healthy"}

# 运行：uvicorn main:app --reload
```

### 4.2 Docker部署

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 📚 本节小结

| 阶段 | 内容 |
|------|------|
| 文档处理 | 加载、分块、向量化 |
| 检索优化 | 混合检索、查询扩展 |
| 生成优化 | Prompt优化、引用来源 |
| 评估监控 | 指标评估、日志记录 |
| 生产部署 | API服务、Docker |

---

## 🎯 下一步

- **18 - 记忆与检索** - Agent记忆系统
- **30 - 综合案例** - 实战项目

---

> 💡 **实践建议**：先用小数据集跑通流程，再逐步优化和扩展。
