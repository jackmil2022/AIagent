---
module: "hello-agents"
title: "LangChain：链式调用"
description: "深入LCEL表达式语言"
tags: [LangChain, LCEL, Chain, Runnable]
---

# 🔗 LangChain：链式调用

> **用LCEL构建灵活的数据管道**

---

## 📝 前言

LCEL（LangChain Expression Language）是LangChain的新语法，更简洁、更强大。

---

## 🔰 1. LCEL 基础

### 1.1 管道操作符

```python
# 使用 | 操作符连接组件
chain = prompt | llm | parser
```

### 1.2 Runnable

LCEL中的一切都是**Runnable**：

```python
# Prompt是Runnable
prompt = ChatPromptTemplate.from_template("hello {name}")

# LLM是Runnable
llm = ChatOpenAI()

# Parser是Runnable
parser = StrOutputParser()

# Chain也是Runnable
chain = prompt | llm | parser
```

### 1.3 invoke方法

```python
# 所有Runnable都有invoke方法
result = chain.invoke({"name": "World"})
```

---

## 🔰 2. 组件详解

### 2.1 RunnablePassthrough

直接传递输入：

```python
from langchain.schema.runnable import RunnablePassthrough

# 直接传递
chain = RunnablePassthrough() | llm
```

### 2.2 RunnableLambda

使用自定义函数：

```python
from langchain.schema.runnable import RunnableLambda

def process(input):
    return input.upper()

chain = RunnableLambda(process) | llm
```

### 2.3 RunnableParallel

并行执行：

```python
from langchain.schema.runnable import RunnableParallel

# 并行执行两个Chain
parallel = RunnableParallel(
    outline=outline_chain,
    keywords=keywords_chain
)

result = parallel.invoke({"topic": "机器学习"})
# result包含outline和keywords两个结果
```

### 2.4 RunnableSequential

顺序执行：

```python
from langchain.schema.runnable import RunnableSequential

# 等同于使用 |
chain = RunnableSequential(prompt, llm, parser)
```

---

## 🔰 3. 高级用法

### 3.1 绑定输入

```python
# 部分参数预设
chain = prompt.partial(style="专业") | llm

# 调用时只需提供剩余参数
result = chain.invoke({"topic": "AI"})
```

### 3.2 添加回调

```python
from langchain.callbacks import StdOutCallbackHandler

chain = prompt | llm | parser

# 带回调的调用
result = chain.invoke(
    {"topic": "AI"},
    config={"callbacks": [StdOutCallbackHandler()]}
)
```

### 3.3 错误处理

```python
from langchain.schema.runnable import RunnableWithFallbacks

# 带回退的Chain
chain_with_fallback = primary_chain.with_fallbacks(
    [fallback_chain],
    except_key="error"
)
```

---

## 🔰 4. 实战示例

### 4.1 多步骤处理

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda

llm = ChatOpenAI(model="gpt-4")
parser = StrOutputParser()

# 步骤1：生成大纲
outline_prompt = ChatPromptTemplate.from_template(
    "请为{topic}生成大纲"
)
outline_chain = outline_prompt | llm | parser

# 步骤2：生成关键词
keywords_prompt = ChatPromptTemplate.from_template(
    "请为{topic}生成5个关键词"
)
keywords_chain = keywords_prompt | llm | parser

# 并行执行
parallel = RunnableParallel(
    outline=outline_chain,
    keywords=keywords_chain
)

# 调用
result = parallel.invoke({"topic": "机器学习"})
print("大纲：", result["outline"])
print("关键词：", result["keywords"])
```

### 4.2 条件分支

```python
from langchain.schema.runnable import RunnableBranch

# 条件分支
branch = RunnableBranch(
    (lambda x: "简单" in x["style"], simple_chain),
    (lambda x: "专业" in x["style"], professional_chain),
    default_chain  # 默认
)

result = branch.invoke({"style": "简单", "topic": "AI"})
```

### 4.3 流式输出

```python
# 流式调用
for chunk in chain.stream({"topic": "AI"}):
    print(chunk, end="", flush=True)
```

---

## 📚 本节小结

| 组件 | 说明 |
|------|------|
| Runnable | 基础单元 |
| \| 操作符 | 连接组件 |
| RunnableParallel | 并行执行 |
| RunnableLambda | 自定义函数 |
| RunnableBranch | 条件分支 |

---

## 🎯 下一步

- **12c - LangChain：工具使用** - Tool使用
- **13 - 智能体经典范式** - Agent模式

---

> 💡 **实践建议**：用LCEL重构之前的Chain，体验更简洁的语法。
