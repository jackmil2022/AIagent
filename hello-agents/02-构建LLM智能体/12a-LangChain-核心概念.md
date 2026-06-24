---
title: "LangChain：核心概念"
description: "LangChain框架的核心组件"
tags: [LangChain, Framework, Chain, Agent]
---

# 🔗 LangChain：核心概念

> **构建LLM应用的瑞士军刀**

---

## 📝 前言

LangChain 是最流行的LLM应用开发框架。

本章将带你理解其核心概念。

---

## 🔰 1. LangChain 是什么？

### 1.1 核心价值

LangChain 提供了一套**标准化的组件**，让你可以快速构建LLM应用。

### 1.2 核心组件

| 组件 | 说明 |
|------|------|
| Models | LLM封装 |
| Prompts | 模板管理 |
| Chains | 链式调用 |
| Agents | 智能代理 |
| Memory | 记忆管理 |
| Tools | 工具调用 |

---

## 🔰 2. Models

### 2.1 LLM vs ChatModel

| 类型 | 说明 | 示例 |
|------|------|------|
| LLM | 文本补全 | GPT-3 |
| ChatModel | 对话模型 | GPT-4, Claude |

### 2.2 使用示例

```python
from langchain_openai import ChatOpenAI

# 创建模型
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    api_key="sk-xxx"
)

# 调用
response = llm.invoke("你好，请介绍一下自己")
print(response.content)
```

### 2.3 多模型支持

```python
# OpenAI
from langchain_openai import ChatOpenAI

# Anthropic
from langchain_anthropic import ChatAnthropic

# 本地模型
from langchain_community.llms import Ollama

# 统一接口
llm = ChatOpenAI(model="gpt-4")
# 或
llm = ChatAnthropic(model="claude-3-opus")
# 或
llm = Ollama(model="llama2")
```

---

## 🔰 3. Prompts

### 3.1 PromptTemplate

```python
from langchain.prompts import PromptTemplate

# 创建模板
template = PromptTemplate(
    input_variables=["topic", "style"],
    template="请用{style}风格介绍{topic}"
)

# 使用
prompt = template.invoke({"topic": "机器学习", "style": "简单易懂"})
print(prompt)
```

### 3.2 ChatPromptTemplate

```python
from langchain.prompts import ChatPromptTemplate

# 聊天模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}"),
    ("human", "{question}")
])

# 使用
messages = prompt.invoke({
    "role": "编程老师",
    "question": "什么是面向对象编程？"
})
```

---

## 🔰 4. Chains（链）

### 4.1 什么是Chain？

Chain 是将多个组件**串联**起来的管道。

```
输入 → Prompt → LLM → 输出解析 → 输出
```

### 4.2 LCEL（LangChain表达式语言）

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# 创建组件
prompt = ChatPromptTemplate.from_template(
    "请用一句话介绍{topic}"
)
llm = ChatOpenAI(model="gpt-4")
parser = StrOutputParser()

# 使用LCEL连接
chain = prompt | llm | parser

# 调用
result = chain.invoke({"topic": "机器学习"})
print(result)
```

### 4.3 多步骤Chain

```python
# 步骤1：生成大纲
outline_prompt = ChatPromptTemplate.from_template(
    "请为{topic}生成一个大纲"
)

# 步骤2：展开内容
expand_prompt = ChatPromptTemplate.from_template(
    "请展开以下大纲的内容：\n{outline}"
)

# 步骤3：总结
summary_prompt = ChatPromptTemplate.from_template(
    "请总结以下内容：\n{content}"
)

# 组合Chain
outline_chain = outline_prompt | llm | parser
expand_chain = expand_prompt | llm | parser
summary_chain = summary_prompt | llm | parser

# 完整流程
full_chain = (
    {"outline": outline_chain}
    | {"content": expand_chain}
    | summary_chain
)

# 调用
result = full_chain.invoke({"topic": "机器学习"})
```

---

## 🔰 5. Output Parsers

### 5.1 常用解析器

| 解析器 | 说明 |
|--------|------|
| StrOutputParser | 字符串 |
| JsonOutputParser | JSON |
| PydanticOutputParser | Pydantic对象 |

### 5.2 使用示例

```python
from langchain.prompts import PromptTemplate
from langchain.output_parsers import JsonOutputParser
from pydantic import BaseModel

# 定义输出格式
class MovieReview(BaseModel):
    title: str
    rating: float
    summary: str

# JSON解析器
parser = JsonOutputParser(pydantic_object=MovieReview)

# Prompt
prompt = PromptTemplate(
    template="请评价电影{title}\n{format_instructions}",
    input_variables=["title"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 调用
chain = prompt | llm | parser
result = chain.invoke({"title": "泰坦尼克号"})
print(result)
```

---

## 🔰 6. 完整示例

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# 组件
llm = ChatOpenAI(model="gpt-4")
prompt = ChatPromptTemplate.from_template(
    "请用{style}风格写一段关于{topic}的文字"
)
parser = StrOutputParser()

# Chain
chain = prompt | llm | parser

# 调用
result = chain.invoke({
    "style": "专业",
    "topic": "人工智能的未来"
})
print(result)
```

---

## 📚 本节小结

| 组件 | 说明 |
|------|------|
| Models | LLM封装 |
| Prompts | 模板管理 |
| Chains | 链式调用 |
| Output Parsers | 输出解析 |

---

## 🎯 下一步

- **12b - LangChain：链式调用** - 深入LCEL
- **12c - LangChain：工具使用** - Tool使用

---

> 💡 **实践建议**：用LCEL构建一个简单的多步骤Chain，体验LangChain的便利。
