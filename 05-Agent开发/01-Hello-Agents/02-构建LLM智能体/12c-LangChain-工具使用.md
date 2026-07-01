---
module: "hello-agents"
title: "LangChain：工具使用"
description: "让LLM调用外部工具"
tags: [LangChain, Tools, Function-Calling]
---

# 🔧 LangChain：工具使用

> **让AI拥有超能力**

---

## 📝 前言

LLM本身只能处理文本。通过**工具（Tools）**，可以让LLM：
- 搜索网页
- 执行代码
- 查询数据库
- 调用API

---

## 🔰 1. 什么是Tool？

### 1.1 核心概念

Tool = **函数 + 描述**

LLM根据描述决定调用哪个工具。

### 1.2 生活类比

想象你是一个助手：
- 你有很多技能（工具）
- 老板告诉你任务
- 你选择合适的技能来完成

---

## 🔰 2. 定义Tool

### 2.1 使用@tool装饰器

```python
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """两数相加"""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """两数相乘"""
    return a * b

@tool
def search_web(query: str) -> str:
    """搜索网页"""
    # 实现搜索逻辑
    return f"搜索结果：{query}"
```

### 2.2 使用Tool类

```python
from langchain_core.tools import Tool

def get_weather(city: str) -> str:
    """获取天气"""
    return f"{city}今天晴"

weather_tool = Tool(
    name="get_weather",
    func=get_weather,
    description="获取指定城市的天气信息"
)
```

---

## 🔰 3. 绑定工具到LLM

### 3.1 绑定工具

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

# 绑定工具
llm_with_tools = llm.bind_tools([add, multiply, search_web])

# 调用
response = llm_with_tools.invoke("计算 2 + 3")
print(response.tool_calls)
# [{'name': 'add', 'args': {'a': 2, 'b': 3}}]
```

### 3.2 完整流程

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        return str(eval(expression))
    except:
        return "计算错误"

# 创建LLM
llm = ChatOpenAI(model="gpt-4")
llm_with_tools = llm.bind_tools([calculator])

# 用户输入
messages = [HumanMessage(content="计算 (2 + 3) * 4")]

# 调用
response = llm_with_tools.invoke(messages)

# 检查是否需要工具
if response.tool_calls:
    for tool_call in response.tool_calls:
        print(f"调用工具：{tool_call['name']}")
        print(f"参数：{tool_call['args']}")
        
        # 执行工具
        result = calculator.invoke(tool_call['args'])
        print(f"结果：{result}")
```

---

## 🔰 4. Agent中的工具

### 4.1 AgentExecutor

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 工具
tools = [calculator, search_web]

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# 创建Agent
llm = ChatOpenAI(model="gpt-4")
agent = create_tool_calling_agent(llm, tools, prompt)

# 创建执行器
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 运行
result = executor.invoke({"input": "计算 123 * 456"})
print(result["output"])
```

### 4.2 自定义Agent

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import MessagesPlaceholder

# 工具
tools = [search_web, calculator]

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个研究助手，可以搜索信息和计算"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Agent
agent = create_tool_calling_agent(llm, tools, prompt)

# 执行器
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5
)

# 带历史的对话
chat_history = []
result = executor.invoke({
    "input": "北京天气怎么样？",
    "chat_history": chat_history
})
```

---

## 🔰 5. 实用工具

### 5.1 搜索工具

```python
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()
result = search.invoke("Python最新版本")
```

### 5.2 文件操作

```python
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=2)
)
result = wiki.invoke("机器学习")
```

### 5.3 代码执行

```python
from langchain_experimental.tools import PythonREPLTool

python_tool = PythonREPLTool()
result = python_tool.invoke("print(2 + 3)")
```

---

## 📚 本节小结

| 概念 | 说明 |
|------|------|
| Tool | 工具定义 |
| bind_tools | 绑定到LLM |
| AgentExecutor | 执行Agent |
| 工具调用 | LLM决定调用哪个工具 |

---

## 🎯 下一步

- **13 - 智能体经典范式** - Agent模式
- **14 - 低代码平台** - Coze、Dify等

---

> 💡 **实践建议**：创建几个自定义工具，体验Agent调用工具的流程。
