---
module: "hello-agents"
title: "Agent架构：Agent编排"
description: "使用框架构建复杂Agent系统"
tags: [Agent, Orchestration, LangGraph, CrewAI]
---

# 🎭 Agent架构：Agent编排

> **用框架构建复杂的Agent系统**

---

## 📝 前言

手动管理多Agent协作很复杂。幸运的是，有现成的框架可以帮助我们。

本章将介绍如何使用LangGraph、CrewAI等框架构建Agent系统。

---

## 🔰 1. 框架对比

### 1.1 主流框架

| 框架 | 公司 | 特点 | 适用场景 |
|------|------|------|----------|
| LangGraph | LangChain | 状态图 | 复杂工作流 |
| CrewAI | 开源 | 角色扮演 | 多Agent协作 |
| AutoGen | 微软 | 对话驱动 | 对话系统 |
| MetaGPT | 开源 | SOP驱动 | 软件开发 |

### 1.2 选择建议

| 场景 | 推荐框架 |
|------|----------|
| 复杂工作流 | LangGraph |
| 角色扮演 | CrewAI |
| 对话系统 | AutoGen |
| 代码生成 | MetaGPT |

---

## 🔰 2. LangGraph 详解

### 2.1 核心概念

| 概念 | 说明 |
|------|------|
| State | 共享状态 |
| Node | 处理节点 |
| Edge | 连接边 |
| Conditional Edge | 条件边 |

### 2.2 基本用法

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI

# 定义状态
class AgentState(TypedDict):
    input: str
    output: str
    next: str

# 创建LLM
llm = ChatOpenAI(model="gpt-4")

# 定义节点
def process(state):
    """处理节点"""
    response = llm.invoke(state["input"])
    return {"output": response.content}

def decide(state):
    """决策节点"""
    if "代码" in state["output"]:
        return {"next": "code"}
    else:
        return {"next": "end"}

# 构建图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("process", process)
workflow.add_node("decide", decide)

# 添加边
workflow.set_entry_point("process")
workflow.add_edge("process", "decide")
workflow.add_conditional_edges(
    "decide",
    lambda x: x["next"],
    {"code": "code", "end": END}
)

# 编译
app = workflow.compile()

# 运行
result = app.invoke({"input": "帮我写一个Python函数"})
print(result["output"])
```

### 2.3 高级特性

#### 子图

```python
# 定义子图
sub_workflow = StateGraph(AgentState)
sub_workflow.add_node("step1", step1)
sub_workflow.add_node("step2", step2)
sub_workflow.add_edge("step1", "step2")
sub_graph = sub_workflow.compile()

# 在主图中使用
main_workflow = StateGraph(AgentState)
main_workflow.add_node("sub", sub_graph)
```

#### 人机交互

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# 持久化
memory = SqliteSaver.from_conn_string(":memory:")

app = workflow.compile(checkpointer=memory)

# 运行时中断
config = {"configurable": {"thread_id": "1"}}
for event in app.stream({"input": "开始任务"}, config):
    if "interrupt" in event:
        # 等待人类输入
        human_input = input("请输入：")
        app.update_state(config, {"input": human_input})
```

---

## 🔰 3. CrewAI 详解

### 3.1 核心概念

| 概念 | 说明 |
|------|------|
| Agent | 角色 |
| Task | 任务 |
| Crew | 团队 |
| Process | 执行方式 |

### 3.2 基本用法

```python
from crewai import Agent, Task, Crew, Process

# 定义Agent
researcher = Agent(
    role="高级研究分析师",
    goal="发现关于AI的最新趋势和突破",
    backstory="你是一位经验丰富的研究分析师，擅长发现隐藏的模式和趋势。",
    verbose=True,
    allow_delegation=False
)

writer = Agent(
    role="技术内容作家",
    goal="将研究发现写成引人入胜的文章",
    backstory="你是一位才华横溢的作家，擅长将复杂概念转化为易于理解的内容。",
    verbose=True,
    allow_delegation=False
)

# 定义Task
research_task = Task(
    description="研究2024年AI领域的主要突破和趋势。",
    expected_output="一份包含5个主要趋势的详细报告。",
    agent=researcher
)

writing_task = Task(
    description="基于研究结果写一篇博客文章。",
    expected_output="一篇1000字的博客文章。",
    agent=writer,
    context=[research_task]  # 依赖研究任务的结果
)

# 创建Crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,  # 顺序执行
    verbose=True
)

# 执行
result = crew.kickoff()
print(result)
```

### 3.3 高级用法

#### 委托任务

```python
# 允许Agent委托任务给其他Agent
researcher = Agent(
    role="研究员",
    allow_delegation=True,  # 允许委托
    ...
)

# Agent可以这样委托
# "我需要数据分析，请数据分析师帮我"
```

#### 自定义工具

```python
from langchain.tools import Tool

def search_tool(query):
    # 实现搜索逻辑
    return f"搜索结果：{query}"

search = Tool(
    name="搜索",
    func=search_tool,
    description="用于搜索信息的工具"
)

researcher = Agent(
    role="研究员",
    tools=[search],
    ...
)
```

---

## 🔰 4. AutoGen 详解

### 4.1 核心概念

| 概念 | 说明 |
|------|------|
| Agent | 对话参与者 |
| GroupChat | 群聊 |
| GroupChatManager | 群聊管理器 |

### 4.2 基本用法

```python
import autogen

# 配置LLM
config_list = [
    {
        "model": "gpt-4",
        "api_key": "xxx"
    }
]

# 创建Agent
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={"config_list": config_list},
    system_message="你是一个有帮助的AI助手。"
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    code_execution_config={"work_dir": "coding"}
)

# 开始对话
user_proxy.initiate_chat(
    assistant,
    message="帮我写一个快速排序算法"
)
```

### 4.3 群聊

```python
# 创建多个Agent
coder = autogen.AssistantAgent(
    name="coder",
    system_message="你是一个程序员，擅长写代码。"
)

reviewer = autogen.AssistantAgent(
    name="reviewer",
    system_message="你是一个代码审查者，擅长发现bug。"
)

# 创建群聊
groupchat = autogen.GroupChat(
    agents=[user_proxy, coder, reviewer],
    messages=[],
    max_round=12
)

manager = autogen.GroupChatManager(groupchat=groupchat)

# 开始对话
user_proxy.initiate_chat(
    manager,
    message="实现一个简单的HTTP服务器并审查代码"
)
```

---

## 🔰 5. 实战案例

### 5.1 代码生成系统

```python
from crewai import Agent, Task, Crew

# 定义角色
planner = Agent(
    role="架构师",
    goal="设计系统架构",
    backstory="你是一位资深架构师..."
)

developer = Agent(
    role="开发者",
    goal="实现代码",
    backstory="你是一位高级开发者..."
)

tester = Agent(
    role="测试工程师",
    goal="编写测试用例",
    backstory="你是一位测试专家..."
)

# 定义任务
design_task = Task(
    description="设计一个RESTful API",
    agent=planner
)

implement_task = Task(
    description="根据设计实现API",
    agent=developer,
    context=[design_task]
)

test_task = Task(
    description="编写API测试用例",
    agent=tester,
    context=[implement_task]
)

# 创建团队
crew = Crew(
    agents=[planner, developer, tester],
    tasks=[design_task, implement_task, test_task],
    process=Process.sequential
)

# 执行
result = crew.kickoff()
```

---

## 📚 本节小结

| 框架 | 特点 | 适用场景 |
|------|------|----------|
| LangGraph | 状态图、灵活 | 复杂工作流 |
| CrewAI | 角色扮演、简单 | 多Agent协作 |
| AutoGen | 对话驱动 | 对话系统 |

---

## 🎯 下一步

- **30 - 综合案例** - 实战项目
- **31 - 毕业设计** - 完整应用

---

> 💡 **实践建议**：选择一个框架，完成一个小项目，深入理解其工作原理。
