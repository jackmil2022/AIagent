---
module: "hello-agents"
title: "Agent架构：多Agent协作"
description: "多个Agent协同工作的设计模式"
tags: [Agent, Multi-Agent, Collaboration, Architecture]
---

# 🤝 Agent架构：多Agent协作

> **让多个AI助手协同完成复杂任务**

---

## 📝 前言

单个Agent能力有限，就像一个人无法完成所有工作。

**多Agent协作** 让不同专长的Agent分工合作，就像一个团队。

---

## 🔰 1. 为什么需要多Agent？

### 1.1 生活类比

一个公司需要：
- 销售：了解客户需求
- 产品：设计解决方案
- 技术：实现产品
- 运营：推广销售

每个角色有不同专长，协作才能完成项目。

### 1.2 多Agent优势

| 优势 | 说明 |
|------|------|
| 专业分工 | 每个Agent专注一个领域 |
| 并行处理 | 同时处理多个任务 |
| 容错能力 | 一个Agent失败不影响整体 |
| 可扩展性 | 容易添加新Agent |

---

## 🔰 2. 协作模式

### 2.1 层级式（Hierarchical）

```
         Boss Agent
        /    |    \
   Agent A  Agent B  Agent C
```

**特点**：有明确的上下级关系

**适用场景**：任务可以明确分解

### 2.2 协作式（Collaborative）

```
Agent A ←→ Agent B
  ↕           ↕
Agent C ←→ Agent D
```

**特点**：平等协作，无中心

**适用场景**：需要多方协商

### 2.3 流水线式（Pipeline）

```
Agent A → Agent B → Agent C → 最终结果
```

**特点**：按顺序处理

**适用场景**：任务有明确流程

### 2.4 辩论式（Debate）

```
Agent A (正方) ←→ Agent B (反方)
         ↓
    评判Agent
         ↓
      最终结论
```

**特点**：通过辩论得出结论

**适用场景**：需要多角度分析

---

## 🔰 3. 角色设计

### 3.1 常见角色

| 角色 | 职责 | 技能 |
|------|------|------|
| Planner | 制定计划 | 任务分解 |
| Researcher | 信息收集 | 搜索、阅读 |
| Coder | 代码实现 | 编程 |
| Reviewer | 质量检查 | 代码审查 |
| Writer | 文档撰写 | 写作 |

### 3.2 角色定义

```python
roles = {
    "planner": {
        "name": "规划师",
        "description": "负责任务分解和计划制定",
        "skills": ["任务分析", "资源分配", "时间管理"],
        "system_prompt": "你是一个项目经理，负责将复杂任务分解为可执行的子任务..."
    },
    "researcher": {
        "name": "研究员",
        "description": "负责信息收集和分析",
        "skills": ["网络搜索", "文献阅读", "信息整理"],
        "system_prompt": "你是一个研究助理，擅长收集和整理相关信息..."
    },
    "coder": {
        "name": "程序员",
        "description": "负责代码实现",
        "skills": ["Python", "调试", "优化"],
        "system_prompt": "你是一个高级程序员，擅长编写高质量代码..."
    }
}
```

---

## 🔰 4. 实现框架

### 4.1 使用 LangGraph

```python
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

# 定义状态
class WorkflowState:
    task: str
    plan: str = None
    research: str = None
    code: str = None
    review: str = None
    final: str = None

# 定义Agent节点
def planner_agent(state):
    """规划Agent"""
    llm = ChatOpenAI(model="gpt-4")
    prompt = f"请将以下任务分解为子任务：{state['task']}"
    plan = llm.invoke(prompt).content
    return {"plan": plan}

def researcher_agent(state):
    """研究Agent"""
    llm = ChatOpenAI(model="gpt-4")
    prompt = f"请研究以下主题：{state['plan']}"
    research = llm.invoke(prompt).content
    return {"research": research}

def coder_agent(state):
    """编程Agent"""
    llm = ChatOpenAI(model="gpt-4")
    prompt = f"请根据以下设计实现代码：{state['research']}"
    code = llm.invoke(prompt).content
    return {"code": code}

def reviewer_agent(state):
    """审查Agent"""
    llm = ChatOpenAI(model="gpt-4")
    prompt = f"请审查以下代码：{state['code']}"
    review = llm.invoke(prompt).content
    return {"review": review}

# 构建工作流
workflow = StateGraph(WorkflowState)

# 添加节点
workflow.add_node("planner", planner_agent)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("coder", coder_agent)
workflow.add_node("reviewer", reviewer_agent)

# 添加边
workflow.set_entry_point("planner")
workflow.add_edge("planner", "researcher")
workflow.add_edge("researcher", "coder")
workflow.add_edge("coder", "reviewer")
workflow.add_edge("reviewer", END)

# 编译
app = workflow.compile()

# 运行
result = app.invoke({"task": "实现一个简单的Web爬虫"})
print(result)
```

### 4.2 使用 CrewAI

```python
from crewai import Agent, Task, Crew

# 定义Agent
researcher = Agent(
    role="研究员",
    goal="收集和分析相关信息",
    backstory="你是一位经验丰富的研究员...",
    verbose=True
)

coder = Agent(
    role="程序员",
    goal="实现高质量代码",
    backstory="你是一位高级程序员...",
    verbose=True
)

# 定义任务
research_task = Task(
    description="研究Python爬虫的最佳实践",
    agent=researcher
)

coding_task = Task(
    description="基于研究结果实现爬虫代码",
    agent=coder
)

# 创建团队
crew = Crew(
    agents=[researcher, coder],
    tasks=[research_task, coding_task],
    verbose=True
)

# 执行
result = crew.kickoff()
print(result)
```

---

## 🔰 5. 通信机制

### 5.1 共享状态

```python
class SharedState:
    def __init__(self):
        self.data = {}
        self.lock = threading.Lock()
    
    def set(self, key, value):
        with self.lock:
            self.data[key] = value
    
    def get(self, key, default=None):
        with self.lock:
            return self.data.get(key, default)
```

### 5.2 消息传递

```python
from queue import Queue

class MessageBus:
    def __init__(self):
        self.queues = {}
    
    def register(self, agent_id):
        self.queues[agent_id] = Queue()
    
    def send(self, to_agent, message):
        self.queues[to_agent].put(message)
    
    def receive(self, agent_id):
        return self.queues[agent_id].get()
```

### 5.3 黑板模式

```python
class Blackboard:
    def __init__(self):
        self.entries = {}
    
    def post(self, key, value, author):
        self.entries[key] = {
            "value": value,
            "author": author,
            "timestamp": time.time()
        }
    
    def read(self, key):
        return self.entries.get(key)
    
    def search(self, keyword):
        return {k: v for k, v in self.entries.items() 
                if keyword in str(v["value"])}
```

---

## 🔰 6. 实战案例

### 6.1 代码审查系统

```python
# 定义Agent
coder = Agent(
    role="开发者",
    goal="编写代码"
)

reviewer = Agent(
    role="审查者",
    goal="审查代码质量"
)

# 定义任务
def code_review_workflow(code_request):
    # 开发者写代码
    code = coder.write_code(code_request)
    
    # 审查者审查
    review = reviewer.review(code)
    
    # 如果有问题，开发者修改
    if review.has_issues:
        code = coder.fix_code(code, review.feedback)
        review = reviewer.review(code)
    
    return code
```

### 6.2 研究报告生成

```python
# 研究团队
planner = Agent(role="规划师")
researcher1 = Agent(role="研究员1")
researcher2 = Agent(role="研究员2")
writer = Agent(role="撰写者")

# 工作流
def research_workflow(topic):
    # 规划
    plan = planner.create_plan(topic)
    
    # 并行研究
    results = parallel([
        researcher1.research(plan.section1),
        researcher2.research(plan.section2)
    ])
    
    # 撰写报告
    report = writer.write_report(results)
    
    return report
```

---

## 📚 本节小结

| 模式 | 特点 | 适用场景 |
|------|------|----------|
| 层级式 | 上下级分工 | 任务可分解 |
| 协作式 | 平等协作 | 需要协商 |
| 流水线 | 顺序处理 | 流程明确 |
| 辩论式 | 对抗评估 | 需要多角度 |

---

## 🎯 下一步

- **26c - Agent架构：Agent编排** - 使用框架构建复杂系统
- **30 - 综合案例** - 实战项目

---

> 💡 **实践建议**：从简单的2-3个Agent协作开始，逐步扩展到更复杂的系统。
