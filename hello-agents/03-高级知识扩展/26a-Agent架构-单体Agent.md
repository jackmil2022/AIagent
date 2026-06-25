---
module: "hello-agents"
title: "Agent架构：单体Agent"
description: "理解单体Agent的设计模式与实现"
tags: [Agent, Architecture, ReAct, Plan-Execute]
---

# 🏗️ Agent架构：单体Agent

> **从简单到复杂，理解Agent的核心设计模式**

---

## 📝 前言

在前面的章节中，我们学习了如何使用LLM、如何编写Prompt、如何使用工具。现在，我们要把这些能力组合起来，构建真正的Agent。

单体Agent是指**单个智能体**独立完成任务的架构。虽然简单，但它是理解多Agent系统的基础。

---

## 🔰 1. Agent 的核心循环

### 1.1 基本模式

Agent的核心是一个**感知-思考-行动**的循环：

```
感知（Perceive）
    ↓
思考（Think）
    ↓
行动（Act）
    ↓
观察结果（Observe）
    ↓
继续循环...
```

### 1.2 生活类比

想象你是一个侦探：
1. **感知**：看到犯罪现场的线索
2. **思考**：分析线索，推理案情
3. **行动**：去调查下一个线索
4. **观察**：得到新的信息
5. 循环，直到破案

---

## 🔰 2. ReAct 模式

### 2.1 什么是ReAct？

**ReAct = Reasoning + Acting（推理 + 行动）**

这是最经典的Agent模式，让LLM交替进行推理和行动。

### 2.2 工作流程

```
问题 → [思考] → [行动] → [观察] → [思考] → [行动] → ... → 答案
```

### 2.3 代码实现

```python
class ReActAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.memory = []
    
    def run(self, question, max_steps=10):
        """运行ReAct循环"""
        
        # 初始Prompt
        prompt = f"""
你是一个智能助手，需要解决以下问题。

问题：{question}

你可以使用以下工具：
{self._format_tools()}

请按照以下格式进行思考和行动：

Thought: [你的思考]
Action: [工具名称]
Action Input: [工具输入]
Observation: [工具返回的结果]

重复上述过程，直到找到答案。

最终答案格式：
Thought: 我已经找到了答案
Final Answer: [最终答案]
"""
        
        for step in range(max_steps):
            # LLM生成下一步
            response = self.llm.generate(prompt)
            
            # 解析响应
            thought, action, action_input = self._parse_response(response)
            
            # 记录思考
            self.memory.append({"thought": thought})
            
            # 检查是否完成
            if action == "Final Answer":
                return action_input
            
            # 执行行动
            observation = self._execute_action(action, action_input)
            
            # 更新Prompt
            prompt += f"""
Thought: {thought}
Action: {action}
Action Input: {action_input}
Observation: {observation}
"""
        
        return "达到最大步数，未能完成任务"
    
    def _execute_action(self, action, action_input):
        """执行工具调用"""
        for tool in self.tools:
            if tool.name == action:
                return tool.run(action_input)
        return f"未知工具：{action}"
```

### 2.4 实际例子

```python
# 问题：北京今天的天气怎么样？

# Step 1
Thought: 用户想知道北京今天的天气，我需要调用天气API
Action: weather_api
Action Input: 北京
Observation: 北京今天晴，温度25°C

# Step 2
Thought: 我已经获得了天气信息，可以回答用户了
Final Answer: 北京今天天气晴朗，温度25°C，非常适合外出。
```

---

## 🔰 3. Plan-and-Solve 模式

### 3.1 什么是Plan-and-Solve？

**先规划，再执行**

对于复杂任务，先制定计划，然后逐步执行。

### 3.2 工作流程

```
问题 → 制定计划 → [执行步骤1] → [执行步骤2] → ... → 完成
```

### 3.3 代码实现

```python
class PlanAndSolveAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
    
    def run(self, question):
        """运行Plan-and-Solve"""
        
        # 第一步：制定计划
        plan = self._create_plan(question)
        print(f"计划：{plan}")
        
        # 第二步：逐步执行
        results = []
        for i, step in enumerate(plan):
            print(f"执行步骤 {i+1}: {step}")
            result = self._execute_step(step, results)
            results.append(result)
            print(f"结果：{result}")
        
        # 第三步：汇总答案
        return self._summarize(question, plan, results)
    
    def _create_plan(self, question):
        """制定计划"""
        prompt = f"""
请为以下问题制定解决计划：

问题：{question}

请列出需要执行的步骤（每行一个步骤）：
"""
        response = self.llm.generate(prompt)
        steps = [s.strip() for s in response.split('\n') if s.strip()]
        return steps
    
    def _execute_step(self, step, previous_results):
        """执行单个步骤"""
        prompt = f"""
执行以下步骤：

步骤：{step}

之前的结果：
{previous_results}

请执行这个步骤并返回结果：
"""
        return self.llm.generate(prompt)
```

### 3.4 适用场景

| 场景 | 推荐模式 |
|------|----------|
| 简单问答 | 直接回答 |
| 需要工具调用 | ReAct |
| 复杂任务 | Plan-and-Solve |
| 需要迭代优化 | ReAct |

---

## 🔰 4. Reflection 模式

### 4.1 什么是Reflection？

**自我反思，迭代改进**

Agent生成答案后，自己检查并改进。

### 4.2 工作流程

```
问题 → 生成答案 → 自我评估 → 改进 → 再评估 → ... → 满意的答案
```

### 4.3 代码实现

```python
class ReflectionAgent:
    def __init__(self, llm):
        self.llm = llm
        self.max_iterations = 3
    
    def run(self, question):
        """运行Reflection循环"""
        
        # 初始生成
        answer = self._generate_answer(question)
        
        for i in range(self.max_iterations):
            print(f"\n--- 第{i+1}轮反思 ---")
            
            # 自我评估
            critique = self._self_critique(question, answer)
            print(f"自我批评：{critique}")
            
            # 如果满意，返回
            if "满意" in critique or "很好" in critique:
                print("答案已满意，停止反思")
                break
            
            # 改进答案
            answer = self._improve_answer(question, answer, critique)
            print(f"改进后：{answer[:100]}...")
        
        return answer
    
    def _generate_answer(self, question):
        """生成初始答案"""
        prompt = f"""
请回答以下问题：

问题：{question}

回答：
"""
        return self.llm.generate(prompt)
    
    def _self_critique(self, question, answer):
        """自我批评"""
        prompt = f"""
请评估以下回答的质量：

问题：{question}
回答：{answer}

请从以下方面评估：
1. 准确性
2. 完整性
3. 清晰度

如果你觉得回答很好，请说"满意"。
否则，请指出需要改进的地方。
"""
        return self.llm.generate(prompt)
    
    def _improve_answer(self, question, answer, critique):
        """改进答案"""
        prompt = f"""
请改进以下回答：

问题：{question}
原回答：{answer}
批评意见：{critique}

请给出改进后的回答：
"""
        return self.llm.generate(prompt)
```

### 4.4 适用场景

- 写作任务
- 代码生成
- 需要高质量输出的场景

---

## 🔰 5. 工具使用（Tool Use）

### 5.1 什么是工具？

工具是Agent可以调用的外部能力：
- 搜索引擎
- 计算器
- 数据库查询
- API调用
- 代码执行

### 5.2 工具定义

```python
from langchain.tools import tool

@tool
def search(query: str) -> str:
    """搜索互联网获取信息"""
    # 实现搜索逻辑
    return f"搜索结果：{query}的相关信息"

@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return str(result)
    except:
        return "计算错误"

@tool
def python_code(code: str) -> str:
    """执行Python代码"""
    try:
        exec(code)
        return "代码执行成功"
    except Exception as e:
        return f"执行错误：{str(e)}"
```

### 5.3 工具绑定

```python
from langchain.agents import initialize_agent

# 定义工具
tools = [search, calculator, python_code]

# 创建Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# 运行
result = agent.run("北京今天天气怎么样？100美元换算成人民币是多少？")
```

---

## 🔰 6. 记忆系统

### 6.1 为什么需要记忆？

Agent需要记住：
- 之前的对话
- 执行过的历史
- 学到的知识

### 6.2 记忆类型

| 类型 | 说明 | 实现方式 |
|------|------|----------|
| 短期记忆 | 当前对话 | 对话历史 |
| 长期记忆 | 跨对话 | 向量数据库 |
| 工作记忆 | 当前任务 | 变量存储 |

### 6.3 代码实现

```python
class Memory:
    def __init__(self):
        self.conversation_history = []
        self.working_memory = {}
    
    def add_message(self, role, content):
        """添加对话消息"""
        self.conversation_history.append({
            "role": role,
            "content": content
        })
    
    def get_context(self, max_messages=10):
        """获取上下文"""
        return self.conversation_history[-max_messages:]
    
    def set(self, key, value):
        """设置工作记忆"""
        self.working_memory[key] = value
    
    def get(self, key, default=None):
        """获取工作记忆"""
        return self.working_memory.get(key, default)
```

---

## 🔰 7. 完整Agent实现

```python
class SmartAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.memory = Memory()
        self.max_steps = 10
    
    def chat(self, user_input):
        """与用户对话"""
        
        # 添加用户消息到记忆
        self.memory.add_message("user", user_input)
        
        # 构建Prompt
        context = self.memory.get_context()
        prompt = self._build_prompt(user_input, context)
        
        # ReAct循环
        for step in range(self.max_steps):
            response = self.llm.generate(prompt)
            
            # 检查是否完成
            if "Final Answer:" in response:
                answer = response.split("Final Answer:")[-1].strip()
                self.memory.add_message("assistant", answer)
                return answer
            
            # 解析并执行行动
            action, action_input = self._parse_action(response)
            observation = self._execute_action(action, action_input)
            
            # 更新Prompt
            prompt += f"\n{response}\nObservation: {observation}\n"
        
        return "抱歉，我无法完成这个任务"
    
    def _build_prompt(self, user_input, context):
        """构建Prompt"""
        tools_desc = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])
        
        return f"""
你是一个智能助手，可以使用以下工具：

{tools_desc}

对话历史：
{context}

用户问题：{user_input}

请按照ReAct格式回答：
Thought: [你的思考]
Action: [工具名称]
Action Input: [工具输入]
"""
    
    def _parse_action(self, response):
        """解析行动"""
        lines = response.split('\n')
        action = None
        action_input = None
        
        for line in lines:
            if line.startswith("Action:"):
                action = line.split("Action:")[-1].strip()
            elif line.startswith("Action Input:"):
                action_input = line.split("Action Input:")[-1].strip()
        
        return action, action_input
    
    def _execute_action(self, action, action_input):
        """执行行动"""
        if action in self.tools:
            return self.tools[action].run(action_input)
        return f"未知工具：{action}"
```

---

## 📚 本节小结

| 模式 | 核心思想 | 适用场景 |
|------|----------|----------|
| ReAct | 推理+行动交替 | 需要工具调用的任务 |
| Plan-and-Solve | 先规划后执行 | 复杂多步骤任务 |
| Reflection | 自我反思改进 | 需要高质量输出 |

---

## 🎯 下一步

- **26b - Agent架构：多Agent协作** - 多个Agent协同工作
- **26c - Agent架构：Agent编排** - 使用框架构建复杂Agent系统

---

> 💡 **实践建议**：从简单的ReAct Agent开始，逐步添加工具和记忆，体验Agent的工作方式。

---

## 📚 相关笔记

### Agent 系列
- [[26b-Agent架构-多Agent协作]] - 多Agent协同
- [[26c-Agent架构-Agent编排]] - LangGraph、CrewAI

### 基础组件
- [[12c-LangChain-工具使用]] - 工具调用
- [[08-记忆与检索]] - 记忆系统
- [[10a-Prompt工程-基础技巧]] - Prompt设计

### 应用框架
- [[12a-LangChain-核心概念]] - LangChain基础
- [[07-构建你的Agent框架]] - 从零构建Agent

### 应用案例
- [[30a-案例-智能客服]] - 客服Agent
- [[13-智能旅行助手]] - 旅行助手

---

> 🏷️ 标签：#Agent #架构 #ReAct #LLM应用
