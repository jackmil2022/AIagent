---
module: "hello-agents"
title: "API调用：OpenAI接口"
description: "掌握OpenAI API的使用方法"
tags: [API, OpenAI, GPT, Integration]
---

# 🔌 API调用：OpenAI接口

> **让大模型为你的应用服务**

---

## 📝 前言

学会了Prompt工程，下一步就是让代码自动调用大模型。OpenAI的API是最常用的选择。

本章将带你：
1. 理解API的基本概念
2. 学会调用Chat Completion API
3. 掌握Function Calling
4. 实现流式输出

---

## 🔰 1. API 基础概念

### 1.1 什么是API？

**API = Application Programming Interface（应用程序编程接口）**

简单说，API就是**程序之间的通信方式**。

### 1.2 生活类比

想象你去餐厅：
- **你** = 客户端（你的程序）
- **服务员** = API
- **厨师** = 服务器（OpenAI）

你告诉服务员想吃什么（发送请求），服务员把菜端给你（返回响应）。

### 1.3 HTTP请求

```
POST https://api.openai.com/v1/chat/completions
Headers:
  Authorization: Bearer sk-xxx
  Content-Type: application/json

Body:
{
  "model": "gpt-3.5-turbo",
  "messages": [{"role": "user", "content": "你好"}]
}
```

---

## 🔰 2. 环境准备

### 2.1 获取API Key

1. 访问 https://platform.openai.com/
2. 注册账号
3. 在API Keys页面创建Key
4. **保存好Key，不要泄露！**

### 2.2 安装依赖

```bash
pip install openai
```

### 2.3 配置Key

```python
import openai

# 方式1：直接设置
openai.api_key = "sk-xxx"

# 方式2：从环境变量读取（推荐）
import os
openai.api_key = os.getenv("OPENAI_API_KEY")
```

---

## 🔰 3. Chat Completion API

### 3.1 基本调用

```python
from openai import OpenAI

client = OpenAI(api_key="sk-xxx")

# 基本调用
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "你好，请介绍一下自己"}
    ]
)

# 获取回复
print(response.choices[0].message.content)
```

### 3.2 消息角色

| 角色 | 说明 | 使用场景 |
|------|------|----------|
| system | 系统提示 | 设定AI的角色和行为 |
| user | 用户输入 | 用户的问题 |
| assistant | AI回复 | AI的回答 |
| function | 函数返回 | Function Calling的结果 |

### 3.3 多轮对话

```python
messages = [
    {"role": "system", "content": "你是一个专业的编程助手"},
    {"role": "user", "content": "什么是Python？"},
    {"role": "assistant", "content": "Python是一种解释型编程语言..."},
    {"role": "user", "content": "它有什么优点？"},
]

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
)
```

### 3.4 参数详解

```python
response = client.chat.completions.create(
    model="gpt-4",                    # 模型选择
    messages=messages,                 # 消息列表
    temperature=0.7,                   # 温度（0-2）
    max_tokens=1000,                   # 最大token数
    top_p=0.9,                         # 核采样
    frequency_penalty=0.0,             # 频率惩罚
    presence_penalty=0.0,              # 存在惩罚
    stream=False,                      # 是否流式输出
)
```

#### 参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| temperature | 随机性，越高越随机 | 0.7 |
| max_tokens | 最大输出长度 | 根据需求 |
| top_p | 核采样 | 0.9 |
| frequency_penalty | 降低重复词 | 0.0 |
| presence_penalty | 鼓励新话题 | 0.0 |

---

## 🔰 4. Function Calling

### 4.1 什么是Function Calling？

让AI**调用你定义的函数**，而不是直接回答问题。

### 4.2 生活类比

你问朋友："明天北京天气怎么样？"

- **普通回答**：根据记忆告诉你（可能不准）
- **Function Calling**：朋友说"我帮你查一下"，然后调用天气API，给你准确答案

### 4.3 定义函数

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如：北京"
                    }
                },
                "required": ["city"]
            }
        }
    }
]
```

### 4.4 调用API

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "北京今天天气怎么样？"}
    ],
    tools=tools,
    tool_choice="auto"  # 让AI决定是否调用函数
)

# 检查是否需要调用函数
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    
    print(f"调用函数：{function_name}")
    print(f"参数：{arguments}")
    
    # 执行函数
    if function_name == "get_weather":
        result = get_weather(arguments["city"])
        print(f"结果：{result}")
```

### 4.5 完整示例

```python
import json
from openai import OpenAI

client = OpenAI()

# 定义函数
def get_weather(city):
    """模拟获取天气"""
    return f"{city}今天晴，温度25°C"

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"}
                },
                "required": ["city"]
            }
        }
    }
]

# 对话
messages = [{"role": "user", "content": "北京和上海今天天气怎么样？"}]

response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=tools
)

# 处理函数调用
tool_calls = response.choices[0].message.tool_calls
if tool_calls:
    for tool_call in tool_calls:
        args = json.loads(tool_call.function.arguments)
        result = get_weather(args["city"])
        
        # 将结果添加到对话
        messages.append(response.choices[0].message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result
        })
    
    # 让AI总结
    final_response = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    print(final_response.choices[0].message.content)
```

---

## 🔰 5. 流式输出 (Streaming)

### 5.1 什么是流式输出？

AI**逐字输出**，而不是等全部生成完再返回。

### 5.2 生活类比

- **普通输出**：等饭菜全部做好再端上来
- **流式输出**：做好一个菜先上一个菜

### 5.3 代码示例

```python
# 流式输出
stream = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "写一首诗"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()  # 换行
```

### 5.4 异步流式

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def stream_chat():
    stream = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "你好"}],
        stream=True
    )
    
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()

asyncio.run(stream_chat())
```

---

## 🔰 6. 错误处理

### 6.1 常见错误

| 错误码 | 说明 | 解决方案 |
|--------|------|----------|
| 401 | API Key无效 | 检查Key是否正确 |
| 429 | 请求过多 | 等待或升级账户 |
| 500 | 服务器错误 | 稍后重试 |

### 6.2 错误处理代码

```python
from openai import OpenAI, APIError, RateLimitError

client = OpenAI()

try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "你好"}]
    )
    print(response.choices[0].message.content)

except RateLimitError as e:
    print("请求过多，请稍后重试")
    # 可以添加重试逻辑

except APIError as e:
    print(f"API错误：{e}")

except Exception as e:
    print(f"未知错误：{e}")
```

### 6.3 重试机制

```python
import time
from openai import OpenAI, RateLimitError

client = OpenAI()

def call_with_retry(messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            return response.choices[0].message.content
        except RateLimitError:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 指数退避
                print(f"等待{wait_time}秒后重试...")
                time.sleep(wait_time)
            else:
                raise
```

---

## 🔰 7. 实战：构建聊天机器人

### 7.1 完整代码

```python
from openai import OpenAI

class ChatBot:
    def __init__(self, model="gpt-3.5-turbo"):
        self.client = OpenAI()
        self.model = model
        self.messages = [
            {"role": "system", "content": "你是一个友好的助手"}
        ]
    
    def chat(self, user_input):
        """对话"""
        # 添加用户消息
        self.messages.append({"role": "user", "content": user_input})
        
        # 调用API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages
        )
        
        # 获取回复
        assistant_message = response.choices[0].message.content
        
        # 添加到历史
        self.messages.append({"role": "assistant", "content": assistant_message})
        
        return assistant_message
    
    def clear_history(self):
        """清空历史"""
        self.messages = [
            {"role": "system", "content": "你是一个友好的助手"}
        ]

# 使用
bot = ChatBot()
print(bot.chat("你好！"))
print(bot.chat("我叫张三"))
print(bot.chat("我叫什么名字？"))  # 能记住之前的对话
```

### 7.2 添加Function Calling

```python
import json

class SmartChatBot(ChatBot):
    def __init__(self, tools=None):
        super().__init__()
        self.tools = tools or []
    
    def chat(self, user_input):
        """智能对话，支持函数调用"""
        self.messages.append({"role": "user", "content": user_input})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.tools if self.tools else None
        )
        
        # 检查是否需要调用函数
        if response.choices[0].message.tool_calls:
            return self._handle_tool_calls(response)
        else:
            return self._handle_response(response)
    
    def _handle_tool_calls(self, response):
        """处理函数调用"""
        # 实现函数调用逻辑
        pass
    
    def _handle_response(self, response):
        """处理普通回复"""
        content = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": content})
        return content
```

---

## 📚 本节小结

| 概念 | 说明 |
|------|------|
| Chat Completion | 基本的对话API |
| 消息角色 | system/user/assistant |
| Function Calling | 让AI调用函数 |
| 流式输出 | 逐字返回结果 |
| 错误处理 | 处理API异常 |

---

## 🎯 下一步

- **11b - API调用：国内模型** - 通义千问、文心一言等
- **11c - API调用：本地部署** - Ollama、vLLM等

---

> 💡 **实践建议**：先用OpenAI API跑通基本流程，再尝试其他模型。
