---
module: "CV-深度学习笔记"
title: "PythonAPI环境安装"
tags: [LLM, DeepSeek, 环境配置]
---

# PythonAPI环境安装

# 01.基本示例.py

```python
"""角色：
        用户:提出问题，请求，并期望得到正确的回应、
        大模型：处理用户问题，给出回应、
        知识库:收录海量资料，根据用户问题检索关联信息
"""

# 从 openai 库中导入 OpenAI 客户端类
from openai import OpenAI
# 导入 os 模块，用于访问环境变量
import os

# 设置环境变量，从操作系统中获取名为 DEEPSEEK_API_KEY 的 API 密钥
# 该密钥需事先在操作系统中配置，例如通过 `os.environ['DEEPSEEK_API_KEY'] = 'your_key'` 或终端设置        
api_key = os.environ['DEEPSEEK_API_KEY']  

#创建一个 OpenAI 客户端对象，指定使用 DeepSeek 的 API 网关地址 和 API 密钥
client = OpenAI(base_url = 'https://api.deepseek.com', api_key = api_key) 
# client = OpenAI(base_url = 'https://api.deepseek.com', api_key = "sk-9340d89277094d5f8d55c296f66513c2") 

#发送聊天请求
completion = client.chat.completions.create(    
    model='deepseek-reasoner',   # 指定使用的聊天模型为 deepseek-reasoner
    messages=[  # 设置对话内容列表
        {
            'role': 'user',  # 角色为用户
            'content': "你是谁"  # 用户提问的内容
        }
    ],
)

# 输出模型返回的回答内容
print(completion.choices[0].message.content)
```

```
你好呀！😊我是 **DeepSeek-R1**，由中国的「深度求索」公司研发的智能助手。你可以把我当作一个知识丰富、乐于助人的聊天伙伴～无论你是想查资料、写文章、解题、翻译、聊聊天，还是处理文件（支持上传 Word、PDF、Excel 等），我都能帮上忙！

✨目前我是**免费的**，也没有语音功能（纯文字交流），但支持超长上下文（最多128K），可以理解复杂的内容和逻辑。如果你有任何问题，尽管问我，我一直都在！

那现在，有什么我可以为你做的吗？🌟

```