---
module: "CV-深度学习笔记"
title: "使用LangChain构建RAG"
tags: [RAG, LangChain, ChromaDB]
---

# 使用LangChain构建RAG

# my_prompt.py

```python
'''提示词模板'''
# 导入 PromptTemplate 类，用于创建提示词模板（Prompt Template）
from langchain.prompts import PromptTemplate
# 使用字符串模板方式定义提示词模板，其中 {subject} 是一个可被替换的占位符
prompt_template = PromptTemplate.from_template('请告诉我一个关于{subject}的笑话')  # 实例化模板对象，定义模板格式   
# 使用模板格式化，替换 {subject} 为“程序员”，生成具体的提示词
prompt_a = prompt_template.format(subject='程序员')
# 替换 {subject} 为“汽车”，生成另一个具体提示词
prompt_b = prompt_template.format(subject='汽车')
# 替换 {subject} 为“国际”，生成第三个提示词
prompt_c = prompt_template.format(subject='国际')
# 输出生成的提示词：关于程序员的笑话
print(prompt_a)
# 输出生成的提示词：关于汽车的笑话
print(prompt_b)
# 输出生成的提示词：关于国际的笑话
print(prompt_c)
```

```
请告诉我一个关于程序员的笑话
请告诉我一个关于汽车的笑话
请告诉我一个关于国际的笑话

```

# simple_rag_v3.py

```python
# 导入 itemgetter，用于从数据结构中提取特定字段
from operator import itemgetter
# 导入 Path，用于处理文件路径
from pathlib import Path
# 导入提示词模板类
from langchain_core.prompts import PromptTemplate
# 导入输出解析器，将模型输出解析为字符串
from langchain_core.output_parsers import StrOutputParser
# 导入可组合的运行单元
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
# 导入文件夹加载器和文本文件加载器
from langchain_community.document_loaders import DirectoryLoader, TextLoader
# 导入文本切分器，用于对文档进行分块
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
# 导入 HuggingFace 向量嵌入模型
from langchain_huggingface import HuggingFaceEmbeddings
# 导入 OpenAI 的聊天模型（兼容 DeepSeek 接口）
from langchain_openai import ChatOpenAI
# 导入 Chroma 向量数据库
from langchain_chroma import Chroma
# 再次导入提示词模板类（可省略，已在前面导入）
from langchain.prompts import PromptTemplate
# 导入 os 模块，用于读取系统环境变量
import os

# 从环境变量中获取名为 DEEPSEEK_API_KEY 的密钥（你需提前设置这个环境变量）
api_key = os.environ['DEEPSEEK_API_KEY']

# 实例化对话语言模型（大语言模型），使用 deepseek-chat，传入 API 密钥
llm = ChatOpenAI(model='deepseek-chat', base_url='https://api.deepseek.com', api_key=api_key)

# 实例化向量嵌入模型，用于将文本转换为向量，路径必须为本地 HuggingFace 模型路径
embedding_model = HuggingFaceEmbeddings(model=r'model_dir\BAAI\bge-large-zh-v1___5')

# 指定包含知识文本的本地文件夹路径
file_dir = Path('my_knowledge')

# 文本切分器：将文档按 500 字为一块进行切分，重叠部分为 100 字
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# 创建 Chroma 向量数据库对象，指定嵌入模型和保存位置
vector_store = Chroma(embedding_function=embedding_model, persist_directory='./chroma_v3')

# 设置检索器，用于从向量数据库中按相似度取出前 k 个文档（默认 5 个）
retriever = vector_store.as_retriever(search_kwargs={'k': 5})

# 创建提示词模板：{context} 和 {question} 是可替换的占位符
prompt_template = PromptTemplate.from_template("""
你是一个严谨的RAG助手。
请根据以下提供的上下文信息来回答问题。
如果上下文信息不足以回答问题，请直接说“根据提供的信息无法回答”
如果回答时使用了上下文中的信息，在回答后输出使用了哪些上下文，
上下文信息:
{context}
-------------
问题：{question}""")

# 创建一个链（Chain），包含以下步骤：
# 1. 接收用户输入的 'question'；
# 2. 将 question 传入 retriever，获取相关 context；
# 3. 使用 prompt_template 构造完整提示词；
# 4. 使用 LLM 生成回答；
# 5. 使用 StrOutputParser 解析输出为字符串。
chain = {
    'question': RunnablePassthrough()  # 原样传递 question
} | RunnablePassthrough.assign(        # 增加 context 字段（从 retriever 中检索 question 相关内容）
    context=itemgetter('question') | retriever
) | prompt_template | llm | StrOutputParser()  # 构造 prompt、调用模型、解析输出

# 执行主程序
if __name__ == '__main__':
    # 使用 DirectoryLoader 加载 my_knowledge 目录中的所有文本文件
    # TextLoader 用于以 UTF-8 编码方式读取每个文件
    docs = DirectoryLoader(str(file_dir),
        loader_cls=lambda path: TextLoader(path, encoding='utf-8')
    ).load()

    # 将加载的文档进行切分，分成多个小段
    docs = text_splitter.split_documents(docs)

    # 将切分好的文档添加到向量数据库中进行存储
    vector_store.add_documents(docs)

    # 测试链条：输入一个问题“能飞多久？”，并打印模型的回答
    print(chain.invoke('能飞多久？'))
```

```
D:\11_Anaconda\envs\RAG\Lib\site-packages\tqdm\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm
Failed to send telemetry event ClientStartEvent: capture() takes 1 positional argument but 3 were given
Failed to send telemetry event ClientCreateCollectionEvent: capture() takes 1 positional argument but 3 were given
Failed to send telemetry event CollectionQueryEvent: capture() takes 1 positional argument but 3 were given

```

```
探索者X100型无人机的电池续航时间长达30分钟。

使用的上下文：
- Document(id='a929d46e-9400-4ce4-bbaa-6bf9a2fc2d27', ...) 中关于电池续航时间的信息

```