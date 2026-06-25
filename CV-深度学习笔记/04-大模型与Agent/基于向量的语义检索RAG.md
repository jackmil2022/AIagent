---
module: "CV-深度学习笔记"
title: "基于向量的语义检索RAG"
tags: [CV, Deep-Learning, PyTorch]
---

# 基于向量的语义检索RAG

# fetch_model.py

```python
# 一次性，进行预训练模型下载
# 从 ModelScope 平台导入 snapshot_download 方法，用于下载模型到本地
from modelscope import snapshot_download

# 使用 snapshot_download 方法从 ModelScope 平台下载指定的模型文件，并将其缓存在指定目录中
modelscope = snapshot_download(
    # 指定要下载的模型 ID（即模型名称），此处为 BAAI 发布的中文大模型
    model_id = 'BAAI/bge-large-zh-v1.5',
    # 指定模型下载到的本地缓存目录，使用原始字符串以避免路径中的反斜杠出错
    cache_dir = r'model_dir'
)

# 打印模型路径验证实际保存位置
print(modelscope)
```

```
D:\11_Anaconda\envs\RAG\Lib\site-packages\tqdm\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm

```

```
Downloading Model from https://www.modelscope.cn to directory: model_dir\BAAI\bge-large-zh-v1.5

```

```
2025-06-30 12:55:16,668 - modelscope - INFO - Creating symbolic link [model_dir\BAAI\bge-large-zh-v1.5].
2025-06-30 12:55:16,669 - modelscope - WARNING - Failed to create symbolic link model_dir\BAAI\bge-large-zh-v1.5 for C:\Users\wangy\Desktop\AgentNew\model_dir\BAAI\bge-large-zh-v1___5.

```

```
model_dir\BAAI\bge-large-zh-v1___5

```

# file_db.py

```python
# 引入标准库和第三方库
# 用于跨平台地处理文件路径（如查找文件、拼接路径）
from pathlib import Path
# 用于加载句子嵌入模型并生成句子向量
from sentence_transformers import SentenceTransformer
# Chroma 是一个轻量级的向量数据库，支持相似度检索等操作
import chromadb  
# 用于获取当前时间，常用于记录创建时间等元信息
from datetime import datetime

# 加载本地已经下载好的句子嵌入模型
# 模型路径是手动从 ModelScope 下载并保存到本地的中文 BGE 模型
# 注意路径中有 "___" 替代了 "."，这是 ModelScope 的默认命名方式
model = SentenceTransformer(r'model_dir\BAAI\bge-large-zh-v1___5')  

# 创建一个 Chroma 的数据库客户端，用于向量存储和检索
# 使用 PersistentClient 表示采用“文件持久化模式”，数据将存储在指定目录
# 如果使用 chromadb.Client()，那就是内存模式，程序结束后数据就消失了
# 数据库存储在当前目录下的 chroma_v2 文件夹中
client = chromadb.PersistentClient('./chroma_v2')

# 从数据库中获取一个名为 "da_fei" 的集合（collection），如果不存在就自动创建
# 每个 collection 就像一个“表”，用来存储一组相似用途的向量数据
collecton = client.get_or_create_collection(
    name = 'da_fei',  # 集合的名称，唯一标识
    metadata={   # 集合的元数据，用于描述集合的信息
        "介绍": "文本文件的向量数据库",   # 自定义描述，说明这个集合用于保存哪些数据
        "创建时间": str(datetime.now()),  # 获取当前时间作为创建时间，转为字符串格式
        "hnsw:space": "cosine" # 设置检索时采用的距离度量方式为 cosine（余弦距离）
        # cosine 距离 = 1 - 余弦相似度，值越小，表示越相似
    }
)

# 定义一个函数：将指定目录（my_knowledge）下的所有 .txt 文件读取、编码，并保存到向量数据库中
def txt_2db():
    # 第一步：加载所有文本文件
    # 使用 pathlib 遍历 my_knowledge 目录下所有以 .txt 结尾的文件，生成路径列表
    path_list = list(Path('my_knowledge').glob('*.txt'))
    # 用于存储所有读取到的文本内容
    text_list = []  

    # 遍历每个文本文件路径
    for path in path_list:
        # 使用 UTF-8 编码读取文本内容，避免中文乱码
        text = path.read_text(encoding='utf-8')  
        # 将每个文件的文本加入列表中
        text_list.append(text)

    # 第二步：将所有文本内容进行句向量嵌入
    # encode() 方法将文本列表转换为向量（嵌入），返回一个 numpy 数组（二维矩阵：每行一个文本向量）
    embeddings = model.encode(text_list)  # shape=(文件数量, 向量维度)，如 (10, 1024)

    # 第三步：将嵌入向量以及原始文本、元数据、ID 存入向量数据库中
    collecton.add(
        embeddings=embeddings.tolist(), # 将 numpy 数组转换为普通 Python 列表（Chroma 要求的格式）
        documents=text_list, # 原始的文本内容列表（与向量一一对应）
        metadatas=[{'id':i} for i in text_list],  # 创建一个元数据列表，每个元素是一个字典（这里只存了 id，可扩展）
        ids=[f'doc_{i}' for i,_ in enumerate(text_list)], # 每个文本对应一个唯一的字符串 ID，如 doc_0、doc_1……
        # 注意：ID 必须是字符串类型，不能是字典或整数，而且不能重复，否则会报错
    )

    # 打印数据库中存储的向量条数，用于验证是否成功写入
    print(f'数据库中的数据量:{collecton.count()}')

# 主程序入口，只有直接运行当前文件时才会执行下面的代码
if __name__ == '__main__':
    # 第一次运行建议先执行 txt_2db()，将所有文本编码后写入数据库
    # 如果数据库已存在，可以注释掉避免重复添加
    txt_2db()

    # 构造一个查询语句，模拟用户的问题（可任意改成你想查的内容）
#     query = ['能飞多久']
    query = ['现在要做什么']

    # 将查询语句转换为嵌入向量，用于相似度检索
    query_embedding = model.encode(query)

    # 使用数据库提供的 query 接口，查找与 query_embedding 最相似的前 5 条文本
    data = collecton.query(query_embedding.tolist(), n_results=5)
    print(data)

    # Chroma 返回的是一个字典，里面包含多个字段，如：
    # 'documents': 文本内容；'distances': 相似度得分；'ids': 文本 ID；'metadatas': 元数据
    # 注意：余弦距离 = 1 - 相似度，距离越小越相似

    # 从返回的结果中提取文档内容（documents）部分，取第一个查询结果的 top5 文本列表
    text_list = data['documents'][0]

    # 遍历相似文本列表并逐条打印
    for t in text_list:
        # 输出与查询语句最相关的文本
        print(t)
```

```
Failed to send telemetry event ClientStartEvent: capture() takes 1 positional argument but 3 were given
Failed to send telemetry event ClientCreateCollectionEvent: capture() takes 1 positional argument but 3 were given
Failed to send telemetry event CollectionAddEvent: capture() takes 1 positional argument but 3 were given
Failed to send telemetry event CollectionQueryEvent: capture() takes 1 positional argument but 3 were given

```

```
数据库中的数据量:2
{'ids': [['doc_0', 'doc_1']], 'embeddings': None, 'documents': [['2023.12.1《从零到一：自己动手构建RAG智能问答系统》课程设计完成，一共5节课\n2024.1.1 第一节课备课完成\n2024.1.2 第二节课备课充成\n2024.1.3 第三节课备课完成\n2024.1.4 今天休息', '探索者X100型无人机是一款专为户外爱好者设计的先进航拍设备。\n它配备了4K超高清摄像头，支持最远5公里的图像传输距离。\n探索者X100拥有智能跟拍、轨迹飞行和一键返航等多种飞行模式。\n电池续航时间长达30分钟，抗风等级达到5级。\n如需技术支持或咨询购买，请访问官网 explorer-drones.com 或致电400-123-4567。\n探索者X100型无人机在2024年春季发布。']], 'uris': None, 'included': ['metadatas', 'documents', 'distances'], 'data': None, 'metadatas': [[{'id': '2023.12.1《从零到一：自己动手构建RAG智能问答系统》课程设计完成，一共5节课\n2024.1.1 第一节课备课完成\n2024.1.2 第二节课备课充成\n2024.1.3 第三节课备课完成\n2024.1.4 今天休息'}, {'id': '探索者X100型无人机是一款专为户外爱好者设计的先进航拍设备。\n它配备了4K超高清摄像头，支持最远5公里的图像传输距离。\n探索者X100拥有智能跟拍、轨迹飞行和一键返航等多种飞行模式。\n电池续航时间长达30分钟，抗风等级达到5级。\n如需技术支持或咨询购买，请访问官网 explorer-drones.com 或致电400-123-4567。\n探索者X100型无人机在2024年春季发布。'}]], 'distances': [[0.5993373394012451, 0.7534679174423218]]}
2023.12.1《从零到一：自己动手构建RAG智能问答系统》课程设计完成，一共5节课
2024.1.1 第一节课备课完成
2024.1.2 第二节课备课充成
2024.1.3 第三节课备课完成
2024.1.4 今天休息
探索者X100型无人机是一款专为户外爱好者设计的先进航拍设备。
它配备了4K超高清摄像头，支持最远5公里的图像传输距离。
探索者X100拥有智能跟拍、轨迹飞行和一键返航等多种飞行模式。
电池续航时间长达30分钟，抗风等级达到5级。
如需技术支持或咨询购买，请访问官网 explorer-drones.com 或致电400-123-4567。
探索者X100型无人机在2024年春季发布。

```