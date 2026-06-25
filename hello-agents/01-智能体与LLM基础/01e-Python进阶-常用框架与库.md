---
module: "hello-agents"
title: "Python进阶：常用框架与库"
description: "数据处理、Web开发、测试、异步并发等常用框架速查"
tags: [Python, Advanced, Frameworks, Pandas, FastAPI, pytest]
---

# 🏗️ Python进阶：常用框架与库

> **工欲善其事，必先利其器**

---

## 📝 前言

在 [[01c-Python基础-文件操作与异常处理]] 中我们学习了基础文件操作。本章介绍 Python 生态中最常用的框架和库，帮助你快速搭建项目：

- **数据处理** — pandas / numpy / polars
- **Web 开发** — FastAPI / Django / Flask
- **测试** — pytest
- **异步与并发** — asyncio / ThreadPoolExecutor
- **实用工具** — pydantic / loguru / rich 等

---

## 📊 1. 数据处理

### 1.1 pandas — 数据分析利器

```python
import pandas as pd

# 读取数据
df = pd.read_csv("data.csv")           # CSV
df = pd.read_excel("data.xlsx")        # Excel
df = pd.read_json("data.json")         # JSON

# 基本探索
df.head()           # 前 5 行
df.info()           # 数据类型、非空统计
df.describe()       # 数值列统计摘要
df.shape            # (行数, 列数)
df.columns          # 列名

# 选择与筛选
df["age"]                    # 单列
df[["name", "age"]]          # 多列
df[df["age"] > 25]           # 条件筛选
df.query("age > 25 and city == '北京'")  # 查询语法

# 分组聚合
df.groupby("city")["age"].mean()     # 各城市平均年龄
df.groupby("city").agg({
    "age": "mean",
    "salary": ["min", "max"]
})

# 保存
df.to_csv("output.csv", index=False)
df.to_excel("output.xlsx", index=False)
```

### 1.2 numpy — 数值计算基础

```python
import numpy as np

# 创建数组
arr = np.array([1, 2, 3, 4, 5])
zeros = np.zeros((3, 4))        # 3x4 全零矩阵
ones = np.ones((2, 3))          # 2x3 全一矩阵
eye = np.eye(4)                 # 4x4 单位矩阵
rand = np.random.randn(3, 3)    # 3x3 标准正态分布

# 运算
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

a + b           # [5, 7, 9]    — 逐元素加法
a * b           # [4, 10, 18]  — 逐元素乘法
np.dot(a, b)    # 32           — 点积
a @ b           # 32           — 矩阵乘法（Python 3.5+）

# 常用函数
np.sum(arr)         # 求和
np.mean(arr)        # 均值
np.std(arr)         # 标准差
np.max(arr)         # 最大值
np.where(arr > 2, arr, 0)  # 条件替换
```

### 1.3 polars — 高性能 DataFrame

```python
import polars as pl

# 读取（比 pandas 快 10-100x）
df = pl.read_csv("data.csv")

# 操作
result = (
    df.lazy()
    .filter(pl.col("age") > 25)
    .group_by("city")
    .agg([
        pl.col("salary").mean().alias("avg_salary"),
        pl.col("name").count().alias("count"),
    ])
    .collect()
)

print(result)
```

---

## 🌐 2. Web 开发

### 2.1 FastAPI（推荐用于 AI Agent）

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Agent API")

# 数据模型
class Message(BaseModel):
    role: str
    content: str

class ChatResponse(BaseModel):
    reply: str
    tokens_used: int

# 路由
@app.get("/")
def root():
    return {"message": "Hello, Agent!"}

@app.post("/chat", response_model=ChatResponse)
def chat(msg: Message):
    if not msg.content:
        raise HTTPException(status_code=400, detail="消息不能为空")
    return ChatResponse(
        reply=f"你说: {msg.content}",
        tokens_used=len(msg.content)
    )

@app.get("/users/{user_id}")
def get_user(user_id: int):
    return {"user_id": user_id, "name": "张三"}

# 启动: uvicorn main:app --reload
# 自动文档: http://localhost:8000/docs
```

### 2.2 Django vs Flask vs FastAPI

| 特性 | FastAPI | Django | Flask |
|------|---------|--------|-------|
| 异步支持 | ✅ 原生 | ⚠️ 部分 | ❌ 需扩展 |
| 性能 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 自动文档 | ✅ OpenAPI | ❌ 需 drf-spectacular | ❌ 需 flask-apispec |
| ORM | 无（用 SQLAlchemy） | ✅ Django ORM | 无 |
| 适合场景 | API、微服务 | 大型全栈应用 | 小型应用、原型 |
| 学习曲线 | 中等 | 较陡 | 简单 |

---

## 🧪 3. 测试框架

### 3.1 pytest 基础

```python
# test_calculator.py
import pytest

def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

# 基本测试
def test_add():
    assert add(1, 2) == 3

def test_subtract():
    assert subtract(5, 3) == 2

# 测试异常
def test_divide_by_zero():
    with pytest.raises(ZeroDivisionError):
        1 / 0

# 运行: pytest test_calculator.py -v
```

### 3.2 参数化测试

```python
import pytest

def add(a, b):
    return a + b

@pytest.mark.parametrize("a,b,expected", [
    (1, 2, 3),
    (-1, 1, 0),
    (0, 0, 0),
    (100, 200, 300),
])
def test_add_parametrized(a, b, expected):
    assert add(a, b) == expected
# 一次运行 4 个测试用例
```

### 3.3 fixture（测试夹具）

```python
import pytest

@pytest.fixture
def sample_user():
    """提供测试用户数据"""
    return {
        "name": "张三",
        "age": 25,
        "email": "test@example.com"
    }

@pytest.fixture
def temp_file(tmp_path):
    """创建临时文件"""
    file = tmp_path / "test.txt"
    file.write_text("hello")
    return file

def test_user_name(sample_user):
    assert sample_user["name"] == "张三"

def test_file_content(temp_file):
    assert temp_file.read_text() == "hello"
```

### 3.4 异步测试

```python
import pytest
import asyncio

async def fetch_data(url):
    await asyncio.sleep(0.1)  # 模拟异步操作
    return {"url": url}

@pytest.mark.asyncio
async def test_fetch_data():
    result = await fetch_data("https://example.com")
    assert result["url"] == "https://example.com"

# 需要: pip install pytest-asyncio
```

---

## ⚡ 4. 异步与并发

### 4.1 asyncio 协程

```python
import asyncio

async def say_hello(name, delay):
    await asyncio.sleep(delay)
    print(f"Hello, {name}!")

async def main():
    # 并发执行（总耗时 = 最长的 delay）
    await asyncio.gather(
        say_hello("Alice", 1),
        say_hello("Bob", 0.5),
        say_hello("Charlie", 1.5),
    )

asyncio.run(main())  # 总耗时约 1.5 秒
```

### 4.2 ThreadPoolExecutor（I/O 密集型）

```python
import time
from concurrent.futures import ThreadPoolExecutor

def slow_io_task(n):
    """模拟 I/O 操作"""
    time.sleep(1)
    return n * 2

# 串行执行（10 个任务 = 10 秒）
results = [slow_io_task(i) for i in range(10)]

# 并行执行（10 个任务 ≈ 2 秒）
with ThreadPoolExecutor(max_workers=5) as pool:
    results = list(pool.map(slow_io_task, range(10)))
```

### 4.3 ProcessPoolExecutor（CPU 密集型）

```python
from concurrent.futures import ProcessPoolExecutor

def heavy_compute(n):
    """CPU 密集计算"""
    return sum(i * i for i in range(n))

# 4 个进程并行计算
with ProcessPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(heavy_compute, [10**6] * 4))
```

---

## 🛠️ 5. 实用工具库速查

| 库 | 用途 | 一句话介绍 |
|------|------|----------|
| `pydantic` | 数据验证 | 类型注解自动校验 |
| `rich` | 终端美化 | 彩色输出、进度条、表格 |
| `loguru` | 日志 | 替代 logging，更简洁 |
| `click` / `typer` | CLI 工具 | 快速构建命令行 |
| `httpx` | HTTP 客户端 | 支持同步和异步 |
| `tenacity` | 重试机制 | 装饰器实现自动重试 |
| `pyyaml` / `toml` | 配置文件 | YAML/TOML 解析 |
| `python-dotenv` | 环境变量 | 从 .env 加载配置 |
| `requests` | HTTP 客户端 | 最流行的同步 HTTP 库 |

### 5.1 pydantic — 数据验证

```python
from pydantic import BaseModel, Field, ValidationError
from typing import Optional

class User(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    age: int = Field(..., ge=0, le=150)
    email: str
    bio: Optional[str] = None

# ✅ 正确
user = User(name="张三", age=25, email="test@example.com")
print(user.dict())

# ❌ 自动校验失败
try:
    bad_user = User(name="", age=-1, email="bad")
except ValidationError as e:
    print(e.errors())
```

### 5.2 loguru — 简洁日志

```python
from loguru import logger

# 基本使用（比 logging 简单 10 倍）
logger.info("应用启动")
logger.warning("磁盘空间不足")
logger.error("文件不存在: {path}", path="/data.txt")
logger.critical("数据库连接失败")

# 添加日志文件（自动轮转）
logger.add(
    "app.log",
    rotation="10 MB",      # 超过 10MB 轮转
    retention="7 days",    # 保留 7 天
    compression="zip",     # 压缩旧日志
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)
```

### 5.3 rich — 终端美化

```python
from rich import print, console
from rich.table import Table
from rich.progress import track
import time

# 彩色输出
print("[bold green]成功[/bold green]")
print("[red]错误: 文件不存在[/red]")

# 表格
table = Table(title="用户列表")
table.add_column("姓名", style="cyan")
table.add_column("年龄", style="magenta")
table.add_row("张三", "25")
table.add_row("李四", "30")
console.print(table)

# 进度条
for i in track(range(100), description="处理中..."):
    time.sleep(0.01)
```

### 5.4 tenacity — 自动重试

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),           # 最多重试 3 次
    wait=wait_exponential(min=1, max=10), # 指数退避
)
def fetch_data(url):
    import random
    if random.random() < 0.7:
        raise ConnectionError("网络错误")
    return {"data": "ok"}

# 自动重试，失败后指数退避
result = fetch_data("https://api.example.com")
```

---

## 📚 本节小结

| 领域 | 推荐库 | 核心特点 |
|------|--------|----------|
| 数据处理 | pandas / polars | DataFrame 操作 |
| 数值计算 | numpy | 矩阵运算 |
| Web API | FastAPI | 异步、自动文档 |
| 测试 | pytest | 简洁、强大 |
| 日志 | loguru | 一行代码搞定 |
| 数据验证 | pydantic | 类型注解校验 |
| 终端美化 | rich | 彩色输出、表格 |
| 重试 | tenacity | 装饰器重试 |

---

## 🎯 下一步

- [[01f-Python进阶-设计模式]] - 单例、工厂、观察者等设计模式
- [[01g-Python进阶-编码风格与最佳实践]] - PEP 8、类型注解、Pythonic 写法

---

> 💡 **实践建议**：
> 1. 用 FastAPI 写一个简单的 REST API
> 2. 用 pytest 为你的代码编写测试
> 3. 用 loguru 替代 print 调试
> 4. 用 pydantic 验证用户输入
