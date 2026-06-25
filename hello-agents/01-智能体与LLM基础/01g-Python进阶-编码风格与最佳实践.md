---
module: "hello-agents"
title: "Python进阶：编码风格与最佳实践"
description: "PEP 8、类型注解、Pythonic 写法、错误处理、项目结构"
tags: [Python, Advanced, PEP8, TypeHints, Best-Practices]
---

# 📏 Python进阶：编码风格与最佳实践

> **写出别人看得懂、自己不头疼的代码**

---

## 📝 前言

好的代码不只是能跑，还要：

- **可读** — 别人能看懂，三个月后的自己也能看懂
- **可维护** — 改一处不会崩一片
- **Pythonic** — 用 Python 的方式写 Python

本章介绍 Python 社区公认的最佳实践。

---

## 📐 1. PEP 8 核心规范

PEP 8 是 Python 的官方编码风格指南。

### 1.1 缩进与空格

```python
# ✅ 缩进：4 个空格
def greet(name):
    if name:
        print(f"Hello, {name}")

# ❌ 不要用 Tab 或混合
```

### 1.2 行长度

```python
# ✅ 代码行 ≤ 79 字符，文档字符串 ≤ 72 字符

# 长行用括号换行
result = (first_variable + second_variable
          + third_variable)

# 或用反斜杠（不推荐）
result = first_variable + \
         second_variable

# 函数参数换行
def long_function_name(
        arg1, arg2, arg3,
        arg4, arg5):
    pass
```

### 1.3 命名约定

```python
# 变量、函数、方法：小写下划线
user_name = "张三"
def get_user_name(): ...

# 类名：大驼峰（PascalCase）
class UserProfile: ...

# 常量：全大写
MAX_RETRY = 3
DATABASE_URL = "sqlite:///app.db"

# 私有：前缀下划线
_internal_data = {}
def _helper(): ...

# 模块级：简短、全小写
# my_module.py
```

### 1.4 空行

```python
# 类之间空 2 行
class FirstClass:
    pass


class SecondClass:
    pass

# 方法之间空 1 行
class MyClass:
    def method_one(self):
        pass

    def method_two(self):
        pass

# 逻辑块之间空 1 行
def process():
    # 第一步
    step_one()

    # 第二步
    step_two()
```

### 1.5 导入顺序

```python
# 1. 标准库
import os
import sys
from datetime import datetime

# 2. 第三方库
import requests
import pandas as pd

# 3. 本地模块
from my_module import my_function
from . import local_module

# 每组之间空一行
```

### 1.6 字符串引号

```python
# 统一风格即可，推荐双引号
name = "张三"
message = 'Hello'

# 三引号用于多行/文档字符串
docstring = """
这是文档字符串
可以跨多行
"""
```

---

## 🏷️ 2. 类型注解（Type Hints）

类型注解让代码更清晰，IDE 能自动补全和检查。

### 2.1 基本类型

```python
from typing import Optional, List, Dict, Tuple, Union, Any

# 函数注解
def greet(name: str) -> str:
    return f"Hello, {name}"

def process(
    items: List[str],
    config: Dict[str, Any],
    timeout: float = 30.0,
    callback: Optional[callable] = None,
) -> bool:
    return True

# 返回类型
def get_user(user_id: int) -> Dict[str, str]:
    return {"name": "张三"}
```

### 2.2 dataclass（Python 3.7+）

```python
from dataclasses import dataclass, field

@dataclass
class User:
    name: str
    age: int
    email: str
    is_active: bool = True
    tags: list = field(default_factory=list)

    def greet(self) -> str:
        return f"我是{self.name}，{self.age}岁"

# 自动生成 __init__, __repr__, __eq__ 等
user = User(name="张三", age=25, email="test@example.com")
print(user)  # User(name='张三', age=25, email='test@example.com', ...)
```

### 2.3 类型别名

```python
from typing import List, Tuple

# 复杂类型起个别名
Vector = List[float]
Matrix = List[Vector]
Point = Tuple[float, float]

def dot_product(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))

def distance(p1: Point, p2: Point) -> float:
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5
```

### 2.4 Literal 与 TypedDict

```python
from typing import Literal, TypedDict

# Literal: 限定可选值
def set_mode(mode: Literal["read", "write", "append"]):
    pass

# TypedDict: 字典结构化
class UserDict(TypedDict):
    name: str
    age: int
    email: str

def process_user(user: UserDict) -> str:
    return user["name"]
```

---

## 🐍 3. Pythonic 写法

用 Python 的方式写代码，简洁、优雅、高效。

### 3.1 推导式

```python
# 列表推导式（替代 map/filter）
squares = [x**2 for x in range(10)]
evens = [x for x in range(10) if x % 2 == 0]

# 字典推导式
word_lengths = {word: len(word) for word in ["hello", "world"]}

# 集合推导式
unique_chars = {char for char in "hello world"}

# 生成器表达式（内存友好）
total = sum(x**2 for x in range(1000000))
```

### 3.2 解包

```python
# 多重赋值
first, *middle, last = [1, 2, 3, 4, 5]
# first=1, middle=[2,3,4], last=5

# 交换变量
a, b = b, a

# 忽略不需要的值
_, name, _, age = (1, "张三", 25, "北京")
```

### 3.3 enumerate 与 zip

```python
names = ["Alice", "Bob", "Charlie"]

# ✅ 用 enumerate
for i, name in enumerate(names):
    print(f"{i}: {name}")

# ❌ 不要这样
for i in range(len(names)):
    print(f"{i}: {names[i]}")

# 并行迭代
names = ["Alice", "Bob"]
ages = [25, 30]
for name, age in zip(names, ages):
    print(f"{name} is {age}")
```

### 3.4 any / all

```python
users = [{"role": "user"}, {"role": "admin"}, {"role": "user"}]

# 是否有任何管理员
has_admin = any(user["role"] == "admin" for user in users)

# 是否全部激活
all_active = all(user.get("active", False) for user in users)
```

### 3.5 Walrus Operator（:=，Python 3.8+）

```python
data = [1, 2, 3, 4, 5]

# 赋值并判断
if (n := len(data)) > 3:
    print(f"列表有 {n} 个元素")

# 循环中使用
while chunk := file.read(8192):
    process(chunk)
```

### 3.6 上下文管理器的隐藏用法

```python
from contextlib import nullcontext

# 条件性上下文管理
with open("file.txt") if use_file else nullcontext() as f:
    if f:
        data = f.read()
```

---

## ⚠️ 4. 错误处理最佳实践

### 4.1 捕获具体异常

```python
# ❌ 反面教材
try:
    do_something()
except:
    pass  # 吞掉所有异常，bug 永远找不到

# ✅ 正确做法
try:
    result = data["key"]
except KeyError:
    result = default_value
except ValueError as e:
    logger.error(f"值错误: {e}")
    raise  # 重新抛出
```

### 4.2 自定义异常层级

```python
# 应用基础异常
class AppError(Exception):
    """所有应用异常的基类"""
    pass

class ValidationError(AppError):
    """数据验证错误"""
    pass

class NotFoundError(AppError):
    """资源未找到"""
    pass

class PermissionError(AppError):
    """权限不足"""
    pass

# 使用
try:
    user = get_user(user_id)
except NotFoundError:
    return {"error": "用户不存在"}
except PermissionError:
    return {"error": "无权访问"}
except AppError as e:
    logger.exception("应用错误")
    return {"error": "服务器内部错误"}
```

### 4.3 异常链（保留原始信息）

```python
class DatabaseError(Exception):
    pass

class UserService:
    def get_user(self, user_id: int):
        try:
            return db.query(user_id)
        except DatabaseError as e:
            # from e 保留原始异常信息
            raise NotFoundError(f"用户 {user_id} 不存在") from e
```

### 4.4 使用 else 避免不必要的 try

```python
# ✅ else 只在没有异常时执行
try:
    file = open("config.json")
except FileNotFoundError:
    config = {}
else:
    # 文件成功打开后才执行
    config = json.load(file)
    file.close()
```

### 4.5 日志记录异常

```python
import logging

logger = logging.getLogger(__name__)

try:
    risky_operation()
except Exception:
    logger.exception("操作失败")  # 自动记录堆栈
    raise  # 继续抛出
```

---

## 📁 5. 项目结构

### 5.1 推荐结构

```
my_project/
├── src/                    # 源代码
│   ├── __init__.py
│   ├── main.py            # 入口
│   ├── config.py          # 配置
│   ├── models/            # 数据模型
│   │   ├── __init__.py
│   │   └── user.py
│   ├── services/          # 业务逻辑
│   │   ├── __init__.py
│   │   └── user_service.py
│   ├── utils/             # 工具函数
│   │   ├── __init__.py
│   │   └── helpers.py
│   └── exceptions.py      # 自定义异常
├── tests/                  # 测试
│   ├── __init__.py
│   ├── test_models/
│   └── test_services/
├── docs/                   # 文档
├── pyproject.toml          # 项目配置
├── README.md
├── .env.example            # 环境变量示例
└── .gitignore
```

### 5.2 pyproject.toml 示例

```toml
[project]
name = "my-project"
version = "0.1.0"
description = "我的项目"
requires-python = ">=3.9"
dependencies = [
    "fastapi>=0.100.0",
    "pydantic>=2.0.0",
    "loguru>=0.7.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "ruff>=0.1.0",
]

[tool.ruff]
line-length = 88
target-version = "py39"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP"]
```

---

## 🔧 6. 常用工具

### 6.1 linter / formatter

| 工具       | 用途                   | 命令                               |
| -------- | -------------------- | -------------------------------- |
| `ruff`   | Lint + Format（推荐，极快） | `ruff check .` / `ruff format .` |
| `black`  | 代码格式化                | `black .`                        |
| `flake8` | Lint                 | `flake8 .`                       |
| `mypy`   | 类型检查                 | `mypy .`                         |

### 6.2 推荐配置

```bash
# 安装 ruff（替代 black + flake8 + isort）
pip install ruff

# 检查
ruff check .

# 自动修复
ruff check --fix .

# 格式化
ruff format .
```

---

## 📚 本节小结

| 主题 | 核心要点 |
|------|----------|
| PEP 8 | 4空格缩进、79字符、命名约定 |
| 类型注解 | 函数参数/返回值、dataclass |
| Pythonic | 推导式、解包、enumerate、walrus |
| 错误处理 | 具体异常、异常链、else |
| 项目结构 | src/tests/docs 分离 |
| 工具 | ruff（lint+format）、mypy（类型） |

---

## 🎯 下一步

- [[01f-Python进阶-设计模式]] - 设计模式详解
- [[01e-Python进阶-常用框架与库]] - 常用框架速查

---

> 💡 **实践建议**：
> 1. 给你的项目配置 ruff，养成自动格式化的习惯
> 2. 用 dataclass 重写你的数据类
> 3. 给所有公开函数加上类型注解
> 4. 用自定义异常替代 ValueError / RuntimeError
