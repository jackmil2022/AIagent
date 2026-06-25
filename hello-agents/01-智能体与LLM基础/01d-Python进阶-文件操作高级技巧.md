---
module: "hello-agents"
title: "Python进阶：文件操作高级技巧"
description: "pathlib、上下文管理器、异步IO、临时文件等进阶技巧"
tags: [Python, Advanced, File-IO, pathlib, ContextManager, AsyncIO]
---

# 🚀 Python进阶：文件操作高级技巧

> **从基础到进阶，掌握更强大的文件操作能力**

---

## 📝 前言

在 [[01c-Python基础-文件操作与异常处理]] 中我们学习了基础的文件读写和异常处理。本章将介绍更高级的文件操作技巧：

- **pathlib** — 面向对象的路径操作
- **上下文管理器** — 自定义 with 语句
- **异步文件操作** — asyncio 并发读写
- **临时文件与内存文件** — StringIO / BytesIO

---

## 🚀 1. pathlib — 面向对象的路径操作

`pathlib` 是 Python 3.4+ 引入的标准库，用对象代替字符串操作路径，更直观、跨平台。

### 1.1 创建路径对象

```python
from pathlib import Path

# 创建路径
p = Path("data/users.json")

# 用 / 拼接路径（自动处理分隔符）
data_dir = Path("data")
file_path = data_dir / "users" / "config.json"
# 结果: data/users/config.json
```

### 1.2 路径属性

```python
from pathlib import Path

p = Path("/home/user/data/report.pdf")

p.name       # 'report.pdf'    — 完整文件名
p.stem       # 'report'        — 不带后缀
p.suffix     # '.pdf'          — 后缀
p.parent     # '/home/user/data' — 父目录
p.parts      # ('/', 'home', 'user', 'data', 'report.pdf')
```

### 1.3 路径判断与查询

```python
from pathlib import Path

p = Path("data.csv")

p.exists()       # 文件或目录是否存在
p.is_file()      # 是否是文件
p.is_dir()       # 是否是目录
p.is_absolute()  # 是否是绝对路径
```

### 1.4 目录操作

```python
from pathlib import Path

# 创建目录
Path("data/logs").mkdir(parents=True, exist_ok=True)
# parents=True: 递归创建父目录
# exist_ok=True: 已存在不报错

# 遍历目录
for f in Path(".").glob("*.py"):        # 当前目录 .py 文件
    print(f)

for f in Path(".").rglob("*.py"):       # 递归所有子目录
    print(f)

# 列出目录内容
list(Path(".").iterdir())
```

### 1.5 文件读写

```python
from pathlib import Path

p = Path("config.json")

# 读取（Python 3.6+）
content = p.read_text(encoding="utf-8")

# 写入
p.write_text('{"key": "value"}', encoding="utf-8")

# 读取二进制
data = p.read_bytes()

# 写入二进制
p.write_bytes(b"\x89PNG\r\n")
```

### 1.6 实用示例：批量重命名

```python
from pathlib import Path

def batch_rename(directory, old_ext, new_ext):
    """批量修改文件后缀"""
    count = 0
    for f in Path(directory).glob(f"*{old_ext}"):
        new_name = f.with_suffix(new_ext)
        f.rename(new_name)
        count += 1
        print(f"  {f.name} -> {new_name.name}")
    return count

# 使用
renamed = batch_rename("./images", ".jpeg", ".jpg")
print(f"共重命名 {renamed} 个文件")
```

### 1.7 pathlib vs os.path 对比

| 操作 | os.path | pathlib |
|------|---------|---------|
| 拼接路径 | `os.path.join(a, b)` | `Path(a) / b` |
| 获取文件名 | `os.path.basename(p)` | `Path(p).name` |
| 获取后缀 | `os.path.splitext(p)[1]` | `Path(p).suffix` |
| 判断存在 | `os.path.exists(p)` | `Path(p).exists()` |
| 创建目录 | `os.makedirs(p, exist_ok=True)` | `Path(p).mkdir(parents=True, exist_ok=True)` |
| 遍历文件 | `glob.glob("*.py")` | `Path(".").glob("*.py")` |

---

## 🎯 2. 上下文管理器 — 自定义 with 语句

上下文管理器确保资源的正确获取和释放，即使发生异常也能安全清理。

### 2.1 类实现（__enter__ / __exit__）

```python
class DatabaseConnection:
    def __init__(self, db_name):
        self.db_name = db_name
        self.conn = None

    def __enter__(self):
        """进入 with 块时调用"""
        print(f"连接数据库: {self.db_name}")
        self.conn = {"status": "connected"}  # 模拟连接
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出 with 块时调用（无论是否发生异常）"""
        print(f"关闭数据库: {self.db_name}")
        self.conn = None
        return False  # False = 不抑制异常，True = 吞掉异常

# 使用
with DatabaseConnection("mydb") as conn:
    print(f"执行查询... {conn}")
# 输出:
# 连接数据库: mydb
# 执行查询... {'status': 'connected'}
# 关闭数据库: mydb
```

### 2.2 contextlib 装饰器（更简洁）

```python
from contextlib import contextmanager
import time

@contextmanager
def timer(label=""):
    """计时上下文管理器"""
    start = time.time()
    yield  # 这里是 with 块的代码执行点
    elapsed = time.time() - start
    print(f"{label} 耗时: {elapsed:.3f}s")

# 使用
with timer("数据处理"):
    time.sleep(1)
    data = [i**2 for i in range(1000000)]
# 输出: 数据处理 耗时: 0.087s
```

### 2.3 抑制异常

```python
from contextlib import suppress
import os

# 文件不存在也不报错
with suppress(FileNotFoundError):
    os.remove("maybe_exists.txt")

# 等价于：
try:
    os.remove("maybe_exists.txt")
except FileNotFoundError:
    pass
```

### 2.4 嵌套与组合

```python
from contextlib import contextmanager

@contextmanager
def open_write(filename):
    print(f"打开文件: {filename}")
    f = open(filename, 'w')
    try:
        yield f
    finally:
        f.close()
        print(f"关闭文件: {filename}")

@contextmanager
def timer(label=""):
    start = time.time()
    yield
    print(f"{label}: {time.time() - start:.3f}s")

# 嵌套使用
with timer("写文件"):
    with open_write("output.txt") as f:
        f.write("hello world")
```

### 2.5 实用示例：数据库事务

```python
from contextlib import contextmanager

@contextmanager
def transaction(connection):
    """数据库事务上下文管理器"""
    try:
        yield connection
        connection.commit()
        print("事务提交成功")
    except Exception as e:
        connection.rollback()
        print(f"事务回滚: {e}")
        raise

# 使用
with transaction(db_conn) as conn:
    conn.execute("INSERT INTO users VALUES (1, '张三')")
    conn.execute("UPDATE accounts SET balance = balance - 100")
```

---

## ⚡ 3. 异步文件操作（asyncio）

异步 IO 在处理大量并发 I/O 操作时性能远优于同步方式。

### 3.1 aiofiles 基础

```python
import asyncio
import aiofiles  # pip install aiofiles

async def read_file(filepath):
    """异步读取文件"""
    async with aiofiles.open(filepath, 'r') as f:
        content = await f.read()
    return content

async def write_file(filepath, data):
    """异步写入文件"""
    async with aiofiles.open(filepath, 'w') as f:
        await f.write(data)

async def main():
    content = await read_file("data.txt")
    print(content[:100])

asyncio.run(main())
```

### 3.2 并发读取多个文件

```python
import asyncio
import aiofiles

async def read_file(filepath):
    async with aiofiles.open(filepath, 'r') as f:
        return await f.read()

async def main():
    files = ["file1.txt", "file2.txt", "file3.txt"]
    tasks = [read_file(f) for f in files]
    contents = await asyncio.gather(*tasks)
    for name, content in zip(files, contents):
        print(f"{name}: {len(content)} bytes")

asyncio.run(main())
```

### 3.3 异步写入日志

```python
import asyncio
import aiofiles
from datetime import datetime

async def async_logger(message, filename="async.log"):
    """异步日志写入"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"
    async with aiofiles.open(filename, 'a') as f:
        await f.write(log_entry)

async def main():
    # 并发写入多条日志
    tasks = [async_logger(f"Event {i}") for i in range(10)]
    await asyncio.gather(*tasks)
    print("所有日志写入完成")

asyncio.run(main())
```

---

## 📦 4. 临时文件与内存文件

### 4.1 临时文件（tempfile）

```python
import tempfile
import os

# 临时文件（自动删除）
with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=True) as f:
    f.write("临时数据")
    temp_name = f.name
    print(f"写入: {temp_name}")
# 文件在这里自动删除

# 保留临时文件
with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    f.write("需要保留的数据")
    temp_name = f.name

print(f"临时文件路径: {temp_name}")
# 手动删除
os.unlink(temp_name)

# 临时目录
with tempfile.TemporaryDirectory() as tmpdir:
    print(f"临时目录: {tmpdir}")
    # 在此范围内使用 tmpdir
    filepath = os.path.join(tmpdir, "test.txt")
    with open(filepath, 'w') as f:
        f.write("hello")
# 目录在这里自动删除
```

### 4.2 内存文件（StringIO / BytesIO）

```python
import io

# StringIO — 文本模式的内存文件
text_buffer = io.StringIO()
text_buffer.write("第一行\n")
text_buffer.write("第二行\n")
text_buffer.seek(0)  # 回到开头
print(text_buffer.read())
text_buffer.close()

# BytesIO — 二进制模式的内存文件
binary_buffer = io.BytesIO()
binary_buffer.write(b"\x89PNG\r\n\x1a\n")  # PNG 魔数
binary_buffer.seek(0)
data = binary_buffer.read(8)
print(data)
binary_buffer.close()
```

### 4.3 实用示例：内存中处理 CSV

```python
import csv
import io

csv_data = """name,age,email
张三,25,zhangsan@example.com
李四,30,lisi@example.com"""

# 在内存中解析 CSV
reader = csv.DictReader(io.StringIO(csv_data))
for row in reader:
    print(row)
# {'name': '张三', 'age': '25', 'email': 'zhangsan@example.com'}
# {'name': '李四', 'age': '30', 'email': 'lisi@example.com'}
```

---

## 📚 本节小结

| 概念 | 说明 | 适用场景 |
|------|------|----------|
| pathlib | 面向对象路径操作 | 替代 os.path |
| 上下文管理器 | 资源安全获取/释放 | 数据库、文件、锁 |
| 异步IO | asyncio + aiofiles | 高并发 I/O |
| 临时文件 | 自动清理的临时存储 | 中间数据处理 |
| 内存文件 | StringIO / BytesIO | 内存中处理文本/二进制 |

---

## 🎯 下一步

- [[01e-Python进阶-常用框架与库]] - pandas、FastAPI、pytest 等常用工具
- [[01c-Python基础-文件操作与异常处理]] - 回顾基础文件操作

---

> 💡 **实践建议**：
> 1. 用 pathlib 重写一个批量文件处理工具
> 2. 用上下文管理器实现一个简单的连接池
> 3. 用 asyncio + aiofiles 并发下载多个文件
