---
title: "Python基础：文件操作与异常处理"
description: "掌握文件读写和错误处理"
tags: [Python, File-IO, Exception, Basics]
---

# 📁 Python基础：文件操作与异常处理

> **让程序能够读写文件、处理错误**

---

## 📝 前言

在实际编程中，我们经常需要：
- 读取配置文件
- 保存用户数据
- 处理可能出现的错误

本章将带你掌握这些基础技能。

---

## 🔰 1. 文件操作基础

### 1.1 打开与关闭文件

```python
# 打开文件
file = open("data.txt", "r")

# 读取内容
content = file.read()

# 关闭文件
file.close()
```

### 1.2 使用 with 语句（推荐）

```python
# 自动关闭文件
with open("data.txt", "r") as file:
    content = file.read()
    # 文件在这里自动关闭
```

### 1.3 文件模式

| 模式 | 说明 |
|------|------|
| 'r' | 只读（默认） |
| 'w' | 写入（覆盖） |
| 'a' | 追加 |
| 'r+' | 读写 |
| 'rb' | 二进制读取 |
| 'wb' | 二进制写入 |

---

## 🔰 2. 读取文件

### 2.1 读取全部内容

```python
with open("data.txt", "r") as file:
    content = file.read()
    print(content)
```

### 2.2 按行读取

```python
# 读取所有行
with open("data.txt", "r") as file:
    lines = file.readlines()  # 返回列表
    for line in lines:
        print(line.strip())  # 去除换行符
```

### 2.3 逐行读取（内存友好）

```python
# 大文件推荐
with open("large_file.txt", "r") as file:
    for line in file:
        print(line.strip())
```

### 2.4 实用示例

```python
# 读取CSV文件
def read_csv(filename):
    data = []
    with open(filename, "r") as file:
        header = file.readline().strip().split(",")
        for line in file:
            row = line.strip().split(",")
            data.append(dict(zip(header, row)))
    return data

# 使用
users = read_csv("users.csv")
print(users)
```

---

## 🔰 3. 写入文件

### 3.1 覆盖写入

```python
with open("output.txt", "w") as file:
    file.write("第一行\n")
    file.write("第二行\n")
```

### 3.2 追加写入

```python
with open("log.txt", "a") as file:
    file.write("新日志\n")
```

### 3.3 写入多行

```python
lines = ["第一行", "第二行", "第三行"]

with open("output.txt", "w") as file:
    file.writelines(lines)  # 注意不会自动换行

# 或者手动换行
with open("output.txt", "w") as file:
    for line in lines:
        file.write(line + "\n")
```

### 3.4 实用示例

```python
# 保存用户数据
def save_users(users, filename):
    with open(filename, "w") as file:
        # 写入表头
        file.write("name,age,email\n")
        # 写入数据
        for user in users:
            file.write(f"{user['name']},{user['age']},{user['email']}\n")

# 使用
users = [
    {"name": "张三", "age": 25, "email": "zhangsan@example.com"},
    {"name": "李四", "age": 30, "email": "lisi@example.com"}
]
save_users(users, "users.txt")
```

---

## 🔰 4. JSON文件操作

### 4.1 读取JSON

```python
import json

with open("config.json", "r") as file:
    config = json.load(file)
    print(config)
```

### 4.2 写入JSON

```python
import json

data = {
    "name": "张三",
    "age": 25,
    "hobbies": ["编程", "读书"]
}

with open("user.json", "w") as file:
    json.dump(data, file, ensure_ascii=False, indent=2)
```

### 4.3 实用示例

```python
# 配置管理
class Config:
    def __init__(self, filename="config.json"):
        self.filename = filename
        self.data = self.load()
    
    def load(self):
        try:
            with open(self.filename, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            return {}
    
    def save(self):
        with open(self.filename, "w") as file:
            json.dump(self.data, file, ensure_ascii=False, indent=2)
    
    def get(self, key, default=None):
        return self.data.get(key, default)
    
    def set(self, key, value):
        self.data[key] = value
        self.save()

# 使用
config = Config()
config.set("theme", "dark")
config.set("language", "zh")
print(config.get("theme"))  # dark
```

---

## 🔰 5. 异常处理

### 5.1 为什么需要异常处理？

```python
# 没有异常处理
result = 10 / 0  # 程序崩溃！

# 有异常处理
try:
    result = 10 / 0
except ZeroDivisionError:
    print("不能除以零")
    result = None
```

### 5.2 try-except 基础

```python
try:
    # 可能出错的代码
    value = int(input("请输入数字："))
    result = 100 / value
    print(f"结果：{result}")
except ValueError:
    print("请输入有效的数字")
except ZeroDivisionError:
    print("不能除以零")
except Exception as e:
    print(f"其他错误：{e}")
```

### 5.3 异常类型

| 异常类型 | 说明 |
|----------|------|
| ValueError | 值错误 |
| TypeError | 类型错误 |
| FileNotFoundError | 文件不存在 |
| ZeroDivisionError | 除零错误 |
| KeyError | 键不存在 |
| IndexError | 索引越界 |

### 5.4 try-except-else-finally

```python
try:
    file = open("data.txt", "r")
    content = file.read()
except FileNotFoundError:
    print("文件不存在")
    content = None
else:
    # 没有异常时执行
    print("文件读取成功")
finally:
    # 无论如何都执行
    print("操作完成")
    if 'file' in locals():
        file.close()
```

---

## 🔰 6. 自定义异常

### 6.1 定义异常类

```python
class AgeError(Exception):
    """年龄错误"""
    def __init__(self, age, message="年龄必须在0-150之间"):
        self.age = age
        self.message = message
        super().__init__(self.message)

class Student:
    def __init__(self, name, age):
        self.name = name
        if age < 0 or age > 150:
            raise AgeError(age)
        self.age = age

# 使用
try:
    student = Student("张三", 200)
except AgeError as e:
    print(f"错误：{e}")  # 错误：年龄必须在0-150之间
```

---

## 🔰 7. 实战示例

### 7.1 日志系统

```python
import json
from datetime import datetime

class Logger:
    def __init__(self, filename="app.log"):
        self.filename = filename
    
    def log(self, level, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        try:
            with open(self.filename, "a") as file:
                file.write(log_entry)
        except Exception as e:
            print(f"日志写入失败：{e}")
    
    def info(self, message):
        self.log("INFO", message)
    
    def error(self, message):
        self.log("ERROR", message)

# 使用
logger = Logger()
logger.info("应用启动")
logger.error("发生错误")
```

### 7.2 数据备份

```python
import json
import shutil
from datetime import datetime

class BackupManager:
    def __init__(self, data_file="data.json"):
        self.data_file = data_file
    
    def backup(self):
        """创建备份"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"{self.data_file}.backup_{timestamp}"
            shutil.copy2(self.data_file, backup_file)
            print(f"备份成功：{backup_file}")
            return backup_file
        except FileNotFoundError:
            print("原文件不存在")
            return None
    
    def restore(self, backup_file):
        """恢复备份"""
        try:
            shutil.copy2(backup_file, self.data_file)
            print("恢复成功")
            return True
        except FileNotFoundError:
            print("备份文件不存在")
            return False

# 使用
manager = BackupManager()
manager.backup()
```

---

## 📚 本节小结

| 概念 | 说明 |
|------|------|
| 文件操作 | 读取、写入、追加 |
| with语句 | 自动关闭文件 |
| JSON | 结构化数据存储 |
| try-except | 捕获异常 |
| 自定义异常 | 特定错误类型 |

---

## 🎯 下一步

- **02a - 数学基础：线性代数** - AI的数学基础
- **03a - 机器学习：基本概念** - ML入门

---

> 💡 **实践建议**：写一个简单的记事本程序，练习文件操作和异常处理。
