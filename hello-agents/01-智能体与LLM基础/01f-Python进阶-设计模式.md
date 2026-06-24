---
title: "Python进阶：设计模式"
description: "单例、工厂、观察者、策略、装饰器、代理等常用设计模式"
tags: [Python, Advanced, Design-Patterns, OOP]
---

# 🎨 Python进阶：设计模式

> **写出可维护、可扩展的代码**

---

## 📝 前言

设计模式是解决常见问题的"套路"。掌握这些模式能让你的代码：

- **更易维护** — 修改一处不影响全局
- **更易扩展** — 新增功能不改旧代码
- **更易理解** — 团队有共同的"语言"

本章介绍 Python 中最常用的设计模式。

---

## 🔒 1. 单例模式（Singleton）

**场景**：全局只需要一个实例，如数据库连接、配置管理。

### 1.1 __new__ 实现

```python
class Database:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.connection = "已连接"

# 使用
db1 = Database()
db2 = Database()
print(db1 is db2)  # True — 同一个实例
```

### 1.2 装饰器实现

```python
def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class Logger:
    def __init__(self):
        self.logs = []

logger1 = Logger()
logger2 = Logger()
print(logger1 is logger2)  # True
```

### 1.3 模块级单例（最简洁推荐）

```python
# config.py
class _Config:
    def __init__(self):
        self.debug = False
        self.db_url = "sqlite:///app.db"
        self.api_key = ""

config = _Config()  # 导入即单例

# main.py
from config import config
config.debug = True  # 全局共享
```

---

## 🏭 2. 工厂模式（Factory）

**场景**：根据条件创建不同类型的对象，如不同存储方式、不同通知渠道。

```python
from abc import ABC, abstractmethod

# 抽象产品
class Storage(ABC):
    @abstractmethod
    def save(self, key, data):
        pass

    @abstractmethod
    def load(self, key):
        pass

# 具体产品
class FileStorage(Storage):
    def save(self, key, data):
        print(f"保存到文件: {key} = {data}")

    def load(self, key):
        print(f"从文件加载: {key}")
        return {}

class RedisStorage(Storage):
    def save(self, key, data):
        print(f"保存到 Redis: {key} = {data}")

    def load(self, key):
        print(f"从 Redis 加载: {key}")
        return {}

# 工厂
class StorageFactory:
    _storages = {
        "file": FileStorage,
        "redis": RedisStorage,
    }

    @classmethod
    def create(cls, storage_type: str) -> Storage:
        if storage_type not in cls._storages:
            raise ValueError(f"未知类型: {storage_type}")
        return cls._storages[storage_type]()

    @classmethod
    def register(cls, name: str, storage_class):
        cls._storages[name] = storage_class

# 使用
storage = StorageFactory.create("file")
storage.save("user:1", {"name": "张三"})

# 动态注册新类型
class MongoStorage(Storage):
    def save(self, key, data):
        print(f"保存到 MongoDB: {key}")

    def load(self, key):
        return {}

StorageFactory.register("mongo", MongoStorage)
```

---

## 👀 3. 观察者模式（Observer）

**场景**：一个对象状态变化时，自动通知多个依赖对象，如事件系统、消息订阅。

```python
from typing import Callable, Dict, List, Any

class EventEmitter:
    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = {}

    def on(self, event: str, callback: Callable):
        """注册监听器"""
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(callback)

    def emit(self, event: str, *args, **kwargs):
        """触发事件"""
        for callback in self._listeners.get(event, []):
            callback(*args, **kwargs)

    def off(self, event: str, callback: Callable):
        """取消监听"""
        if event in self._listeners:
            self._listeners[event].remove(callback)

# 使用
emitter = EventEmitter()

def log_event(user):
    print(f"[日志] {user} 执行了操作")

def send_email(user):
    print(f"[邮件] 通知 {user}")

def update_stats(user):
    print(f"[统计] 更新 {user} 的统计数据")

# 注册多个监听器
emitter.on("login", log_event)
emitter.on("login", send_email)
emitter.on("login", update_stats)

# 触发事件（一次触发所有监听器）
emitter.emit("login", "张三")
# 输出:
# [日志] 张三 执行了操作
# [邮件] 通知 张三
# [统计] 更新 张三 的统计数据
```

---

## 🎯 4. 策略模式（Strategy）

**场景**：同一操作有多种算法，运行时可动态切换，如排序、定价、认证方式。

```python
from typing import List

# 策略函数
def sort_bubble(arr: List) -> List:
    """冒泡排序"""
    a = arr.copy()
    for i in range(len(a)):
        for j in range(len(a) - 1 - i):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
    return a

def sort_quick(arr: List) -> List:
    """快速排序"""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return sort_quick(left) + middle + sort_quick(right)

def sort_builtin(arr: List) -> List:
    """内置排序"""
    return sorted(arr)

# 上下文（使用策略的类）
class Sorter:
    def __init__(self, strategy=None):
        self._strategy = strategy or sort_builtin

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, strategy):
        self._strategy = strategy

    def sort(self, arr: List) -> List:
        return self._strategy(arr)

# 使用
sorter = Sorter()
print(sorter.sort([3, 1, 2]))           # [1, 2, 3]

sorter.strategy = sort_bubble           # 切换策略
print(sorter.sort([3, 1, 2]))           # [1, 2, 3]

sorter.strategy = sort_quick            # 再切换
print(sorter.sort([3, 1, 2]))           # [1, 2, 3]
```

---

## 🎁 5. 装饰器模式（Decorator）

**场景**：动态添加功能，如日志、计时、权限检查、缓存。

### 5.1 计时装饰器

```python
import time
import functools

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"⏱️ {func.__name__} 耗时: {elapsed:.4f}s")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "done"

slow_function()  # ⏱️ slow_function 耗时: 1.0012s
```

### 5.2 重试装饰器

```python
import time
import functools

def retry(max_attempts=3, delay=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    print(f"🔄 重试 {attempt + 1}/{max_attempts}: {e}")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=0.5)
def fetch_data(url):
    import random
    if random.random() < 0.7:
        raise ConnectionError("网络错误")
    return {"data": "ok"}
```

### 5.3 缓存装饰器

```python
import functools

def cache(func):
    """简单缓存"""
    memo = {}
    @functools.wraps(func)
    def wrapper(*args):
        if args not in memo:
            memo[args] = func(*args)
        return memo[args]
    return wrapper

@cache
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Python 内置缓存
@functools.lru_cache(maxsize=128)
def expensive_computation(x):
    print(f"计算 {x}...")
    return x * x
```

---

## 🛡️ 6. 代理模式（Proxy）

**场景**：控制对真实对象的访问，如缓存、权限、延迟加载。

```python
class RealService:
    def request(self):
        return "真实服务响应"

class Proxy:
    def __init__(self, service):
        self._service = service
        self._cache = {}
        self._access_count = 0

    def request(self):
        # 缓存检查
        if "response" not in self._cache:
            print("📡 代理: 首次请求，转发到真实服务")
            self._cache["response"] = self._service.request()
        else:
            print("📦 代理: 返回缓存结果")

        self._access_count += 1
        return self._cache["response"]

    def get_stats(self):
        return {"access_count": self._access_count}

# 使用
real = RealService()
proxy = Proxy(real)

print(proxy.request())  # 📡 首次请求... → 真实服务响应
print(proxy.request())  # 📦 返回缓存结果 → 真实服务响应
print(proxy.get_stats())  # {'access_count': 2}
```

---

## 🔗 7. 责任链模式（Chain of Responsibility）

**场景**：多个处理器依次处理请求，如中间件、审批流程。

```python
from typing import Optional

class Handler:
    def __init__(self):
        self._next: Optional[Handler] = None

    def set_next(self, handler: 'Handler') -> 'Handler':
        self._next = handler
        return handler  # 支持链式调用

    def handle(self, request):
        if self._next:
            return self._next.handle(request)
        return None

class AuthHandler(Handler):
    def handle(self, request):
        if not request.get("token"):
            print("❌ 认证失败：缺少 token")
            return None
        print("✅ 认证通过")
        return super().handle(request)

class RateLimitHandler(Handler):
    def handle(self, request):
        if request.get("rate_exceeded"):
            print("❌ 频率限制：请求过于频繁")
            return None
        print("✅ 频率检查通过")
        return super().handle(request)

class BusinessHandler(Handler):
    def handle(self, request):
        print(f"✅ 处理业务逻辑: {request.get('action')}")
        return {"status": "ok"}

# 构建责任链
auth = AuthHandler()
rate_limit = RateLimitHandler()
business = BusinessHandler()

auth.set_next(rate_limit).set_next(business)

# 使用
request = {"token": "abc123", "action": "query"}
auth.handle(request)
# ✅ 认证通过
# ✅ 频率检查通过
# ✅ 处理业务逻辑: query
```

---

## 📚 本节小结

| 模式 | 核心思想 | 适用场景 |
|------|----------|----------|
| 单例 | 全局唯一实例 | 配置、数据库连接 |
| 工厂 | 封装对象创建 | 多种存储/通知方式 |
| 观察者 | 一对多通知 | 事件系统、消息订阅 |
| 策略 | 算法可切换 | 排序、定价、认证 |
| 装饰器 | 动态增强功能 | 日志、重试、缓存 |
| 代理 | 控制访问 | 缓存、权限、延迟加载 |
| 责任链 | 依次处理 | 中间件、审批流程 |

---

## 🎯 下一步

- [[01g-Python进阶-编码风格与最佳实践]] - PEP 8、类型注解、Pythonic 写法
- [[01e-Python进阶-常用框架与库]] - 常用框架速查

---

> 💡 **实践建议**：
> 1. 用工厂模式实现一个支持多种存储的缓存系统
> 2. 用观察者模式实现一个事件总线
> 3. 用装饰器实现一个自动日志 + 重试的工具
