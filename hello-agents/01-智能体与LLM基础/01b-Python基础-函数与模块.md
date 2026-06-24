# 01b Python基础：函数与模块

> 在上一节中，我们学习了Python的基础语法，包括变量、数据类型、条件判断和循环。
> 这一节，我们将学习函数和模块——这是让代码变得整洁、可复用的关键工具。

---

## 1. 为什么需要函数？把重复的事情封装起来

### 1.1 生活中的例子

想象一下你每天早上出门前的准备工作：

1. 刷牙
2. 洗脸
3. 穿衣服
4. 吃早餐
5. 出门

如果你每天都要把这些步骤重复写一遍，是不是很麻烦？

函数就像是把这套流程"打包"成一个模板，取个名字叫"出门准备"。
以后每次只需要说"执行出门准备"，系统就会自动帮你完成所有步骤。

### 1.2 代码中的重复问题

假设你需要写一个程序，计算3个班级的平均成绩：

```python
# 班级1的成绩
scores1 = [85, 90, 78, 92, 88]
total1 = 0
for score in scores1:
    total1 += score
average1 = total1 / len(scores1)
print(f"班级1平均分: {average1}")

# 班级2的成绩
scores2 = [76, 85, 90, 88, 95]
total2 = 0
for score in scores2:
    total2 += score
average2 = total2 / len(scores2)
print(f"班级2平均分: {average2}")

# 班级3的成绩
scores3 = [90, 82, 88, 95, 79]
total3 = 0
for score in scores3:
    total3 += score
average3 = total3 / len(scores3)
print(f"班级3平均分: {average3}")
```

运行结果：
```
班级1平均分: 86.6
班级2平均分: 86.8
班级3平均分: 86.8
```

你发现了吗？计算平均分的代码重复了3次！如果要计算100个班级，就要写100遍？

### 1.3 函数的魔力

用函数来改写上面的代码：

```python
def calculate_average(scores):
    """计算平均分"""
    total = 0
    for score in scores:
        total += score
    return total / len(scores)

# 班级1的成绩
scores1 = [85, 90, 78, 92, 88]
average1 = calculate_average(scores1)
print(f"班级1平均分: {average1}")

# 班级2的成绩
scores2 = [76, 85, 90, 88, 95]
average2 = calculate_average(scores2)
print(f"班级2平均分: {average2}")

# 班级3的成绩
scores3 = [90, 82, 88, 95, 79]
average3 = calculate_average(scores3)
print(f"班级3平均分: {average3}")
```

运行结果：
```
班级1平均分: 86.6
班级2平均分: 86.8
班级3平均分: 86.8
```

现在只需要定义一次 `calculate_average` 函数，就可以反复使用了！

---

## 2. 函数的定义与调用

### 2.1 定义函数的基本格式

```python
def 函数名(参数):
    """函数的说明文档（可选，但建议写）"""
    函数体（要执行的代码）
    return 返回值
```

让我们来写第一个函数：

```python
def say_hello():
    """打印一句问候语"""
    print("你好，欢迎来到Python世界！")

# 调用函数
say_hello()
```

运行结果：
```
你好，欢迎来到Python世界！
```

### 2.2 调用函数

定义函数只是创建了一个"模板"，要真正执行，需要**调用**它：

```python
def say_hello():
    """打印一句问候语"""
    print("你好，欢迎来到Python世界！")

# 这只是定义，不会执行
# 现在调用函数
say_hello()
say_hello()  # 可以多次调用
```

运行结果：
```
你好，欢迎来到Python世界！
你好，欢迎来到Python世界！
```

### 2.3 函数命名规范

函数名就像给一个人取名字，要遵循Python的命名规则：

```python
# 正确的命名方式
def calculate_average():    # 使用下划线分隔单词（推荐）
    pass

def get_user_info():        # 动词开头，表示"做什么"
    pass

def is_valid():             # 返回布尔值的函数可以用 is_ 开头
    pass

# 错误的命名方式（语法错误）
# def 123abc():             # 不能以数字开头
# def my-function():        # 不能包含连字符
# def class():              # 不能使用Python关键字
```

### 2.4 函数的文档字符串（Docstring）

文档字符串用来说明函数的作用，用三个引号包裹：

```python
def greet(name, age):
    """
    向某人打招呼并告知其年龄

    参数:
        name (str): 姓名
        age (int): 年龄

    返回:
        str: 问候语
    """
    return f"你好，{name}！你今年{age}岁了。"

# 可以通过 help() 查看函数的文档
help(greet)
```

运行结果：
```
Help on function greet in module __main__:

greet(name, age)
    向某人打招呼并告知其年龄

    参数:
        name (str): 姓名
        age (int): 年龄

    返回:
        str: 问候语
```

---

## 3. 参数与返回值

### 3.1 参数：函数的"输入"

参数就像点餐时的"定制选项"，让同一个函数可以处理不同的数据：

```python
def greet(name):
    """向某人打招呼"""
    print(f"你好，{name}！")

# 不同的输入，相同的处理
greet("小明")
greet("小红")
greet("Python学习者")
```

运行结果：
```
你好，小明！
你好，小红！
你好，Python学习者！
```

### 3.2 多个参数

函数可以接收多个参数，用逗号分隔：

```python
def introduce(name, age, city):
    """做一个自我介绍"""
    print(f"大家好，我叫{name}，今年{age}岁，来自{city}。")

introduce("小明", 25, "北京")
introduce("小红", 22, "上海")
```

运行结果：
```
大家好，我叫小明，今年25岁，来自北京。
大家好，我叫小红，今年22岁，来自上海。
```

### 3.3 返回值：函数的"输出"

函数处理完数据后，可以通过 `return` 把结果"返回"给调用者：

```python
def add(a, b):
    """计算两个数的和"""
    result = a + b
    return result  # 把结果返回出去

# 调用函数并获取返回值
sum_result = add(3, 5)
print(f"3 + 5 = {sum_result}")

# 也可以直接打印
print(f"10 + 20 = {add(10, 20)}")
```

运行结果：
```
3 + 5 = 8
10 + 20 = 30
```

### 3.4 返回多个值

Python函数可以同时返回多个值：

```python
def get_min_max(numbers):
    """返回列表中的最小值和最大值"""
    return min(numbers), max(numbers)

# 调用函数
minimum, maximum = get_min_max([5, 2, 8, 1, 9, 3])
print(f"最小值: {minimum}")
print(f"最大值: {maximum}")
```

运行结果：
```
最小值: 1
最大值: 9
```

### 3.5 没有返回值的函数

有些函数只是执行某些操作，不需要返回值：

```python
def print_separator():
    """打印一条分隔线"""
    print("-" * 40)

print_separator()
print("这是内容")
print_separator()
```

运行结果：
```
----------------------------------------
这是内容
----------------------------------------
```

不写 `return` 或者写 `return` 不带任何值，函数默认返回 `None`。

### 3.6 实际例子：BMI计算器

```python
def calculate_bmi(weight, height):
    """
    计算BMI指数
    weight: 体重（千克）
    height: 身高（米）
    """
    bmi = weight / (height ** 2)
    return bmi

def get_bmi_category(bmi):
    """根据BMI判断体重状态"""
    if bmi < 18.5:
        return "偏瘦"
    elif bmi < 24:
        return "正常"
    elif bmi < 28:
        return "偏胖"
    else:
        return "肥胖"

# 使用函数
bmi = calculate_bmi(65, 1.75)
category = get_bmi_category(bmi)
print(f"你的BMI指数是: {bmi:.1f}")
print(f"体重状态: {category}")
```

运行结果：
```
你的BMI指数是: 21.2
体重状态: 正常
```

---

## 4. 默认参数与可变参数

### 4.1 默认参数

有时候，某个参数大多数情况下都用同一个值，就可以给它设个默认值：

```python
def greet(name, greeting="你好"):
    """打招呼，默认问候语是"你好" """
    print(f"{greeting}，{name}！")

# 使用默认参数
greet("小明")

# 自定义问候语
greet("小明", "早上好")
greet("小明", "下午好")
```

运行结果：
```
你好，小明！
早上好，小明！
下午好，小明！
```

### 4.2 默认参数的注意事项

**重要规则**：默认参数必须放在非默认参数的后面！

```python
# 正确的写法
def greet(name, greeting="你好"):
    print(f"{greeting}，{name}！")

# 错误的写法（语法错误）
# def greet(greeting="你好", name):
#     print(f"{greeting}，{name}！")
```

### 4.3 可变参数 *args

当你不确定函数会接收多少个参数时，可以使用 `*args`：

```python
def calculate_sum(*numbers):
    """计算任意多个数的和"""
    total = 0
    for num in numbers:
        total += num
    return total

# 可以传任意个参数
print(f"1+2+3 = {calculate_sum(1, 2, 3)}")
print(f"1+2+3+4+5 = {calculate_sum(1, 2, 3, 4, 5)}")
print(f"10+20 = {calculate_sum(10, 20)}")
```

运行结果：
```
1+2+3 = 6
1+2+3+4+5 = 15
10+20 = 30
```

### 4.4 可变关键字参数 **kwargs

当你需要传递任意个"键=值"对时，可以使用 `**kwargs`：

```python
def print_info(**kwargs):
    """打印任意个键值对"""
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="小明", age=25, city="北京")
```

运行结果：
```
name: 小明
age: 25
city: 北京
```

### 4.5 参数的综合运用

```python
def create_profile(name, age, *hobbies, city="未知", **extra_info):
    """创建用户档案"""
    print(f"姓名: {name}")
    print(f"年龄: {age}")
    print(f"爱好: {', '.join(hobbies)}")
    print(f"城市: {city}")
    for key, value in extra_info.items():
        print(f"{key}: {value}")

create_profile("小明", 25, "编程", "游泳", "阅读",
               city="北京", job="工程师", education="本科")
```

运行结果：
```
姓名: 小明
年龄: 25
爱好: 编程, 游泳, 阅读
城市: 北京
job: 工程师
education: 本科
```

---

## 5. Lambda函数

### 5.1 什么是Lambda函数？

Lambda函数是一种**匿名函数**（没有名字的函数），适合写一些简单的、一次性的函数。

就像餐厅的"外卖小票"——简单的任务不需要复杂的流程单。

```python
# 普通函数
def add(a, b):
    return a + b

# Lambda函数（完全等价）
add_lambda = lambda a, b: a + b

# 使用
print(f"普通函数: {add(3, 5)}")
print(f"Lambda函数: {add_lambda(3, 5)}")
```

运行结果：
```
普通函数: 8
Lambda函数: 8
```

### 5.2 Lambda函数的格式

```python
lambda 参数: 返回值
```

```python
# 一些Lambda函数的例子

# 求平方
square = lambda x: x ** 2
print(f"5的平方: {square(5)}")

# 判断是否为偶数
is_even = lambda x: x % 2 == 0
print(f"4是偶数吗？{is_even(4)}")
print(f"7是偶数吗？{is_even(7)}")

# 取绝对值
absolute = lambda x: x if x >= 0 else -x
print(f"|5| = {absolute(5)}")
print(f"|-3| = {absolute(-3)}")
```

运行结果：
```
5的平方: 25
4是偶数吗？True
7是偶数吗？False
|5| = 5
|-3| = 3
```

### 5.3 Lambda的实际用途

Lambda函数最常见的用途是配合 `sorted()`、`map()`、`filter()` 等函数使用：

```python
# 按照年龄排序
students = [("小明", 25), ("小红", 22), ("小刚", 28)]
students_sorted = sorted(students, key=lambda x: x[1])
print(f"按年龄排序: {students_sorted}")

# 使用 map 对每个元素进行操作
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x ** 2, numbers))
print(f"平方: {squared}")

# 使用 filter 过滤元素
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print(f"偶数: {even_numbers}")
```

运行结果：
```
按年龄排序: [('小红', 22), ('小明', 25), ('小刚', 28)]
平方: [1, 4, 9, 16, 25]
偶数: [2, 4, 6, 8, 10]
```

---

## 6. 作用域：变量在哪里有效

### 6.1 什么是作用域？

作用域决定了变量在哪里可以被访问。就像你的银行卡，只能在特定的ATM机上使用。

```python
# 全局变量（在整个程序中都可以访问）
global_var = "我是全局变量"

def my_function():
    # 局部变量（只在函数内部有效）
    local_var = "我是局部变量"
    print(global_var)  # 可以访问全局变量
    print(local_var)

my_function()
print(global_var)  # 可以访问
# print(local_var)  # 错误！local_var在函数外无效
```

运行结果：
```
我是全局变量
我是局部变量
我是全局变量
```

### 6.2 局部变量与全局变量

```python
x = 100  # 全局变量

def my_function():
    x = 200  # 局部变量（与全局变量同名，但是不同的变量）
    print(f"函数内的x: {x}")

my_function()
print(f"函数外的x: {x}")
```

运行结果：
```
函数内的x: 200
函数外的x: 100
```

### 6.3 使用 global 关键字

如果你想在函数内修改全局变量，需要使用 `global` 关键字：

```python
counter = 0  # 全局变量

def increment():
    global counter  # 声明要使用全局变量
    counter += 1
    print(f"函数内: counter = {counter}")

increment()
increment()
increment()
print(f"函数外: counter = {counter}")
```

运行结果：
```
函数内: counter = 1
函数内: counter = 2
函数内: counter = 3
函数外: counter = 3
```

### 6.4 作用域的实际例子

```python
def calculate_area(length, width):
    """计算矩形面积"""
    area = length * width
    return area

def calculate_perimeter(length, width):
    """计算矩形周长"""
    perimeter = 2 * (length + width)
    return perimeter

# 使用
length = 10
width = 5

area = calculate_area(length, width)
perimeter = calculate_perimeter(length, width)

print(f"长方形: 长={length}, 宽={width}")
print(f"面积: {area}")
print(f"周长: {perimeter}")
```

运行结果：
```
长方形: 长=10, 宽=5
面积: 50
周长: 30
```

---

## 7. 模块与导入：站在巨人的肩膀上

### 7.1 什么是模块？

模块就像一本工具手册，里面封装了很多现成的函数和工具。

Python自带了很多模块，比如：
- `math`：数学计算
- `random`：随机数
- `datetime`：日期时间
- `os`：操作系统相关
- `sys`：系统相关

### 7.2 导入模块的几种方式

```python
# 方式1：导入整个模块
import math
print(f"圆周率: {math.pi}")
print(f"2的平方根: {math.sqrt(2)}")

# 方式2：导入模块中的特定函数
from math import pi, sqrt
print(f"圆周率: {pi}")
print(f"2的平方根: {sqrt(2)}")

# 方式3：导入模块中的所有内容（不推荐，容易命名冲突）
from math import *
print(f"圆周率: {pi}")

# 方式4：给模块起别名
import math as m
print(f"圆周率: {m.pi}")
```

运行结果：
```
圆周率: 3.141592653589793
2的平方根: 1.4142135623730951
圆周率: 3.141592653589793
2的平方根: 1.4142135623730951
圆周率: 3.141592653589793
圆周率: 3.141592653589793
```

### 7.3 查看模块的内容

```python
import math

# 查看模块中的所有内容
print(dir(math))

# 查看模块的帮助文档
# help(math)
```

运行结果：
```
['__doc__', '__loader__', '__name__', '__package__', '__spec__', 'acos', 'acosh', ...]
```

### 7.4 自己创建模块

创建一个文件 `my_module.py`：

```python
# my_module.py
def greet(name):
    """打招呼"""
    return f"你好，{name}！"

def add(a, b):
    """求和"""
    return a + b

PI = 3.14159
```

在另一个文件中导入使用：

```python
# main.py
import my_module

print(my_module.greet("小明"))
print(f"3 + 5 = {my_module.add(3, 5)}")
print(f"PI = {my_module.PI}")
```

运行结果：
```
你好，小明！
3 + 5 = 8
PI = 3.14159
```

---

## 8. 常用内置模块

### 8.1 os模块：操作系统相关

```python
import os

# 获取当前工作目录
print(f"当前目录: {os.getcwd()}")

# 列出目录中的文件
print(f"文件列表: {os.listdir('.')[:5]}...")  # 只显示前5个

# 创建目录
# os.makedirs("test_folder", exist_ok=True)

# 检查文件或目录是否存在
print(f"当前目录存在: {os.path.exists('.')}")

# 获取文件大小
# file_size = os.path.getsize("some_file.txt")
```

运行结果：
```
当前目录: /Users/yin/PycharmProjects/AIagent
文件列表: ['.DS_Store', '.git', '.gitignore', '.idea', '01-智能体与LLM基础']...
当前目录存在: True
```

### 8.2 sys模块：系统相关

```python
import sys

# Python版本
print(f"Python版本: {sys.version}")

# 平台信息
print(f"平台: {sys.platform}")

# 命令行参数
print(f"脚本名称: {sys.argv[0]}")

# 路径
print(f"Python路径: {sys.executable}")
```

运行结果：
```
Python版本: 3.10.12 (main, Jul  7 2023, 15:04:12) [Clang 14.0.3 (clang-1403.0.22.14.1)]
平台: darwin
脚本名称: /Users/yin/.local/share/JetBrains/PyCharm2023.3/helpers/pydev/pydevconsole.py
Python路径: /usr/local/bin/python3
```

### 8.3 json模块：JSON处理

JSON（JavaScript Object Notation）是一种轻量级的数据交换格式，在AI应用中非常常用：

```python
import json

# Python字典转换为JSON字符串
user = {
    "name": "小明",
    "age": 25,
    "skills": ["Python", "机器学习", "深度学习"],
    "is_student": False
}

# 转换为JSON字符串
json_string = json.dumps(user, ensure_ascii=False, indent=2)
print("JSON字符串:")
print(json_string)

# JSON字符串转换回Python字典
parsed = json.loads(json_string)
print(f"\n解析后的字典: {parsed}")
print(f"姓名: {parsed['name']}")
```

运行结果：
```
JSON字符串:
{
  "name": "小明",
  "age": 25,
  "skills": [
    "Python",
    "机器学习",
    "深度学习"
  ],
  "is_student": false
}

解析后的字典: {'name': '小明', 'age': 25, 'skills': ['Python', '机器学习', '深度学习'], 'is_student': False}
姓名: 小明
```

### 8.4 re模块：正则表达式

正则表达式用于文本匹配和提取，在AI应用中经常用到：

```python
import re

# 查找手机号码
text = "我的手机号是13812345678，备用号码是15999999999"
phones = re.findall(r'1[3-9]\d{9}', text)
print(f"找到的手机号: {phones}")

# 提取邮箱地址
text2 = "联系我：test@example.com 或 support@company.org"
emails = re.findall(r'[\w.+-]+@[\w-]+\.[\w.]+', text2)
print(f"找到的邮箱: {emails}")

# 替换文本
text3 = "今天是2024年1月1日"
new_text = re.sub(r'\d{4}年\d{1,2}月\d{1,2}日', 'XXXX年XX月XX日', text3)
print(f"替换后: {new_text}")
```

运行结果：
```
找到的手机号: ['13812345678', '15999999999']
找到的邮箱: ['test@example.com', 'support@company.org']
替换后: 今天是XXXX年XX月XX日
```

---

## 9. pip包管理：安装第三方库

### 9.1 什么是pip？

pip是Python的包管理工具，可以让你轻松安装和管理第三方库。

就像手机的"应用商店"，pip就是Python的"库商店"。

### 9.2 常用pip命令

```bash
# 安装库
pip install requests

# 安装指定版本
pip install numpy==1.24.0

# 升级库
pip install --upgrade requests

# 卸载库
pip uninstall requests

# 查看已安装的库
pip list

# 查看某个库的信息
pip show requests

# 导出当前环境的依赖
pip freeze > requirements.txt

# 从文件安装依赖
pip install -r requirements.txt
```

### 9.3 AI开发常用库

```bash
# 数据处理
pip install numpy pandas

# 机器学习
pip install scikit-learn

# 深度学习
pip install torch tensorflow

# 自然语言处理
pip install transformers tokenizers

# 数据可视化
pip install matplotlib seaborn

# HTTP请求
pip install requests

# API框架
pip install fastapi uvicorn
```

### 9.4 使用国内镜像源加速

由于网络原因，使用国内镜像可以大幅提高下载速度：

```bash
# 临时使用
pip install requests -i https://pypi.tuna.tsinghua.edu.cn/simple

# 永久设置
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 9.5 requirements.txt文件

这个文件记录了项目所需的所有依赖，方便其他人安装：

```
# requirements.txt
numpy>=1.24.0
pandas>=2.0.0
requests>=2.28.0
torch>=2.0.0
transformers>=4.30.0
```

安装所有依赖：
```bash
pip install -r requirements.txt
```

---

## 10. 实战练习：制作一个简单计算器

让我们综合运用所学知识，制作一个简单计算器程序：

```python
import math

def add(a, b):
    """加法"""
    return a + b

def subtract(a, b):
    """减法"""
    return a - b

def multiply(a, b):
    """乘法"""
    return a * b

def divide(a, b):
    """除法"""
    if b == 0:
        return "错误：除数不能为零！"
    return a / b

def power(a, b):
    """幂运算"""
    return a ** b

def square_root(a):
    """开平方"""
    if a < 0:
        return "错误：不能对负数开平方！"
    return math.sqrt(a)

def show_menu():
    """显示菜单"""
    print("\n" + "=" * 40)
    print("       简易计算器")
    print("=" * 40)
    print("1. 加法 (+)")
    print("2. 减法 (-)")
    print("3. 乘法 (*)")
    print("4. 除法 (/)")
    print("5. 幂运算 (^)")
    print("6. 开平方 (sqrt)")
    print("0. 退出")
    print("=" * 40)

def calculator():
    """主计算器函数"""
    while True:
        show_menu()
        choice = input("请选择操作 (0-6): ")

        if choice == '0':
            print("感谢使用计算器，再见！")
            break

        if choice in ['1', '2', '3', '4', '5']:
            try:
                a = float(input("请输入第一个数: "))
                b = float(input("请输入第二个数: "))

                if choice == '1':
                    result = add(a, b)
                    op = '+'
                elif choice == '2':
                    result = subtract(a, b)
                    op = '-'
                elif choice == '3':
                    result = multiply(a, b)
                    op = '*'
                elif choice == '4':
                    result = divide(a, b)
                    op = '/'
                elif choice == '5':
                    result = power(a, b)
                    op = '^'

                print(f"\n结果: {a} {op} {b} = {result}")

            except ValueError:
                print("错误：请输入有效的数字！")

        elif choice == '6':
            try:
                a = float(input("请输入要开平方的数: "))
                result = square_root(a)
                print(f"\n结果: sqrt({a}) = {result}")
            except ValueError:
                print("错误：请输入有效的数字！")

        else:
            print("无效的选择，请重新输入！")

        input("\n按回车键继续...")

# 运行计算器
if __name__ == "__main__":
    calculator()
```

这个计算器程序展示了：
1. 函数的定义和调用
2. 参数和返回值的使用
3. 条件判断和循环
4. 错误处理（try-except）
5. 模块化编程思想

---

## 11. 常见错误与调试

### 11.1 语法错误

```python
# 错误1：缩进错误
def my_function():
    print("Hello")
    print("World")  # 缩进不一致会导致错误

# 错误2：缺少冒号
def my_function()  # 缺少冒号
    print("Hello")

# 错误3：括号不匹配
print("Hello"  # 缺少右括号
```

### 11.2 运行时错误

```python
# 错误1：除以零
result = 10 / 0  # ZeroDivisionError

# 错误2：变量未定义
print(undefined_variable)  # NameError

# 错误3：类型错误
"hello" + 10  # TypeError: 只能拼接字符串

# 错误4：索引超出范围
my_list = [1, 2, 3]
print(my_list[10])  # IndexError

# 错误5：字典键不存在
my_dict = {"name": "小明"}
print(my_dict["age"])  # KeyError
```

### 11.3 使用try-except处理错误

```python
def safe_divide(a, b):
    """安全的除法运算"""
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print("错误：除数不能为零！")
        return None
    except TypeError:
        print("错误：请输入有效的数字！")
        return None

# 测试
print(f"10 / 2 = {safe_divide(10, 2)}")
print(f"10 / 0 = {safe_divide(10, 0)}")
print(f"'10' / 2 = {safe_divide('10', 2)}")
```

运行结果：
```
10 / 2 = 5.0
错误：除数不能为零！
10 / 0 = None
错误：请输入有效的数字！
'10' / 2 = None
```

### 11.4 调试技巧

1. **使用print输出中间结果**

```python
def calculate_bmi(weight, height):
    """调试版本的BMI计算"""
    print(f"输入: weight={weight}, height={height}")
    bmi = weight / (height ** 2)
    print(f"计算过程: {weight} / ({height} ** 2) = {bmi}")
    return bmi

calculate_bmi(65, 1.75)
```

运行结果：
```
输入: weight=65, height=1.75
计算过程: 65 / (1.75 ** 2) = 21.224489795918366
```

2. **使用logging模块记录日志**

```python
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_data(data):
    """处理数据"""
    logger.info(f"开始处理数据: {data}")
    result = [x * 2 for x in data]
    logger.info(f"处理完成: {result}")
    return result

process_data([1, 2, 3, 4, 5])
```

运行结果：
```
INFO:__main__:开始处理数据: [1, 2, 3, 4, 5]
INFO:__main__:处理完成: [2, 4, 6, 8, 10]
```

---

## 总结

恭喜你完成了Python函数与模块的学习！让我们回顾一下关键知识点：

| 知识点 | 说明 |
|--------|------|
| 函数定义 | 使用 `def` 关键字定义函数 |
| 参数 | 函数的输入，支持默认参数和可变参数 |
| 返回值 | 使用 `return` 返回结果 |
| Lambda | 匿名函数，适合简单的操作 |
| 作用域 | 变量的有效范围 |
| 模块 | 代码组织和复用的工具 |
| 导入 | 使用 `import` 导入模块 |
| pip | 第三方库的安装和管理 |

### 下一步学习建议

1. 多练习函数的定义和调用
2. 熟悉常用的内置模块
3. 尝试用pip安装一些第三方库
4. 学习异常处理的最佳实践

---

## 练习题

1. **基础练习**：
   - 编写一个函数，接收一个列表，返回其中的最大值和最小值
   - 编写一个函数，接收一个字符串，统计其中每个字符出现的次数

2. **进阶练习**：
   - 编写一个函数，接收任意个数字，返回它们的平均值
   - 使用lambda函数和sorted()对字典列表进行排序

3. **实战练习**：
   - 完善本节的计算器程序，添加更多功能（如取余、取整等）
   - 编写一个密码生成器，使用random模块生成随机密码

---

## 📚 相关笔记

### 基础
- [[01a-Python基础-变量与数据类型]] - 变量、数据类型
- [[01c-Python基础-文件操作与异常处理]] - 文件操作、异常处理

### 进阶
- [[01d-Python进阶-文件操作高级技巧]] - pathlib、上下文管理器、异步IO
- [[01e-Python进阶-常用框架与库]] - pandas、FastAPI、pytest 等
- [[01f-Python进阶-设计模式]] - 单例、工厂、观察者等模式
- [[01g-Python进阶-编码风格与最佳实践]] - PEP 8、类型注解、Pythonic 写法

---

> **下一节预告**：我们将学习文件操作和异常处理，让程序能够读写文件、优雅地处理错误。
