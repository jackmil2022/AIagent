---
aliases:
  - "01a-Python基础-变量与数据类型"
title: "Python基础：变量与数据类型"
module: "hello-agents"
tags: ['Python', 'Programming', 'Basics']
---
# 01a Python基础：变量与数据类型
> 这是为零基础读者准备的Python入门教程。如果你从未写过代码，也不用担心——我们会从最基础的概念开始，用生活中的例子来理解编程。

---

## 1. 什么是编程？为什么要学Python？

### 1.1 什么是编程？

想象一下，你要教一个完全不懂中文的外国朋友做番茄炒蛋。你会怎么做？

你可能会一步一步告诉他：
1. 先把番茄洗干净
2. 把番茄切成小块
3. 打两个鸡蛋到碗里
4. 用筷子搅拌均匀
5. 锅里倒油，开火
6. ...

**编程，就是用计算机能听懂的语言，一步一步告诉它该做什么。**

计算机非常聪明，但也非常"笨"——它只能精确执行你写的每一步指令，不会自己猜测你的意图。所以我们要把任务拆解成非常细致的步骤。

### 1.2 为什么要学Python？

Python（读作"派森"）是目前最受欢迎的编程语言之一。它有几个特别适合新手的优点：

- **像说话一样简单**：Python的语法接近自然语言，读起来像英语
- **无所不能**：网站开发、数据分析、人工智能、自动化办公……几乎什么都能做
- **社区强大**：遇到问题，网上有海量的教程和答案
- **大模型时代的必备工具**：学习AI和大模型，Python是首选语言

### 1.3 你的第一行代码

打开Python（推荐使用Jupyter Notebook或PyCharm），输入下面这行代码：

```python
print("Hello, World! 你好，世界！")
```

运行后你会看到：

```
Hello, World! 你好，世界！
```

**解释一下：**
- `print()` 是Python内置的一个函数，意思是"打印/输出"
- 括号里用引号包起来的文字，叫**字符串**（后面会详细讲）
- 这行代码的作用就是：让计算机在屏幕上显示这段文字

> 就像你按下打印机的"打印"按钮，打印机会吐出一张纸一样，`print()` 会让计算机"吐出"你指定的内容。

---

## 2. 变量：给数据起名字

### 2.1 什么是变量？

在生活中，我们会给东西起名字。比如：
- 你的名字叫"小明"
- 你的年龄是25岁
- 你的体重是65.5公斤

在编程中，**变量就是给数据起的名字**。你可以把变量想象成一个贴了标签的盒子，盒子里装着某个数据。

```python
name = "小明"
age = 25
weight = 65.5
```

**解释一下：**
- `name` 是变量名（标签），`"小明"` 是存在里面的数据（盒子里的东西）
- `=` 在这里不是"等于"的意思，而是"赋值"——把右边的值**放进**左边的变量里
- 这就像你在盒子上贴了"名字"的标签，然后把写着"小明"的纸条放进去

### 2.2 变量命名规则

给变量起名字不是随便起的，要遵守几个规则：

| 规则 | 正确示例 | 错误示例 |
|------|---------|---------|
| 只能包含字母、数字、下划线 | `my_name` | `my-name`（不能有连字符） |
| 不能以数字开头 | `age1` | `1age`（数字不能放开头） |
| 不能是Python关键字 | `my_class` | `class`（class是保留字） |
| 区分大小写 | `Name` 和 `name` 是两个不同变量 | - |

**起名字的好习惯（虽然不是强制的，但强烈推荐）：**
- 用有意义的英文单词：`student_name` 比 `x` 好
- 小写字母加下划线：`my_first_name`（这种风格叫蛇形命名法）
- 见名知意：看到名字就知道盒子里装的是什么

### 2.3 变量可以修改

变量最大的特点是：**里面的值可以换！**

```python
score = 100
print(score)   # 输出：100

score = 95     # 考试成绩更新了
print(score)   # 输出：95
```

**就像一个可以反复贴标签的盒子**：之前盒子里装的是100，现在你换成了95。

### 2.4 同时给多个变量赋值

Python还支持一次性给多个变量赋值：

```python
# 同时给三个变量赋值
x, y, z = 1, 2, 3
print(x)  # 输出：1
print(y)  # 输出：2
print(z)  # 输出：3
```

### 2.5 练习题

**练习1：** 创建三个变量，分别存储你的姓名、年龄和城市，然后用 `print()` 把它们打印出来。

```python
# 在这里写你的代码
# 提示：
# my_name = "你的名字"
# my_age = 你的年龄
# my_city = "你所在的城市"
```

**练习2：** 创建一个变量 `temperature` 表示温度，先设为30，再改成25，每次修改后都打印出来。

---

## 3. 数据类型：整数、浮点数、字符串、布尔值

### 3.1 为什么要有数据类型？

生活中，不同类型的物品需要不同的处理方式：
- 数字可以做加减乘除
- 文字可以拼接、截取
- 对错判断只有"是"和"否"

Python也是这样。**数据类型就是告诉计算机：这个数据是什么类型的，该怎么处理它。**

### 3.2 整数（int）

整数就是没有小数点的数字，正数、负数、零都算。

```python
age = 25          # 正整数
temperature = -5  # 负整数
count = 0         # 零

print(type(age))  # 输出：<class 'int'>
```

`type()` 这个函数可以查看变量的数据类型。`int` 就是 integer（整数）的缩写。

**生活例子：** 班级里有35个学生、你的银行卡余额是10000元（不考虑小数）、一年有365天——这些都是整数。

```python
students = 35
bank_balance = 10000
days_in_year = 365

print("班级人数：", students)
print("银行卡余额：", bank_balance)
print("一年天数：", days_in_year)
```

运行结果：
```
班级人数： 35
银行卡余额： 10000
一年天数： 365
```

### 3.3 浮点数（float）

浮点数就是带小数点的数字。

```python
height = 1.75       # 身高1.75米
pi = 3.14159        # 圆周率
price = 9.99        # 商品价格

print(type(height))  # 输出：<class 'float'>
```

`float` 就是浮点数的意思。为什么叫"浮点"？因为小数点的位置是"浮动"的。

**生活例子：** 你去超市买了3.5斤苹果，每斤8.99元。

```python
weight = 3.5
price_per_jin = 8.99
total = weight * price_per_jin

print("苹果总价：", total)
```

运行结果：
```
苹果总价： 31.465
```

### 3.4 字符串（str）

字符串就是一段文字，用引号包起来。Python中可以用单引号 `'` 或双引号 `"`，效果一样。

```python
name = "小明"
greeting = '你好啊'
message = "Python is fun!"

print(type(name))  # 输出：<class 'str'>
```

`str` 是 string（字符串）的缩写。

**注意引号必须配对：**

```python
# 正确
name = "小明"
name = '小明'

# 错误！
name = "小明'  # 引号不匹配
name = '小明"  # 引号不匹配
```

**字符串可以包含任何文字：**

```python
# 中文
poem = "床前明月光"

# 英文
english = "Hello World"

# 数字也可以存成字符串（但不能做数学运算）
phone = "13800138000"
zip_code = "100000"

print(phone)       # 输出：13800138000
print(type(phone)) # 输出：<class 'str'>
```

**三引号：** 如果字符串很长，或者需要换行，可以用三个引号：

```python
poem = """
    白日依山尽，
    黄河入海流。
    欲穷千里目，
    更上一层楼。
"""
print(poem)
```

运行结果：
```
    白日依山尽，
    黄河入海流。
    欲穷千里目，
    更上一层楼。
```

### 3.5 布尔值（bool）

布尔值只有两个：`True`（真）和 `False`（假）。就像开关一样，只有开和关两种状态。

```python
is_student = True
is_rich = False

print(type(is_student))  # 输出：<class 'bool'>
```

`bool` 是 boolean（布尔）的缩写，以数学家乔治·布尔的名字命名。

**生活例子：**
- "今天下雨吗？" —— 要么下雨(True)，要么没下雨(False)
- "你是会员吗？" —— 要么是(True)，要么不是(False)
- "考试及格了吗？" —— 及格(True)，没及格(False)

布尔值通常在**条件判断**中使用（后面会讲到）：

```python
age = 20
can_vote = age >= 18   # 20大于等于18，所以是True
print("有投票资格吗？", can_vote)  # 输出：有投票资格吗？ True
```

### 3.6 数据类型转换

有时候我们需要在不同类型之间转换：

```python
# 字符串 → 整数
age_str = "25"
age_num = int(age_str)   # 用 int() 把字符串转成整数
print(age_num + 5)        # 输出：30（现在可以做数学运算了）

# 整数 → 字符串
score = 100
score_str = str(score)    # 用 str() 把整数转成字符串
print("你的成绩是：" + score_str)  # 可以和文字拼接了

# 整数 → 浮点数
x = int(3.9)   # 注意：会截断小数，结果是3，不是4！
print(x)       # 输出：3

# 浮点数 → 整数
y = float(3)   # 结果是3.0
print(y)       # 输出：3.0
```

**常见陷阱：**

```python
# 字符串的数字不能直接运算
number = "10"
result = number + 5    # 报错！字符串不能和数字相加

# 必须先转换
result = int(number) + 5  # 正确：15
print(result)
```

### 3.7 练习题

**练习1：** 判断下面变量分别是什么数据类型：

```python
a = 42
b = 3.14
c = "hello"
d = True
e = "100"

# 用 type() 函数验证你的答案
```

**练习2：** 把字符串 "123" 转换成整数，然后乘以2，看看结果是什么。

---

## 4. 运算符：算术、比较、逻辑

### 4.1 算术运算符

就像小学学的加减乘除，Python也支持：

| 运算符 | 含义 | 示例 | 结果 |
|--------|------|------|------|
| `+` | 加 | `3 + 2` | `5` |
| `-` | 减 | `3 - 2` | `1` |
| `*` | 乘 | `3 * 2` | `6` |
| `/` | 除 | `3 / 2` | `1.5` |
| `//` | 整除（取商） | `7 // 2` | `3` |
| `%` | 取余（取余数） | `7 % 2` | `1` |
| `**` | 幂（次方） | `2 ** 3` | `8` |

**生活例子：去超市购物**

```python
# 买了3瓶可乐，每瓶3元
cola_price = 3
cola_count = 3
total = cola_price * cola_count
print("可乐总价：", total)  # 输出：9

# 给了收银员50元
paid = 50
change = paid - total
print("找零：", change)  # 输出：41

# 买了7个苹果，要平均分给2个人
apples = 7
people = 2
each_person = apples // people  # 整除：每人几个
leftover = apples % people      # 取余：还剩几个
print(f"每人{each_person}个，还剩{leftover}个")  # 输出：每人3个，还剩1个

# 计算面积：长5米，宽3米
length = 5
width = 3
area = length * width
print("面积：", area, "平方米")  # 输出：15 平方米

# 2的10次方
result = 2 ** 10
print("2的10次方：", result)  # 输出：1024
```

**注意除法的结果：**

```python
print(10 / 3)    # 输出：3.3333333333333335（总是返回浮点数）
print(10 // 3)   # 输出：3（只取整数部分）
print(10 % 3)    # 输出：1（余数是1）
```

### 4.2 比较运算符

比较运算符用来比较两个值，结果是布尔值（True 或 False）。

| 运算符 | 含义 | 示例 | 结果 |
|--------|------|------|------|
| `==` | 等于 | `3 == 3` | `True` |
| `!=` | 不等于 | `3 != 5` | `True` |
| `>` | 大于 | `5 > 3` | `True` |
| `<` | 小于 | `5 < 3` | `False` |
| `>=` | 大于等于 | `5 >= 5` | `True` |
| `<=` | 小于等于 | `5 <= 3` | `False` |

**重要区分：** `=` 是赋值（把值放进变量），`==` 是比较（判断是否相等）。

```python
age = 20
print(age == 20)   # True：判断age是否等于20
print(age != 20)   # False：判断age是否不等于20
print(age > 18)    # True：判断age是否大于18
print(age < 18)    # False：判断age是否小于18

# 字符串也可以比较
print("apple" == "apple")  # True
print("apple" != "banana") # True
print("apple" < "banana")  # True（按字母顺序比较）
```

**生活例子：** 判断是否成年

```python
age = 16
is_adult = age >= 18
print("是否成年：", is_adult)  # 输出：是否成年： False
```

### 4.3 逻辑运算符

逻辑运算符用于组合多个条件。

| 运算符 | 含义 | 示例 | 结果 |
|--------|------|------|------|
| `and` | 与（两个都为True才是True） | `True and False` | `False` |
| `or` | 或（只要有一个为True就是True） | `True or False` | `True` |
| `not` | 非（取反） | `not True` | `False` |

**生活例子：**

```python
age = 25
has_id = True

# and：既要满足年龄，也要有身份证
can_enter = age >= 18 and has_id
print("可以进入：", can_enter)  # 输出：True

# or：会游泳或会骑车，至少会一个就能参加
can_swim = False
can_ride = True
can_participate = can_swim or can_ride
print("可以参加：", can_participate)  # 输出：True

# not：取反
is_raining = True
should_bring_umbrella = not is_raining  # 没下雨就不需要带伞
print("需要带伞吗：", should_bring_umbrella)  # 输出：False
```

**组合使用：**

```python
age = 20
income = 8000
has_house = False

# 年龄在18-35之间 且 月收入超过5000
can_apply = age >= 18 and age <= 35 and income > 5000
print("可以申请：", can_apply)  # 输出：True

# 有房子 或 月收入超过10000
can_loan = has_house or income > 10000
print("可以贷款：", can_loan)  # 输出：False
```

### 4.4 运算符优先级

就像数学中的"先乘除后加减"，Python也有优先级：

```
括号 > 幂 > 正负号 > 乘除 > 加减 > 比较 > not > and > or
```

**最简单的规则：不确定的时候就加括号！**

```python
result = (3 + 2) * 4    # 先加后乘：20
result2 = 3 + 2 * 4     # 先乘后加：11
print(result)   # 20
print(result2)  # 11
```

### 4.5 练习题

**练习1：** 计算你一周的学习时间。假设每天学2小时，一周7天，总共学了多少小时？如果每天学1.5小时呢？

**练习2：** 写出以下表达式的结果（先自己算，再用Python验证）：

```python
print(10 + 3 * 2)     # ?
print((10 + 3) * 2)   # ?
print(2 ** 3 ** 2)    # ?
print(10 % 3)         # ?
print(10 // 3)        # ?
```

---

## 5. 容器类型：列表、元组、字典、集合

### 5.1 为什么需要容器？

变量就像一个盒子，只能装一个东西。但如果你想管理一整个班级的学生名单怎么办？一个个变量太麻烦了。

**容器就是可以装多个数据的"大箱子"。** Python提供了四种主要的容器。

### 5.2 列表（list）

列表是最常用的容器，用方括号 `[]` 表示。列表里的每个数据叫**元素**，元素之间用逗号分隔。

```python
# 班级学生名单
students = ["小明", "小红", "小刚", "小美"]
print(students)       # 输出：['小明', '小红', '小刚', '小美']
print(type(students)) # 输出：<class 'list'>

# 成绩单
scores = [95, 87, 92, 78, 100]
print(scores)

# 可以混合类型（但不推荐）
mixed = [1, "hello", True, 3.14]
```

**访问列表中的元素：** 用索引（编号），从0开始！

```python
fruits = ["苹果", "香蕉", "橙子", "葡萄"]

print(fruits[0])  # 第一个元素：苹果
print(fruits[1])  # 第二个元素：香蕉
print(fruits[2])  # 第三个元素：橙子
print(fruits[3])  # 第四个元素：葡萄

# 从后面数，用负数
print(fruits[-1])  # 最后一个：葡萄
print(fruits[-2])  # 倒数第二个：橙子
```

> 就像电影院的座位号，从0开始编号。0号座是第一个，1号座是第二个……虽然有点反直觉，但这是编程世界的惯例。

**修改列表：**

```python
colors = ["红", "蓝", "绿"]
print(colors)  # ['红', '蓝', '绿']

colors[1] = "黄"  # 把第2个元素改成"黄"
print(colors)     # ['红', '黄', '绿']
```

**添加和删除元素：**

```python
shopping_list = ["牛奶", "面包"]
print(shopping_list)  # ['牛奶', '面包']

# 添加元素
shopping_list.append("鸡蛋")  # 在末尾添加
print(shopping_list)  # ['牛奶', '面包', '鸡蛋']

shopping_list.insert(1, "果汁")  # 在第2个位置插入
print(shopping_list)  # ['牛奶', '果汁', '面包', '鸡蛋']

# 删除元素
shopping_list.remove("面包")  # 按值删除
print(shopping_list)  # ['牛奶', '果汁', '鸡蛋']

popped = shopping_list.pop()  # 删除最后一个并返回
print(popped)           # 鸡蛋
print(shopping_list)    # ['牛奶', '果汁']
```

**列表常用操作：**

```python
numbers = [3, 1, 4, 1, 5, 9, 2, 6]

print(len(numbers))     # 长度：8
print(max(numbers))     # 最大值：9
print(min(numbers))     # 最小值：1
print(sum(numbers))     # 求和：31

numbers.sort()          # 排序（从小到大）
print(numbers)          # [1, 1, 2, 3, 4, 5, 6, 9]

numbers.reverse()       # 反转
print(numbers)          # [9, 6, 5, 4, 3, 2, 1, 1]
```

**列表可以嵌套：**

```python
# 二维列表（矩阵）
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

print(matrix[0])       # [1, 2, 3]  第一行
print(matrix[0][1])    # 2  第一行第二列
print(matrix[2][0])    # 7  第三行第一列
```

### 5.3 元组（tuple）

元组和列表很像，但元组一旦创建就**不能修改**。用圆括号 `()` 表示。

```python
# 元组不可修改
colors = ("红", "蓝", "绿")
print(colors)          # ('红', '蓝', '绿')

# 尝试修改会报错
# colors[0] = "黄"    # TypeError: 'tuple' object does not support item assignment
```

**为什么要用元组？** 因为不可修改意味着更安全，适合存储不应该改变的数据。

```python
# 一年的月份（不会变）
months = ("一月", "二月", "三月", "四月", "五月", "六月",
          "七月", "八月", "九月", "十月", "十一月", "十二月")

# 坐标点（不应该被修改）
origin = (0, 0)
point = (3, 4)

# RGB颜色值
red = (255, 0, 0)
```

**元组也可以访问，但不能修改：**

```python
point = (3, 4, 5)
print(point[0])   # 3
print(point[1])   # 4
# point[0] = 10  # 报错！
```

**列表和元组的转换：**

```python
# 列表转元组
my_list = [1, 2, 3]
my_tuple = tuple(my_list)

# 元组转列表
my_tuple2 = (4, 5, 6)
my_list2 = list(my_tuple2)
```

### 5.4 字典（dict）

字典用花括号 `{}` 表示，存储的是**键值对**（key-value pair）。就像一本真正的字典：通过"单词"（键）查找"解释"（值）。

```python
# 学生信息
student = {
    "姓名": "小明",
    "年龄": 20,
    "专业": "计算机科学",
    "成绩": 95
}

print(student)
```

**访问字典中的值：** 用键来查找。

```python
print(student["姓名"])     # 小明
print(student["年龄"])     # 20

# 也可以用 get() 方法
print(student.get("专业"))  # 计算机科学
print(student.get("身高"))  # None（键不存在返回None）
print(student.get("身高", 170))  # 170（键不存在返回默认值）
```

**修改和添加：**

```python
student = {"姓名": "小明", "年龄": 20}

# 修改已有值
student["年龄"] = 21
print(student)  # {'姓名': '小明', '年龄': 21}

# 添加新的键值对
student["城市"] = "北京"
print(student)  # {'姓名': '小明', '年龄': 21, '城市': '北京'}

# 删除
del student["城市"]
print(student)  # {'姓名': '小明', '年龄': 21}
```

**字典常用操作：**

```python
person = {"name": "小红", "age": 18, "city": "上海"}

print(person.keys())    # 所有键：dict_keys(['name', 'age', 'city'])
print(person.values())  # 所有值：dict_values(['小红', 18, '上海'])
print(person.items())   # 所有键值对：dict_items([('name', '小红'), ...])

# 检查键是否存在
print("name" in person)      # True
print("email" in person)     # False

# 遍历字典
for key, value in person.items():
    print(f"{key}: {value}")
```

运行结果：
```
name: 小红
age: 18
city: 上海
```

**字典可以嵌套：**

```python
# 班级信息
classroom = {
    "班级": "三年级一班",
    "人数": 40,
    "学生": [
        {"姓名": "小明", "成绩": 95},
        {"姓名": "小红", "成绩": 98},
        {"姓名": "小刚", "成绩": 87}
    ]
}

# 访问嵌套数据
print(classroom["班级"])              # 三年级一班
print(classroom["学生"][0]["姓名"])   # 小明
print(classroom["学生"][1]["成绩"])   # 98
```

### 5.5 集合（set）

集合用花括号 `{}` 表示（和字典一样，但没有键值对），特点是：**元素不重复，且无序**。

```python
# 集合会自动去重
numbers = {1, 2, 3, 2, 1, 4}
print(numbers)  # {1, 2, 3, 4}  重复的被自动删除了
```

**集合的常用操作：**

```python
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

# 并集（两个集合的所有元素）
print(a | b)       # {1, 2, 3, 4, 5, 6}

# 交集（两个集合共有的元素）
print(a & b)       # {3, 4}

# 差集（在a中但不在b中的元素）
print(a - b)       # {1, 2}

# 对称差集（不同时在两个集合中的元素）
print(a ^ b)       # {1, 2, 5, 6}
```

**生活例子：** 去重

```python
# 网站的访客记录（有重复访问）
visitors = ["小明", "小红", "小明", "小刚", "小红", "小明"]
unique_visitors = set(visitors)
print(f"总访问次数：{len(visitors)}")        # 6
print(f"独立访客数：{len(unique_visitors)}")  # 3
print(f"独立访客：{unique_visitors}")          # {'小明', '小红', '小刚'}
```

### 5.6 容器类型对比

| 特性 | 列表 list | 元组 tuple | 字典 dict | 集合 set |
|------|----------|-----------|----------|---------|
| 符号 | `[]` | `()` | `{}` | `{}` |
| 有序 | 有序 | 有序 | 有序(3.7+) | 无序 |
| 可修改 | 可以 | 不可以 | 可以 | 可以 |
| 重复元素 | 允许 | 允许 | 键不重复 | 不允许 |
| 索引访问 | 支持 | 支持 | 键访问 | 不支持 |

### 5.7 练习题

**练习1：** 创建一个购物清单列表，实现添加商品、删除商品、打印清单的功能。

**练习2：** 创建一个字典，存储你最喜欢的3本书的信息（书名、作者、价格），然后遍历打印出来。

---

## 6. 字符串操作详解

### 6.1 字符串是不可变的

字符串一旦创建，不能修改其中的某个字符。

```python
name = "hello"
# name[0] = "H"  # 报错！字符串不支持修改
```

但可以创建新的字符串来代替。

### 6.2 字符串拼接

```python
first_name = "小"
last_name = "明"

# 方法一：用 + 号
full_name = first_name + last_name
print(full_name)  # 小明

# 方法二：用 join() 方法
words = ["Hello", "World", "!"]
sentence = " ".join(words)
print(sentence)  # Hello World !

# 方法三：用逗号分隔（print函数自动加空格）
print("Hello", "World")  # Hello World
```

### 6.3 f-string 格式化（推荐！）

f-string 是Python 3.6+引入的字符串格式化方式，用 `f` 开头，花括号 `{}` 里放变量名。

```python
name = "小明"
age = 20
score = 95.5

# 基本用法
message = f"我叫{name}，今年{age}岁"
print(message)  # 我叫小明，今年20岁

# 可以在花括号里做运算
print(f"明年我就{age + 1}岁了")  # 明年我就21岁了

# 控制小数位数
print(f"成绩：{score:.1f}")  # 成绩：95.5（保留1位小数）
print(f"成绩：{score:.0f}")  # 成绩：96（四舍五入到整数）

# 控制对齐和宽度
print(f"{'姓名':<10}{'成绩':>8}")  # 左对齐姓名，右对齐成绩
print(f"{'小明':<10}{95:>8}")
print(f"{'小红':<10}{98:>8}")
```

运行结果：
```
姓名            成绩
小明              95
小红              98
```

### 6.4 常用字符串方法

```python
text = "  Hello, World!  "

# 大小写转换
print(text.upper())      # "  HELLO, WORLD!  "
print(text.lower())      # "  hello, world!  "
print(text.title())      # "  Hello, World!  "

# 去除首尾空格
print(text.strip())      # "Hello, World!"

# 查找和替换
print(text.find("World"))     # 9（返回首次出现的位置）
print(text.count("l"))        # 3（出现次数）
print(text.replace("World", "Python"))  # "  Hello, Python!  "

# 判断
email = "user@example.com"
print(email.startswith("user"))   # True
print(email.endswith(".com"))     # True
print(email.isdigit())            # False
print("123".isdigit())            # True
print("hello".isalpha())          # True
print("hello123".isalnum())       # True（字母或数字）
```

### 6.5 字符串切片

切片可以获取字符串的一部分，语法是 `字符串[起始:结束]`（不包含结束位置）。

```python
text = "Hello, World!"

print(text[0:5])     # Hello（从第0到第4个字符）
print(text[7:12])    # World（从第7到第11个字符）
print(text[:5])      # Hello（从开头到第4个字符）
print(text[7:])      # World!（从第7到末尾）
print(text[-6:])     # World!（从倒数第6个到末尾）

# 步长
print(text[::2])     # Hlo ol!（每隔一个取一个）
print(text[::-1])    # !dlroW ,olleH（字符串反转）
```

### 6.6 字符串分割

```python
# 分割字符串
csv_data = "小明,95,北京,计算机"
parts = csv_data.split(",")
print(parts)  # ['小明', '95', '北京', '计算机']

# 多行文本
poem = """白日依山尽
黄河入海流
欲穷千里目
更上一层楼"""

lines = poem.split("\n")
print(lines)  # ['白日依山尽', '黄河入海流', '欲穷千里目', '更上一层楼']

# 用空格分割
sentence = "I love Python"
words = sentence.split()
print(words)  # ['I', 'love', 'Python']
```

### 6.7 练习题

**练习1：** 给定一个字符串 "Python is awesome"，请：
- 转换成全大写
- 统计字母 'o' 出现的次数
- 替换 "awesome" 为 "great"
- 反转整个字符串

**练习2：** 用f-string格式化输出一个成绩报告：
```
姓名：小明
语文：95
数学：98
英语：92
平均分：95.0
```

---

## 7. 实战练习：制作一个简单的个人信息卡片

让我们把前面学到的知识综合运用，制作一个个人信息卡片。

### 7.1 简单版本

```python
# === 个人信息卡片（简单版）===

# 基本信息
name = "张三"
age = 25
city = "北京"
job = "程序员"
hobbies = ["编程", "读书", "跑步"]

# 输出卡片
print("=" * 40)
print(f"         个人信息卡片")
print("=" * 40)
print(f"  姓名：{name}")
print(f"  年龄：{age}岁")
print(f"  城市：{city}")
print(f"  职业：{job}")
print(f"  爱好：{', '.join(hobbies)}")
print("=" * 40)
```

运行结果：
```
========================================
         个人信息卡片
========================================
  姓名：张三
  年龄：25岁
  城市：北京
  职业：程序员
  爱好：编程, 读书, 跑步
========================================
```

### 7.2 进阶版本：带数据统计

```python
# === 成绩单统计 ===

# 学生信息
student_name = "李华"
student_class = "三年级二班"

# 成绩数据（字典）
scores = {
    "语文": 92,
    "数学": 98,
    "英语": 85,
    "物理": 90,
    "化学": 88
}

# 计算统计信息
total = sum(scores.values())
average = total / len(scores)
max_subject = max(scores, key=scores.get)
min_subject = min(scores, key=scores.get)

# 输出成绩单
print("=" * 45)
print(f"           {student_name} 的成绩单")
print(f"           {student_class}")
print("=" * 45)
print(f"  {'科目':<8}{'成绩':>8}")
print("-" * 45)

for subject, score in scores.items():
    status = "优秀" if score >= 90 else ("良好" if score >= 80 else "及格")
    print(f"  {subject:<8}{score:>6}  ({status})")

print("-" * 45)
print(f"  总分：{total}")
print(f"  平均分：{average:.1f}")
print(f"  最高分：{max_subject} ({scores[max_subject]}分)")
print(f"  最低分：{min_subject} ({scores[min_subject]}分)")
print("=" * 45)
```

运行结果：
```
=============================================
           李华 的成绩单
           三年级二班
=============================================
  科目              成绩
---------------------------------------------
  语文          92  (优秀)
  数学          98  (优秀)
  英语          85  (良好)
  物理          90  (优秀)
  化学          88  (良好)
---------------------------------------------
  总分：453
  平均分：90.6
  最高分：数学 (98分)
  最低分：英语 (85分)
=============================================
```

### 7.3 挑战版本：联系人管理系统

```python
# === 简易通讯录 ===

contacts = []

def add_contact():
    """添加联系人"""
    name = input("请输入姓名：")
    phone = input("请输入电话：")
    email = input("请输入邮箱：")
    contacts.append({
        "姓名": name,
        "电话": phone,
        "邮箱": email
    })
    print(f"✅ 成功添加联系人：{name}")

def show_contacts():
    """显示所有联系人"""
    if not contacts:
        print("📭 通讯录为空")
        return
    print("\n" + "=" * 50)
    print(f"  {'序号':<5}{'姓名':<8}{'电话':<15}{'邮箱'}")
    print("-" * 50)
    for i, contact in enumerate(contacts, 1):
        print(f"  {i:<5}{contact['姓名']:<8}{contact['电话']:<15}{contact['邮箱']}")
    print("=" * 50)
    print(f"  共 {len(contacts)} 个联系人")

def search_contact():
    """搜索联系人"""
    keyword = input("请输入搜索关键词：")
    found = [c for c in contacts if keyword in c["姓名"] or keyword in c["电话"]]
    if found:
        for contact in found:
            print(f"  找到：{contact['姓名']} - {contact['电话']} - {contact['邮箱']}")
    else:
        print("  未找到匹配的联系人")

# 演示使用
print("欢迎使用简易通讯录！")
add_contact()
add_contact()
show_contacts()
search_contact()
```

### 7.4 练习题

**练习1：** 修改上面的个人信息卡片，添加以下功能：
- 输入自己的真实信息
- 添加一个"个人简介"字段
- 计算从出生到现在过了多少天

**练习2：** 创建一个"图书管理"程序：
- 用列表存储书籍信息
- 每本书是一个字典（书名、作者、价格、是否已读）
- 实现添加、查看、统计已读数量的功能

---

## 8. 常见错误与调试

### 8.1 语法错误（SyntaxError）

语法错误是最常见的错误，通常是写错了代码格式。

```python
# 错误1：忘记引号
name = 小明      # SyntaxError: invalid syntax
# 正确：
name = "小明"

# 错误2：括号不匹配
print("hello"    # SyntaxError: unexpected EOF while parsing
# 正确：
print("hello")

# 错误3：冒号遗漏
if x > 5         # SyntaxError: invalid syntax
    print("big")
# 正确：
if x > 5:
    print("big")
```

### 8.2 名称错误（NameError）

使用了未定义的变量。

```python
# 错误：变量未定义
print(age)       # NameError: name 'age' is not defined
# 正确：先定义再使用
age = 20
print(age)
```

### 8.3 类型错误（TypeError）

不同类型的数据不能直接运算。

```python
# 错误：字符串和数字不能相加
name = "小明"
age = 20
print(name + age)  # TypeError: can only concatenate str to str

# 正确：先转换类型
print(name + str(age))     # 小明20
print(f"{name}{age}")       # 小明20（推荐用f-string）
```

### 8.4 索引错误（IndexError）

访问了列表中不存在的位置。

```python
# 错误：索引超出范围
fruits = ["苹果", "香蕉"]
print(fruits[5])   # IndexError: list index out of range
# 列表只有2个元素，索引最大是1

# 正确：检查长度
if len(fruits) > 5:
    print(fruits[5])
```

### 8.5 键错误（KeyError）

字典中不存在该键。

```python
# 错误：键不存在
student = {"姓名": "小明", "年龄": 20}
print(student["成绩"])  # KeyError: '成绩'

# 正确：先检查是否存在
if "成绩" in student:
    print(student["成绩"])

# 或者用get()方法，键不存在时返回默认值
print(student.get("成绩", "暂无成绩"))
```

### 8.6 调试技巧

**技巧1：用 print() 检查变量**

```python
x = 10
y = 20
# 调试：看看每个步骤的值
print(f"x = {x}")
print(f"y = {y}")
print(f"x + y = {x + y}")
```

**技巧2：分步骤写代码，先运行看看对不对**

```python
# 不要一次写完整个程序，分步骤来：
# 第一步：先定义变量
name = "小明"
age = 20

# 第二步：测试一下变量对不对
print(name, age)

# 第三步：再加功能
message = f"我叫{name}，{age}岁"
print(message)
```

**技巧3：读懂错误信息**

```
Traceback (most recent call last):
  File "test.py", line 3, in <module>
    print(age)
NameError: name 'age' is not defined
```

从下往上看：
- `NameError`：错误类型
- `name 'age' is not defined`：错误原因
- `line 3`：第3行出错
- `File "test.py"`：哪个文件出错

**技巧4：使用注释标记问题区域**

```python
# TODO: 这里有问题，需要调试
# problem_area = something_wrong()

# DEBUG: 检查这个变量
# print(f"debug: variable = {variable}")
```

### 8.7 练习题

**练习1：** 找出下面代码的错误并修复：

```python
# 代码有3个错误，找出来并修复
name = '小明
age = "25"
print(name + age)
```

**练习2：** 写一段代码，接受用户输入两个数字，输出它们的和。考虑用户可能输入非数字的情况（提示：可以用 try-except）。

---

## 总结

恭喜你完成了Python基础的学习！让我们回顾一下今天学到的内容：

| 知识点 | 要点 |
|--------|------|
| 变量 | 给数据起名字，用 `=` 赋值 |
| 数据类型 | 整数int、浮点数float、字符串str、布尔值bool |
| 运算符 | 算术`+-*/`、比较`==!=><`、逻辑`and or not` |
| 列表 | `[元素1, 元素2]`，有序，可修改 |
| 元组 | `(元素1, 元素2)`，有序，不可修改 |
| 字典 | `{键:值}`，键值对，快速查找 |
| 集合 | `{元素}`，无序，自动去重 |
| 字符串操作 | f-string格式化、split、join、strip等 |
| 调试 | 读懂错误信息，用print检查，分步骤调试 |

### 下一步学习建议

1. **条件语句**：`if-elif-else`，让程序学会做判断
2. **循环语句**：`for`和`while`，让程序重复执行任务
3. **函数**：把代码封装起来，像乐高积木一样复用
4. **文件操作**：读写文件，处理真实世界的数据

> 编程就像学骑自行车，看再多教程都不如亲自上手试一试。多敲代码，多犯错，多调试，你就会越来越熟练！

---

## 附录：Python关键字速查表

以下是Python中不能用作变量名的关键字：

```
False      await      else       import     pass
None       break      except     in         raise
True       class      finally    is         return
and        continue   for        lambda     try
as         def        from       nonlocal   while
assert     del        global     not        with
async      elif       if         or         yield
```

---

> 本教程由 AI 辅助生成，旨在帮助零基础读者入门Python编程。如有疑问，欢迎查阅 Python 官方文档或在网上搜索相关教程。

---

## 📚 相关笔记

### 下一步学习
- [[01b-Python基础-函数与模块]] - 学习函数和模块
- [[01c-Python基础-文件操作与异常处理]] - 学习文件操作

### 数学基础（可并行学习）
- [[02a-数学基础-线性代数]] - 向量和矩阵
- [[02b-数学基础-概率论与统计]] - 概率论基础

### 深度学习中的应用
- [[04c-深度学习-PyTorch实战]] - 用Python构建神经网络
- [[05a-NLP基础-文本表示]] - 文本向量化

---

> 🏷️ 标签：#Python #编程基础 #入门
