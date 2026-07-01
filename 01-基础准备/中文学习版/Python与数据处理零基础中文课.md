---
title: "Python 与数据处理零基础中文课"
aliases:
  - "Python 数据科学零基础"
  - "Python 数据处理入门"
tags: [Python, 数据处理, 数据科学, 中文教材, 零基础]
source: "microsoft/Data-Science-For-Beginners"
---

# Python 与数据处理零基础中文课

对应源文件：

- [[../源文件/GitHub原文/Data-Science-For-Beginners/README|Microsoft Data Science for Beginners 总课程]]
- [[../源文件/GitHub原文/Data-Science-For-Beginners/2-Working-With-Data/07-python/README|原文章节：Working with Python]]
- [[../源文件/GitHub原文/Data-Science-For-Beginners/2-Working-With-Data/08-data-preparation/README|原文章节：Data Preparation]]
- [[../源文件/GitHub原文/Data-Science-For-Beginners/3-Data-Visualization/09-visualization-quantities/README|原文章节：Visualizing Quantities]]
- [[../源文件/GitHub原文/Data-Science-For-Beginners/examples/README|原文示例目录]]

> [!note]
> 这篇笔记按中文学习习惯重写，核心内容来自上面的本地课程原文，但不是逐句硬翻。目标是让零基础读者先学会，再回头看英文原文。

这门课的目标不是把 Python 学成专家，而是让你能进入机器学习：

1. 会写最基本的 Python；
2. 会读 CSV 和表格数据；
3. 会用 `pandas` 做数据体检；
4. 会处理缺失值和重复值；
5. 会画基础图表；
6. 会写一份中文数据分析结论。

## 学习顺序

```mermaid
graph LR
    A[理解数据科学在做什么] --> B[学会 Python 和 pandas]
    B --> C[读入并理解表格数据]
    C --> D[清洗缺失值与重复值]
    D --> E[画图观察规律]
    E --> F[写结论]
    F --> G[进入机器学习]
```

## 第 1 章：为什么数据科学里常用 Python

原文章节说明了一个很现实的事实：数据库适合存数据和查数据，但很多“灵活处理数据”的工作，还是要靠编程。

比如下面这些事情，用程序做通常更方便：

- 批量清洗很多文件；
- 把多个表拼起来再做统计；
- 按规则生成新字段；
- 画图看异常值和分布；
- 为后面的机器学习做准备。

原文提到数据科学里常见三种语言：

| 语言 | 中文理解 | 适合谁 |
|---|---|---|
| `Python` | 通用编程语言，生态最丰富 | 零基础入门首选 |
| `R` | 统计分析传统强项 | 偏统计学背景的人 |
| `Julia` | 科学计算性能较强 | 进阶后可了解 |

对当前学习阶段来说，你只要抓住一句话：

> 机器学习之前，大部分时间都在“读数据、改数据、看数据”，而 Python 正好很适合做这件事。

## 第 2 章：数据科学常见的三类数据

原文章节把数据分成三类：

| 数据类型 | 例子 | 机器学习前要做什么 |
|---|---|---|
| 表格数据 | Excel、CSV、数据库导出表 | 清洗、筛选、统计、画图 |
| 文本数据 | 评论、论文、文章、客服记录 | 提取关键词、统计频率、做分类 |
| 图像数据 | 照片、截图、医学影像 | 提取对象、分类、检测 |

零基础阶段先只专注第一类：**表格数据**。

因为机器学习里最常见的起点就是一张表：

- 每一行是一条样本；
- 每一列是一个字段；
- 某一列通常是我们要预测的目标。

## 第 3 章：先认识四个基础库

原文重点介绍了四个常见库：

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

| 库            | 作用        | 初学者怎么理解          |
| ------------ | --------- | ---------------- |
| `pandas`     | 处理表格数据    | Excel 的 Python 版 |
| `numpy`      | 处理数组和数值计算 | 更底层、更快的数值工具      |
| `matplotlib` | 画图        | 基础绘图库            |
| `scipy`      | 科学计算      | 进阶再深入            |

当前阶段最重要的是：

- 用 `pandas` 读表；
- 用 `pandas` 清洗；
- 用 `matplotlib` 画图。

## 第 4 章：Series 是什么

原文先讲 `Series`。你可以把它理解成“**带索引的一列数据**”。

普通列表只关心顺序，`Series` 不只关心顺序，还关心每个值对应的索引。

```python
import pandas as pd
import numpy as np

idx = pd.date_range("2020-01-01", "2020-01-07")
sales = pd.Series(np.random.randint(25, 50, size=len(idx)), index=idx)
print(sales)
```

为什么 `Series` 重要？

因为很多真实数据都不是单纯的数字列表，而是“某一天对应一个值”“某个城市对应一个值”“某个商品对应一个值”。

原文有一个关键点很值得记住：**Series 做运算时，会按索引对齐**。

```python
extra = pd.Series(10, index=pd.date_range("2020-01-01", "2020-01-07", freq="2D"))
total = sales.add(extra, fill_value=0)
```

这里的 `fill_value=0` 很重要。因为有些日期只出现在 `sales` 里，没有出现在 `extra` 里，如果不补默认值，就容易得到 `NaN`。

这就是初学者经常遇到的第一个坑：

- 你以为是在“两个列表相加”；
- 其实 `pandas` 在按索引对齐后再相加。

## 第 5 章：DataFrame 是什么

`DataFrame` 就是一张二维表，本质上可以理解为“多个 `Series` 拼在一起”。

```python
import pandas as pd

df = pd.DataFrame({
    "area": [80, 95, 120],
    "rooms": [2, 3, 4],
    "price": [200, 260, 380]
})
print(df)
```

输出可以理解成：

| 行号 | area | rooms | price |
|---|---:|---:|---:|
| 0 | 80 | 2 | 200 |
| 1 | 95 | 3 | 260 |
| 2 | 120 | 4 | 380 |

在机器学习前，绝大多数数据处理都围绕 `DataFrame` 展开。

## 第 6 章：DataFrame 最常见操作

这一章基本就是原文 `07-python` 里最核心的部分。

### 1. 选列

```python
df["price"]
df[["area", "price"]]
```

- `df["price"]` 返回一列；
- `df[["area", "price"]]` 返回一个子表。

### 2. 按条件筛选行

```python
df[df["price"] > 220]
```

多个条件时，要用 `&` 和括号：

```python
df[(df["price"] > 220) & (df["rooms"] >= 3)]
```

不要写成：

```python
# 错误示例
# df[df["price"] > 220 and df["rooms"] >= 3]
```

因为 `pandas` 这里不是普通 Python 布尔值，而是一整列布尔结果。

### 3. 新建计算列

```python
df["price_per_room"] = df["price"] / df["rooms"]
```

原文强调了一点：列运算通常应该是“整列对整列”，而不是混用普通 Python 的单值逻辑。

### 4. 用 `apply` 处理复杂逻辑

```python
df["room_level"] = df["rooms"].apply(lambda x: "大户型" if x >= 4 else "普通")
```

当你要逐个元素处理时，`apply` 很常用。

### 5. 用 `iloc` 按位置取行

```python
df.iloc[:2]
```

就是取前两行。

### 6. 分组统计

```python
city_df = pd.DataFrame({
    "city": ["北京", "北京", "上海", "上海", "上海"],
    "price": [200, 260, 300, 330, 360]
})

city_df.groupby("city")["price"].mean()
```

这一步很重要，因为很多数据分析最后都要回答类似问题：

- 每个城市平均房价是多少？
- 每个类别样本数多少？
- 每个用户平均消费多少？

## 第 7 章：读入数据后第一件事该做什么

原文和示例都在强调一个习惯：**不要一上来建模，先看数据本身**。

最常用的第一轮检查代码如下：

```python
import pandas as pd

df = pd.read_csv("data.csv")

print(df.shape)
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())
```

你要先回答 5 个问题：

| 问题 | 对应写法 |
|---|---|
| 数据有多少行多少列？ | `df.shape` |
| 前几行长什么样？ | `df.head()` |
| 后几行是否正常？ | `df.tail()` |
| 每列是什么类型？ | `df.info()` |
| 数值列的范围和均值如何？ | `df.describe()` |

如果一张表你连这些都还没看，就开始训练模型，后面大概率会返工。

## 第 8 章：为什么要做数据清洗

`08-data-preparation` 这一章的核心可以总结成三句话：

1. 脏数据会让分析结果失真；
2. 脏数据会让模型效果下降；
3. 脏数据会让团队后续复用更困难。

原文强调的数据清洗价值主要有三个：

| 价值 | 中文解释 |
|---|---|
| 易于使用和复用 | 结构统一后，更容易搜索、共享、重复使用 |
| 保持一致性 | 多个数据源合并时，更不容易乱 |
| 提高模型准确率 | 输入更干净，模型更稳定 |

常见清洗目标有四类：

| 目标 | 典型问题 |
|---|---|
| 探索数据 | 先发现哪里不正常 |
| 统一格式 | 日期、空格、数值类型不一致 |
| 去重 | 同一条数据出现多次 |
| 处理缺失值 | 有些位置根本没有值 |

## 第 9 章：怎么查看缺失值

原文先讲了 `NaN` 和 `None`。

- `NaN` 常见于数值型缺失；
- `None` 常见于 Python 对象缺失；
- 在 `pandas` 里，它们都意味着“这里没有有效数据”。

最常用检查方式：

```python
df.isnull().sum()
```

或者：

```python
df.isna().sum().sort_values(ascending=False)
```

小例子：

```python
import pandas as pd
import numpy as np

example = pd.Series([0, np.nan, "", None])
print(example.isnull())
```

这里有个初学者容易误会的地方：

- `0` 不是缺失值；
- `""` 空字符串也不一定算缺失值；
- `NaN` 和 `None` 才是典型缺失标记。

## 第 10 章：缺失值怎么处理

原文给出了三类最常见做法。

### 1. 删除缺失

```python
df.dropna()
```

适合：

- 缺失行很少；
- 删掉后对整体影响不大。

### 2. 按列删除

```python
df.dropna(axis="columns")
```

适合：

- 某列几乎没法用；
- 缺失过多，保留意义不大。

### 3. 填充缺失

```python
df["age"] = df["age"].fillna(df["age"].median())
df["city"] = df["city"].fillna("未知")
```

也可以前向填充或后向填充：

```python
df.fillna(method="ffill")
df.fillna(method="bfill")
```

给零基础学习者的简单规则：

| 情况 | 先用什么办法 |
|---|---|
| 数值列偶尔缺失 | 中位数填充 |
| 类别列偶尔缺失 | `"未知"` 填充 |
| 时间序列连续数据 | 可尝试 `ffill` / `bfill` |
| 缺失比例极高 | 考虑删列 |

> [!warning]
> 缺失值没有“统一标准答案”。你要先理解它为什么缺，再决定是删、补，还是单独保留成一个信息信号。

## 第 11 章：怎么处理重复值

真实数据里，重复数据很常见。比如：

- 同一个订单重复导出；
- 两个表合并后重复；
- 表单重复提交；
- 人工录入多次。

原文给出的两个核心方法：

```python
df.duplicated()
df.drop_duplicates()
```

例子：

```python
example = pd.DataFrame({
    "letters": ["A", "B", "A", "B", "B"],
    "numbers": [1, 2, 1, 3, 3]
})

print(example.duplicated())
print(example.drop_duplicates())
```

如果你只想根据某几列判断重复：

```python
example.drop_duplicates(["letters"])
```

这说明“是否重复”取决于业务定义，不一定非要全列完全一样才算重复。

## 第 12 章：为什么要画图

原文 `09-visualization-quantities` 的核心思想很直接：

> 只看数字表格不够，你需要把数据画出来，才能更快发现数量关系、分布和异常点。

常见图形选择：

| 图 | 适合什么问题 |
|---|---|
| 折线图 | 看时间趋势 |
| 条形图 | 比较不同类别数量 |
| 散点图 | 看异常点或变量关系 |
| 饼图 | 看占比，少量类别时可用 |

一个很重要的经验是：**先选合适的图，再讨论结论**。

## 第 13 章：用图发现异常值

原文用鸟类翼展数据演示了一个很典型的过程：

1. 先画图；
2. 发现极端异常值；
3. 怀疑是录入错误；
4. 过滤异常值后再继续分析。

这比死盯着原始表格更高效。

下面是适合零基础理解的简化版：

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

plt.scatter(df["area"], df["price"])
plt.xlabel("area")
plt.ylabel("price")
plt.show()
```

如果有某些点离群特别严重，你要先问：

- 这是正常的大值吗？
- 还是录错了一个 0？
- 是单位不统一吗？
- 还是少数极端样本本来就应该保留？

不要看到异常值就直接删。先判断业务含义。

## 第 14 章：零基础必做的一次完整练习

请找一个 CSV 文件，按下面流程走一遍。

### 练习目标

把一份陌生数据从“完全看不懂”变成“能说出基本结论”。

### 练习步骤

1. `pd.read_csv()` 读入数据；
2. 看 `shape`、`head()`、`tail()`、`info()`、`describe()`；
3. 区分数值列、类别列；
4. 统计缺失值；
5. 统计重复值；
6. 画至少 3 张图；
7. 写出 5 条观察；
8. 写出 3 个数据问题；
9. 写出下一步建模思路。

### 验收标准

- 你能说清每一列大概在表示什么；
- 你知道哪些列质量差；
- 你知道哪些图能说明问题；
- 你能写出一段中文结论。

## 第 15 章：可直接复用的代码模板

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

print("数据规模:", df.shape)
print("\n前 5 行:")
print(df.head())
print("\n后 5 行:")
print(df.tail())
print("\n字段信息:")
print(df.info())
print("\n描述统计:")
print(df.describe())
print("\n缺失值统计:")
print(df.isna().sum().sort_values(ascending=False))
print("\n重复行数量:", df.duplicated().sum())

num_cols = df.select_dtypes(include="number").columns
cat_cols = df.select_dtypes(exclude="number").columns

print("\n数值列:", list(num_cols))
print("\n类别列:", list(cat_cols))

if len(num_cols) > 0:
    df[num_cols].hist(figsize=(10, 8))
    plt.tight_layout()
    plt.show()
```

## 第 16 章：中文数据体检报告模板

```markdown
# 数据体检报告

## 1. 数据主题
这份数据描述了……

## 2. 数据规模
- 行数：
- 列数：

## 3. 字段理解
- 标签列：
- 重要特征：

## 4. 缺失值与重复值
- 缺失最多的列：
- 重复行数量：
- 处理建议：

## 5. 图表观察
1. ...
2. ...
3. ...

## 6. 数据问题
1. ...
2. ...
3. ...

## 7. 下一步建模想法
我会先尝试……
```

## 本地原文与示例入口

### 对应原文章节

1. [[../源文件/GitHub原文/Data-Science-For-Beginners/2-Working-With-Data/07-python/README|Working with Python]]
2. [[../源文件/GitHub原文/Data-Science-For-Beginners/2-Working-With-Data/08-data-preparation/README|Data Preparation]]
3. [[../源文件/GitHub原文/Data-Science-For-Beginners/3-Data-Visualization/09-visualization-quantities/README|Visualizing Quantities]]

### 本地示例脚本

1. [[../源文件/GitHub原文/Data-Science-For-Beginners/examples/01_hello_world_data_science.py|01-第一个数据科学程序]]
2. [[../源文件/GitHub原文/Data-Science-For-Beginners/examples/02_loading_data.py|02-读入与查看数据]]
3. [[../源文件/GitHub原文/Data-Science-For-Beginners/examples/03_simple_analysis.py|03-基础分析]]
4. [[../源文件/GitHub原文/Data-Science-For-Beginners/examples/04_basic_visualization.py|04-基础可视化]]
5. [[../源文件/GitHub原文/Data-Science-For-Beginners/examples/05_real_world_example.py|05-完整案例]]

## 学完去哪里

1. [[Pandas中文学习课]]
2. [[机器学习数学中文预备课]]
3. [[数据挖掘项目中文入门]]
4. [[../../02-机器学习/机器学习零基础中文速学]]
5. [[../../91-项目实战/01-房价预测/房价预测项目说明]]
