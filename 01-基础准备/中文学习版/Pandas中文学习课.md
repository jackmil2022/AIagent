---
title: "Pandas 中文学习课"
aliases:
  - "Joyful Pandas 中文学习版"
  - "Pandas 数据处理入门"
tags: [pandas, 数据处理, 中文教材]
source: "datawhalechina/joyful-pandas"
---

# Pandas 中文学习课

对应源文件：[[../源文件/GitHub原文/joyful-pandas/README|Joyful Pandas]]

Pandas 是 Python 里处理表格数据的核心工具。机器学习里，表格数据几乎都会先变成 `DataFrame` 再进入特征工程和建模。

## 第 1 课：三个核心对象

| 对象 | 中文理解 | 例子 |
|---|---|---|
| `Series` | 一列数据 | 一列房价 |
| `DataFrame` | 一张二维表 | 一份房屋数据表 |
| `Index` | 行或列标签 | 行号、列名 |

理解方式：

- `Series` 像 Excel 里的一列；
- `DataFrame` 像 Excel 里的一张表；
- `Index` 像表格的行号或索引。

## 第 2 课：先会读表

```python
import pandas as pd

df = pd.read_csv("data.csv")

print(df.shape)      # 行数、列数
print(df.head())     # 前 5 行
print(df.columns)    # 列名
print(df.dtypes)     # 每列类型
print(df.info())     # 更完整的字段信息
```

最先回答的 5 个问题：

1. 数据有多少行多少列？
2. 哪列是标签？
3. 哪些是数值列？
4. 哪些是类别列？
5. 是否有缺失值？

## 第 3 课：选列和选行

### 选一列

```python
price = df["price"]
```

### 选多列

```python
X = df[["area", "rooms", "age"]]
```

### 按位置选

```python
df.iloc[:5, :3]
```

### 按标签选

```python
df.loc[:4, ["area", "price"]]
```

记忆：

- `loc` 看标签；
- `iloc` 看位置。

## 第 4 课：条件筛选

```python
adult = df[df["age"] >= 18]
high_price = df[df["price"] > 200]
```

多个条件：

```python
subset = df[(df["age"] >= 18) & (df["city"] == "Beijing")]
```

常见错误：

- 用 `and` / `or` 代替 `&` / `|`；
- 忘记给每个条件加括号。

## 第 5 课：缺失值处理

先统计：

```python
print(df.isna().sum().sort_values(ascending=False))
```

删除缺失：

```python
df_drop = df.dropna()
```

填充缺失：

```python
df["age"] = df["age"].fillna(df["age"].median())
df["city"] = df["city"].fillna("未知")
```

新手先用这两条规则：

- 数值列：优先中位数；
- 类别列：优先 `"未知"`。

## 第 6 课：排序和去重

```python
df.sort_values("price", ascending=False).head(10)
df.drop_duplicates()
```

典型问题：

- 最贵的 10 套房子是谁？
- 有没有重复订单？
- 哪些记录明显重复？

## 第 7 课：分组聚合

这是表格分析最常用的能力。

```python
df.groupby("city")["price"].mean()
df.groupby("city")["price"].agg(["mean", "max", "min", "count"])
```

多列聚合：

```python
df.groupby("city")[["price", "area"]].mean()
```

例子：

- 每个城市平均房价；
- 每个用户平均消费；
- 每个类别样本数。

## 第 8 课：合并表

现实项目常常不止一张表。

```python
merged = pd.merge(left, right, on="id", how="left")
```

`how` 含义：

| 写法 | 理解 |
|---|---|
| `left` | 保留左表全部 |
| `right` | 保留右表全部 |
| `inner` | 只保留两边都有的 |
| `outer` | 两边都保留，缺失补空 |

先记住 `left join` 就够。

## 第 9 课：快速探索模板

```python
import pandas as pd

df = pd.read_csv("data.csv")

print("shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe())
print(df.isna().sum().sort_values(ascending=False).head(10))
print(df.select_dtypes(include="number").columns)
print(df.select_dtypes(exclude="number").columns)
```

## 第 10 课：最常用操作速查

| 任务 | 写法 |
|---|---|
| 读 CSV | `pd.read_csv("data.csv")` |
| 看前 5 行 | `df.head()` |
| 看后 5 行 | `df.tail()` |
| 看缺失值 | `df.isna().sum()` |
| 选一列 | `df["price"]` |
| 选多列 | `df[["area", "price"]]` |
| 筛选 | `df[df["price"] > 100]` |
| 排序 | `df.sort_values("price")` |
| 去重 | `df.drop_duplicates()` |
| 分组平均 | `df.groupby("city")["price"].mean()` |
| 合并表 | `pd.merge(a, b, on="id")` |
| 保存 CSV | `df.to_csv("out.csv", index=False)` |

## 实战练习

用 Titanic 或房价数据完成：

1. 输出每列缺失值；
2. 区分数值列和类别列；
3. 统计每个类别的平均标签；
4. 找出标签最高和最低的 10 条样本；
5. 找出是否有重复值；
6. 写 5 条中文发现。

## 学完去哪里

1. [[机器学习数学中文预备课]]
2. [[../../02-机器学习/机器学习零基础中文速学]]
3. [[../../02-机器学习/机器学习核心知识中文教程]]
4. [[../../91-项目实战/01-房价预测/房价预测项目说明]]

