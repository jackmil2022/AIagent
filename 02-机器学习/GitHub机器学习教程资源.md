---
title: "GitHub 机器学习教程资源"
aliases:
  - "机器学习 GitHub 教程"
  - "ML GitHub Resources"
tags: [机器学习, GitHub, 教程, 资源]
---

# GitHub 机器学习教程资源

这份清单只收“适合学习路径”的仓库，不追求数量。选择标准：

- 有清晰目录或课程结构；
- 面向初学者或进阶学习者；
- 有代码、Notebook、练习或项目；
- 内容覆盖经典机器学习，而不是只讲大模型；
- 优先选择维护时间长、社区使用多的项目。

> [!warning]
> GitHub 仓库适合跟着做，不适合只收藏。每看一个教程，至少跑通一个 Notebook 或复现一个小项目。

> [!tip]
> 不知道从哪里开始时，先按 [[机器学习从零到高手学习路径]] 走；本页只负责解释每个 GitHub 教程适合放在哪个阶段。

## 1. 主线课程

| 仓库 | 适合阶段 | 内容重点 | 怎么用 |
|---|---|---|---|
| [[90-资料库/01-GitHub原文/机器学习/ML-For-Beginners/README|microsoft/ML-For-Beginners]] | 零基础到入门 | 12 周、26 课，经典机器学习，主要使用 Scikit-learn | 当第一条主线，每天 1 小节，先建立完整地图 |
| [[90-资料库/01-GitHub原文/机器学习/zero-to-mastery-ml/README|mrdbourke/zero-to-mastery-ml]] | 入门到实战 | Python、数据分析、Scikit-learn、TensorFlow、项目 Notebook | 跟着 Notebook 敲，适合补实践手感 |
| [[90-资料库/01-GitHub原文/机器学习/handson-ml3/README|ageron/handson-ml3]] | 入门后进阶 | 《Hands-On Machine Learning》第 3 版配套 Notebook，Scikit-learn、Keras、TensorFlow | 学完基础后按章节复现，重点看端到端项目 |
| [[90-资料库/01-GitHub原文/机器学习/tutorials-scikit-learn/README|glouppe/tutorials-scikit-learn]] | 入门 | Scikit-learn 初级教程、稳健估计和校准 | 想快速熟悉 sklearn API 时看 |

## 2. 从零实现算法

| 仓库 | 适合阶段 | 内容重点 | 怎么用 |
|---|---|---|---|
| [[90-资料库/01-GitHub原文/机器学习/ML-From-Scratch/README|eriklindernoren/ML-From-Scratch]] | 学完基础后 | 用 Python 从零实现常见机器学习算法 | 用来理解算法内部，不要当第一门课 |
| [[90-资料库/01-GitHub原文/机器学习/MLfromscratch/README|patrickloeber/MLfromscratch]] | 入门后 | KNN、线性回归、Logistic 回归、朴素贝叶斯、SVM、决策树、随机森林、PCA、K-Means 等 | 每学一个算法后，对照看一次实现 |
| [[90-资料库/01-GitHub原文/机器学习/Implementation-of-Machine-Learning-Algorithm-from-Scratch/README|ghimiresunil/Implementation-of-Machine-Learning-Algorithm-from-Scratch]] | 入门后 | 从基础到进阶的机器学习算法 Python 实现 | 作为补充实现参考 |

## 3. 路线图和资源导航

| 仓库 | 适合阶段 | 内容重点 | 怎么用 |
|---|---|---|---|
| [[90-资料库/01-GitHub原文/机器学习/Machine-Learning-Roadmap/README|mlacademyai/Machine-Learning-Roadmap]] | 规划阶段 | Python、数据科学、机器学习、深度学习、MLOps 学习路线 | 只用来定路线，不要陷入无限找资源 |
| [[90-资料库/01-GitHub原文/机器学习/data-science-roadmap/README|Moataz-Elmesmary/Data-Science-Roadmap]] | 规划阶段 | 统计、概率、Python、pandas、NumPy、清洗、可视化 | 补数据科学基础 |
| [[90-资料库/01-GitHub原文/机器学习/AI-ML-Roadmap-from-scratch/README|aadi1011/AI-ML-Roadmap-from-scratch]] | 从 AI 全栈视角规划 | 数学、数据科学、机器学习、CV、DL、GenAI、NLP、RL、Agentic AI | 看大方向，别按全量内容硬啃 |
| [[90-资料库/01-GitHub原文/机器学习/start-machine-learning/README|louisfb01/start-machine-learning]] | 零基础规划 | 面向零基础的 ML/AI 学习入口 | 用来补充资源选择 |
| [[90-资料库/01-GitHub原文/机器学习/Machine-Learning-Tutorials/README|ujjwalkarn/Machine-Learning-Tutorials]] | 查资料 | 主题式 ML/DL 教程合集 | 当索引，不当主线课程 |

## 4. 项目和实战

| 仓库 | 适合阶段 | 内容重点 | 怎么用 |
|---|---|---|---|
| [[90-资料库/01-GitHub原文/机器学习/Made-With-ML/README|GokuMohandas/Made-With-ML]] | 有基础后 | 面向生产的机器学习工程：实验、测试、部署、监控 | 学完 sklearn 基础后再看 |
| [[90-资料库/01-GitHub原文/机器学习/Machine-Learning-with-Python/README|tirthajyoti/Machine-Learning-with-Python]] | 入门到进阶 | 回归、分类、聚类、降维、基础神经网络 Notebook | 按专题查 Notebook |
| [[90-资料库/01-GitHub原文/机器学习/machine_learning_tutorials/README|akmand/machine_learning_tutorials]] | 入门 | Python + Scikit-learn 教程 Notebook | 作为轻量补充练习 |
| [[90-资料库/01-GitHub原文/机器学习/Machine-Learning-Basics/README|SamBelkacem/Machine-Learning-Basics]] | 入门 | ML 基础、Notebook、参考指南、cheatsheet | 查漏补缺 |

## 5. 官方和平台资源

| 资源 | 适合阶段 | 内容重点 | 怎么用 |
|---|---|---|---|
| `scikit-learn/scikit-learn` / [官方文档](https://scikit-learn.org/stable/) | 全阶段 | Python 经典机器学习库 | 查 API、看官方例子 |
| [Kaggle Learn: Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning) | 零基础练习 | 决策树、随机森林、模型验证、提交预测 | 快速跑通第一个项目 |
| [DeepLearning.AI Machine Learning Specialization](https://www.deeplearning.ai/courses/machine-learning-specialization/) | 系统补课 | 监督学习、无监督学习、推荐系统、强化学习简介 | 想系统补理论时看 |

## 6. 推荐组合

### 只想最快入门

1. [[机器学习零基础入门]]
2. [Kaggle Learn: Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning)
3. [[90-资料库/01-GitHub原文/机器学习/ML-For-Beginners/README|microsoft/ML-For-Beginners]]
4. [[03-深度学习/01-神经网络与深度学习/chap2机器学习概述/机器学习概述-上|机器学习概述-上]]

### 想系统学经典机器学习

1. [[90-资料库/01-GitHub原文/机器学习/ML-For-Beginners/README|microsoft/ML-For-Beginners]]
2. [DeepLearning.AI Machine Learning Specialization](https://www.deeplearning.ai/courses/machine-learning-specialization/)
3. [[90-资料库/01-GitHub原文/机器学习/handson-ml3/README|ageron/handson-ml3]]
4. [[90-资料库/01-GitHub原文/机器学习/ML-From-Scratch/README|eriklindernoren/ML-From-Scratch]]

### 想偏工程实践

1. [[90-资料库/01-GitHub原文/机器学习/zero-to-mastery-ml/README|mrdbourke/zero-to-mastery-ml]]
2. [[90-资料库/01-GitHub原文/机器学习/handson-ml3/README|ageron/handson-ml3]]
3. [[90-资料库/01-GitHub原文/机器学习/Made-With-ML/README|GokuMohandas/Made-With-ML]]

## 7. 不建议的学习方式

- 不要同时开 5 门课。主线只保留 1 门，其他当参考。
- 不要先从“从零实现算法”开始。先用 sklearn 跑通流程，再看实现。
- 不要直接跳到 MLOps。先会训练、评估、调参，再谈部署。
- 不要只看路线图。路线图不能替你建立模型直觉。
