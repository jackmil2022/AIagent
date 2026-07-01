---
aliases:
  - "00-PyTorch教程索引"
title: "PyTorch教程索引"
description: "从零开始学习PyTorch深度学习框架"
tags: [PyTorch, Deep-Learning, Index, Tutorial]
---
aliases:
  - "00-PyTorch教程索引"

# 🚀 PyTorch教程索引

> **从零开始，一步步掌握PyTorch**

---
aliases:
  - "00-PyTorch教程索引"

## 📝 关于本教程

本教程基于 [PyTorch_Tutorial](https://github.com/TingsongYu/PyTorch_Tutorial) 整理，专为**零基础**学习者设计：

- ✅ **通俗易懂** — 用生活例子解释复杂概念
- ✅ **配套代码** — 每个知识点都有可运行的代码
- ✅ **循序渐进** — 从简单到复杂，不跳步
- ✅ **实战导向** — 最终能独立训练模型

---
aliases:
  - "00-PyTorch教程索引"

## 📚 学习路径

```
Tensor基础（积木）
    ↓
数据处理（食材）
    ↓
模型构建（菜谱）
    ↓
损失函数（评分标准）
    ↓
优化器（改进方法）
    ↓
训练循环（做菜过程）
    ↓
模型部署（上菜）
```

---
aliases:
  - "00-PyTorch教程索引"

## 🗂️ 教程目录

### 第一阶段：基础入门

| 章节 | 内容 | 预计时间 |
|------|------|----------|
| [[01-PyTorch基础-Tensor张量]] | Tensor是什么、创建、运算、GPU加速 | 1小时 |
| [[02-PyTorch基础-Autograd自动求导]] | �度是什么、自动求导机制 | 1小时 |
| [[03-PyTorch数据处理-Dataset与DataLoader]] | 数据从哪来、如何批量加载 | 1.5小时 |

### 第二阶段：模型构建

| 章节 | 内容 | 预计时间 |
|------|------|----------|
| [[04-PyTorch模型构建-nn.Module]] | 神经网络怎么搭、层是什么 | 1.5小时 |
| [[05-PyTorch损失函数大全]] | 模型好不好、怎么评判 | 1小时 |
| [[06-PyTorch优化器详解]] | 模型怎么学、学习率是什么 | 1小时 |

### 第三阶段：实战训练

| 章节 | 内容 | 预计时间 |
|------|------|----------|
| [[07-PyTorch完整训练流程]] | 训练一个完整模型的步骤 | 2小时 |
| [[08-PyTorch模型保存与加载]] | 训练好的模型怎么保存、怎么用 | 1小时 |
| [[09-PyTorch可视化与调试]] | 看看模型学到了什么 | 1.5小时 |

### 第四阶段：进阶应用

| 章节 | 内容 | 预计时间 |
|------|------|----------|
| [[10-PyTorch实战-CIFAR10图像分类]] | 完整项目实战 | 2小时 |
| [[11-PyTorch实战-迁移学习]] | 站在巨人肩膀上 | 1.5小时 |

---
aliases:
  - "00-PyTorch教程索引"

## 🎯 学习建议

### 零基础学习者

1. **不要跳过基础** — Tensor和Autograd是后续一切的基础
2. **多敲代码** — 看懂≠会写，必须动手
3. **不怕报错** — 错误是最好的老师
4. **先跑通再理解** — 先让代码跑起来，再研究为什么

### 有编程基础但没学过深度学习

1. **重点理解概念** — 梯度、损失、优化是什么
2. **类比学习** — 把神经网络想象成做菜
3. **画图辅助** — 用TensorBoard可视化

---
aliases:
  - "00-PyTorch教程索引"

## 🛠️ 环境准备

### 安装PyTorch

```bash
# CPU版本（推荐先用这个）
pip install torch torchvision

# GPU版本（需要CUDA支持）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 验证安装

```python
import torch

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU数量: {torch.cuda.device_count()}")

# 简单测试
x = torch.randn(3, 3)
print(f"Tensor形状: {x.shape}")
```

---
aliases:
  - "00-PyTorch教程索引"

## 📖 参考资源

- [PyTorch官方文档](https://pytorch.org/docs/stable/)
- [PyTorch官方教程](https://pytorch.org/tutorials/)
- [PyTorch_Tutorial（本教程来源）](https://github.com/TingsongYu/PyTorch_Tutorial)
- [PyTorch中文文档](https://pytorch.apachecn.org/)

---
aliases:
  - "00-PyTorch教程索引"

## 🎯 下一步

- 从 [[01-PyTorch基础-Tensor张量]] 开始学习
- 建议按照目录顺序逐步学习

---
aliases:
  - "00-PyTorch教程索引"

> 💡 **记住**：深度学习没有捷径，但有方法。跟着教程一步步来，你一定能学会！
