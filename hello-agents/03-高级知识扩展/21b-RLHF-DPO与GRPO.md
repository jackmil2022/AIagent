---
title: "RLHF：DPO与GRPO"
description: "人类反馈强化学习的新方法"
tags: [RLHF, DPO, GRPO, Alignment]
---

# 🎯 RLHF：DPO与GRPO

> **让AI更符合人类偏好**

---

## 📝 前言

大模型虽然强大，但可能：
- 生成有害内容
- 回答不准确
- 不符合用户期望

**RLHF（人类反馈强化学习）** 就是让模型学会"什么是对的"。

---

## 🔰 1. RLHF 基础回顾

### 1.1 传统RLHF流程

```
步骤1：监督微调（SFT）
    ↓
步骤2：训练奖励模型（RM）
    ↓
步骤3：PPO强化学习优化
```

### 1.2 问题

| 问题 | 说明 |
|------|------|
| 训练复杂 | 需要多个模型协同 |
| 不稳定 | PPO训练容易崩溃 |
| 资源消耗大 | 需要同时维护多个模型 |

---

## 🔰 2. DPO 原理

### 2.1 核心思想

**DPO = Direct Preference Optimization（直接偏好优化）**

跳过奖励模型，直接从偏好数据优化策略模型。

### 2.2 数学原理

传统RLHF的目标：
```
max E[r(x, y)] - β * KL[π(y|x) || π_ref(y|x)]
```

DPO的简化：
```
L_DPO = -E[log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]
```

其中：
- y_w = 被偏好的回答
- y_l = 不被偏好的回答
- π_ref = 参考模型（通常是SFT后的模型）

### 2.3 直观理解

想象你在教孩子：
- **传统RLHF**：先教孩子打分（奖励模型），再根据分数调整行为
- **DPO**：直接告诉孩子"这样做对，那样做错"

### 2.4 优势

| 优势 | 说明 |
|------|------|
| 简单 | 不需要训练奖励模型 |
| 稳定 | 比PPO更稳定 |
| 高效 | 资源消耗更少 |

---

## 🔰 3. DPO 实现

### 3.1 数据格式

```json
{
  "prompt": "解释什么是机器学习",
  "chosen": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习...",
  "rejected": "机器学习就是让计算机自己学习，不需要人教..."
}
```

### 3.2 训练代码

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

# 加载模型
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
ref_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 加载数据
dataset = load_dataset("json", data_files="preference_data.json")

# 配置
training_args = DPOConfig(
    output_dir="./dpo_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=5e-7,
    beta=0.1,  # KL散度系数
    max_length=1024,
    max_prompt_length=512,
)

# 创建训练器
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
)

# 训练
trainer.train()
```

---

## 🔰 4. GRPO 原理

### 4.1 什么是GRPO？

**GRPO = Group Relative Policy Optimization（组相对策略优化）**

DeepSeek提出的创新方法，用于训练推理模型（如DeepSeek-R1）。

### 4.2 核心思想

不需要奖励模型，而是通过**组内对比**来评估回答质量。

### 4.3 工作流程

```
对于每个问题q：
1. 从当前策略采样G个回答：{o_1, o_2, ..., o_G}
2. 用规则奖励评估每个回答：{r_1, r_2, ..., r_G}
3. 计算组内相对优势：A_i = (r_i - mean(r)) / std(r)
4. 更新策略：最大化优势高的回答的概率
```

### 4.4 奖励规则

```python
# 数学题的奖励规则
def math_reward(solution, answer):
    """检查数学答案是否正确"""
    if extract_answer(solution) == answer:
        return 1.0  # 正确
    else:
        return 0.0  # 错误

# 代码题的奖励规则
def code_reward(code, test_cases):
    """检查代码是否通过测试"""
    try:
        exec(code)
        passed = all(run_test(test) for test in test_cases)
        return 1.0 if passed else 0.5
    except:
        return 0.0
```

---

## 🔰 5. GRPO 实现

### 5.1 训练代码

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# 加载模型
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

# 定义奖励函数
def math_reward_func(prompts, completions, **kwargs):
    """数学题奖励函数"""
    rewards = []
    for prompt, completion in zip(prompts, completions):
        # 提取答案
        try:
            # 假设答案在最后
            answer = extract_answer(completion)
            ground_truth = kwargs.get("ground_truth")[0]
            
            if answer == ground_truth:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except:
            rewards.append(0.0)
    return rewards

# 配置
training_args = GRPOConfig(
    output_dir="./grpo_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=1e-6,
    beta=0.04,  # KL惩罚系数
    num_generations=8,  # 每个prompt生成8个回答
    max_completion_length=512,
)

# 创建训练器
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    reward_funcs=[math_reward_func],
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
)

# 训练
trainer.train()
```

---

## 🔰 6. DPO vs GRPO

### 6.1 对比

| 特性 | DPO | GRPO |
|------|-----|------|
| 是否需要奖励模型 | 否 | 否 |
| 偏好数据 | 需要 | 不需要 |
| 奖励信号 | 二元（好/坏） | 规则奖励 |
| 适用场景 | 通用对齐 | 推理任务 |
| 训练稳定性 | 高 | 高 |

### 6.2 选择建议

| 场景 | 推荐方法 |
|------|----------|
| 通用对话对齐 | DPO |
| 数学推理 | GRPO |
| 代码生成 | GRPO |
| 有偏好数据 | DPO |
| 有明确规则 | GRPO |

---

## 🔰 7. 高级技巧

### 7.1 数据质量

```python
# 过滤低质量数据
def filter_data(dataset):
    filtered = []
    for item in dataset:
        # 检查回答长度
        if len(item["chosen"]) < 50:
            continue
        # 检查是否包含有害内容
        if contains_harmful(item["chosen"]):
            continue
        filtered.append(item)
    return filtered
```

### 7.2 多轮训练

```python
# 迭代训练
for round in range(3):
    # 训练一轮
    trainer.train()
    
    # 生成新数据
    new_data = generate_preference_data(model)
    
    # 更新数据集
    dataset = dataset + new_data
```

---

## 📚 本节小结

| 方法 | 核心思想 | 适用场景 |
|------|----------|----------|
| DPO | 直接偏好优化 | 通用对齐 |
| GRPO | 组相对策略优化 | 推理任务 |
| PPO | 近端策略优化 | 复杂任务 |

---

## 🎯 下一步

- **22a - 多模态：视觉语言模型** - 图文理解
- **23 - 智能体通信协议** - MCP、A2A等

---

> 💡 **实践建议**：尝试用DPO训练一个小模型，感受对齐的效果。
