---
title: "微调：LoRA与QLoRA"
description: "参数高效微调的最佳实践"
tags: [Fine-tuning, LoRA, QLoRA, PEFT]
---

# ⚙️ 微调：LoRA与QLoRA

> **用更少的资源，训练更好的模型**

---

## 📝 前言

Full Fine-tuning（全量微调）需要更新模型的**所有参数**，这需要巨大的计算资源。

有没有更高效的方法？

**LoRA** 和 **QLoRA** 就是解决方案：**只训练少量参数，达到接近全量微调的效果**。

---

## 🔰 1. 为什么需要参数高效微调？

### 1.1 Full Fine-tuning 的问题

| 问题 | 说明 |
|------|------|
| 显存需求大 | 7B模型需要约28GB显存 |
| 训练时间长 | 需要大量GPU时间 |
| 存储成本高 | 每个任务都要保存完整模型 |
| 灾难性遗忘 | 可能忘记原有知识 |

### 1.2 参数高效微调（PEFT）的优势

| 优势 | 说明 |
|------|------|
| 显存需求小 | 只需训练0.1%-1%的参数 |
| 训练速度快 | 大幅减少训练时间 |
| 存储成本低 | 只保存适配器权重 |
| 保持能力 | 原有模型能力不受影响 |

---

## 🔰 2. LoRA 原理

### 2.1 核心思想

**低秩适配（Low-Rank Adaptation）**

假设：模型微调时的权重变化是**低秩的**，可以用小矩阵来近似。

### 2.2 数学原理

```
原始权重矩阵：W (d × d)
微调后的权重：W' = W + ΔW

LoRA假设：ΔW = B × A
其中：B (d × r), A (r × d), r << d

参数量对比：
- 原始：d × d = 4096 × 4096 = 16M
- LoRA：d × r + r × d = 4096 × 64 + 64 × 4096 = 0.5M
- 减少了 97%！
```

### 2.3 直观理解

想象你要调整一幅画的颜色：
- **Full Fine-tuning**：重新画整幅画
- **LoRA**：在原画上加一层透明滤镜

### 2.4 架构图

```
输入 x
   │
   ├──→ W (冻结) ──→ h1
   │
   └──→ A ──→ B ──→ h2 (可训练)
   
输出 = h1 + h2
```

---

## 🔰 3. LoRA 实现

### 3.1 使用 Hugging Face PEFT

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_4bit=True  # 4bit量化
)

# 配置LoRA
lora_config = LoraConfig(
    r=16,                    # 秩
    lora_alpha=32,           # 缩放因子
    target_modules=[         # 要适配的层
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 创建LoRA模型
model = get_peft_model(model, lora_config)

# 查看可训练参数
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 3,740,000,000 || trainable%: 0.11%
```

### 3.2 关键参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| r | 低秩矩阵的秩 | 8-64 |
| lora_alpha | 缩放因子 | 2*r |
| target_modules | 要适配的层 | q,k,v,o_proj |
| lora_dropout | Dropout率 | 0.05-0.1 |

---

## 🔰 4. QLoRA 原理

### 4.1 什么是QLoRA？

**QLoRA = Quantized LoRA（量化LoRA）**

在LoRA基础上，对基础模型进行**4bit量化**，进一步减少显存需求。

### 4.2 技术创新

1. **4-bit NormalFloat (NF4)**：优化的4bit量化
2. **双重量化**：量化参数本身也量化
3. **分页优化器**：智能管理显存

### 4.3 显存对比

| 模型 | Full FT | LoRA | QLoRA |
|------|---------|------|-------|
| LLaMA-7B | 28GB | 12GB | 6GB |
| LLaMA-13B | 48GB | 20GB | 10GB |
| LLaMA-70B | 160GB | 60GB | 24GB |

---

## 🔰 5. QLoRA 实战

### 5.1 完整训练代码

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# 1. 配置量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 2. 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# 3. 准备模型
model = prepare_model_for_kbit_training(model)

# 4. 配置LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# 5. 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

# 6. 加载数据集
dataset = load_dataset("json", data_files="train.jsonl")

def preprocess(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

tokenized_dataset = dataset.map(preprocess, batched=True)

# 7. 训练配置
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=100,
    logging_steps=10,
    save_steps=100,
    fp16=True,
    optim="paged_adamw_32bit",
)

# 8. 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
)

# 9. 开始训练
trainer.train()

# 10. 保存LoRA权重
model.save_pretrained("./lora_weights")
```

### 5.2 训练数据格式

```jsonl
{"text": "<|begin_of_text|>用户：你好<|end_of_text|>助手：你好！有什么我可以帮助你的吗？"}
{"text": "<|begin_of_text|>用户：解释量子计算<|end_of_text|>助手：量子计算是利用量子力学原理..."}
```

---

## 🔰 6. 合并与部署

### 6.1 合并LoRA到基础模型

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16
)

# 加载LoRA权重
model = PeftModel.from_pretrained(base_model, "./lora_weights")

# 合并
merged_model = model.merge_and_unload()

# 保存完整模型
merged_model.save_pretrained("./merged_model")
```

### 6.2 使用 vLLM 部署

```python
from vllm import LLM, SamplingParams

# 加载合并后的模型
llm = LLM(model="./merged_model")

# 推理
params = SamplingParams(temperature=0.7, max_tokens=500)
outputs = llm.generate(["你好，请介绍一下自己"], params)
print(outputs[0].outputs[0].text)
```

---

## 🔰 7. LoRA 变体

### 7.1 常见变体

| 变体 | 特点 |
|------|------|
| LoRA | 基础版本 |
| QLoRA | 量化版本 |
| LoRA+ | 不同层不同学习率 |
| rsLoRA | 秩稳定缩放 |
| DoRA | 权重分解 |

### 7.2 选择建议

| 场景 | 推荐 |
|------|------|
| 显存充足 | LoRA |
| 显存有限 | QLoRA |
| 追求最佳效果 | Full Fine-tuning |
| 多任务 | LoRA（保存多个适配器） |

---

## 📚 本节小结

| 概念 | 说明 |
|------|------|
| LoRA | 低秩适配，只训练少量参数 |
| QLoRA | 量化LoRA，进一步减少显存 |
| r参数 | 低秩矩阵的秩，越大越强但越慢 |
| target_modules | 要适配的模型层 |

---

## 🎯 下一步

- **20c - 微调：数据准备与实战** - 数据处理、训练技巧、模型合并
- **21a - RLHF：基础概念** - 人类反馈强化学习

---

> 💡 **实践建议**：用QLoRA在7B模型上实验，感受参数高效微调的威力。
