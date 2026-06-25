---
module: "CV-深度学习笔记"
title: "循环神经网络RNN"
tags: [CV, Deep-Learning, PyTorch, RNN, Sequence]
---

# 循环神经网络RNN

> 来源：李沐《动手学深度学习》| [视频讲解](https://space.bilibili.com/1567748478/lists/358497)

---

## 1. 为什么需要 RNN？

### 1.1 序列数据的特点

传统神经网络假设输入是**独立同分布**的，但很多现实数据是**序列**的：

- **文本**：词与词之间有顺序关系
- **语音**：时间序列数据
- **视频**：帧与帧之间有时间关系
- **时间序列预测**：股票、天气等

### 1.2 传统网络的问题

```
全连接网络：
- 固定大小输入
- 不考虑顺序
- 无法处理变长序列
```

---

## 2. RNN 基本结构

### 2.1 核心思想

**维护一个隐藏状态（hidden state），捕捉序列信息**

```
h_t = f(W_hh * h_{t-1} + W_xh * x_t + b)

其中：
- h_t: 当前时刻的隐藏状态
- h_{t-1}: 上一时刻的隐藏状态
- x_t: 当前时刻的输入
- W_hh, W_xh: 权重矩阵
- f: 激活函数（通常是 tanh）
```

### 2.2 展开图

```
时间步:    t-1      t       t+1
           ↓       ↓       ↓
输入:     x_{t-1}  x_t    x_{t+1}
           ↓       ↓       ↓
隐藏:     h_{t-1}  h_t    h_{t+1}
           ↓       ↓       ↓
输出:     y_{t-1}  y_t    y_{t+1}
```

### 2.3 数学公式

```python
# 前向传播
h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)
y_t = W_hy * h_t + b_hy
```

---

## 3. PyTorch 实现

### 3.1 基本 RNN

```python
import torch
import torch.nn as nn

# 定义 RNN
rnn = nn.RNN(
    input_size=10,    # 输入特征维度
    hidden_size=20,   # 隐藏状态维度
    num_layers=1,     # RNN 层数
    batch_first=True  # 输入格式 (batch, seq, feature)
)

# 输入数据
x = torch.randn(32, 5, 10)  # batch=32, seq_len=5, features=10

# 前向传播
output, h_n = rnn(x)

print(f"输出形状: {output.shape}")  # (32, 5, 20)
print(f"最终隐藏状态: {h_n.shape}")  # (1, 32, 20)
```

### 3.2 LSTM（长短期记忆）

```python
lstm = nn.LSTM(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True,
    dropout=0.2
)

x = torch.randn(32, 5, 10)
output, (h_n, c_n) = lstm(x)

print(f"输出: {output.shape}")
print(f"隐藏状态: {h_n.shape}")
print(f"细胞状态: {c_n.shape}")
```

### 3.3 GRU（门控循环单元）

```python
gru = nn.GRU(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True
)

x = torch.randn(32, 5, 10)
output, h_n = gru(x)

print(f"输出: {output.shape}")
print(f"隐藏状态: {h_n.shape}")
```

---

## 4. RNN 的变体

### 4.1 LSTM（长短期记忆）

**解决长期依赖问题**

```
门控机制：
1. 遗忘门 (Forget Gate): 决定丢弃什么信息
2. 输入门 (Input Gate): 决定存储什么信息
3. 输出门 (Output Gate): 决定输出什么信息
```

```python
# LSTM 内部结构
f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)  # 遗忘门
i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)  # 输入门
C̃_t = tanh(W_C * [h_{t-1}, x_t] + b_C)    # 候选值
C_t = f_t * C_{t-1} + i_t * C̃_t            # 更新细胞状态
o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)  # 输出门
h_t = o_t * tanh(C_t)                        # 更新隐藏状态
```

### 4.2 GRU（门控循环单元）

**LSTM 的简化版本**

```
只有两个门：
1. 重置门 (Reset Gate): 控制如何组合新输入和之前的记忆
2. 更新门 (Update Gate): 控制保留多少之前的记忆
```

### 4.3 双向 RNN

```python
# 双向 RNN
bi_rnn = nn.RNN(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True,
    bidirectional=True  # 双向
)

x = torch.randn(32, 5, 10)
output, h_n = bi_rnn(x)

print(f"输出: {output.shape}")  # (32, 5, 40) - 2*hidden_size
print(f"隐藏状态: {h_n.shape}")  # (4, 32, 20) - 2*num_layers
```

---

## 5. 应用场景

| 任务 | 输入 | 输出 | 示例 |
|------|------|------|------|
| 多对一 | 序列 | 单值 | 情感分类 |
| 一对多 | 单值 | 序列 | 图像描述 |
| 多对多 | 序列 | 序列 | 机器翻译 |
| 同步多对多 | 序列 | 序列 | 词性标注 |

---

## 6. 梯度问题

### 6.1 梯度消失

RNN 难以学习**长距离依赖**：

```
"我 去 北京 天安门 看 升旗 仪式"

处理到最后时，可能已经忘了"我"
```

### 6.2 解决方案

- **LSTM/GRU**：门控机制
- **残差连接**：缓解梯度消失
- **梯度裁剪**：防止梯度爆炸

```python
# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 7. 实战示例

### 7.1 文本分类

```python
class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=2, 
                          batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        embeds = self.dropout(self.embedding(x))
        output, (h_n, c_n) = self.rnn(embeds)
        
        # 拼接双向的最后隐藏状态
        hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        
        output = self.fc(self.dropout(hidden))
        return output

# 使用
model = TextRNN(vocab_size=10000, embed_dim=128, hidden_dim=256, output_dim=2)
```

---

## 总结

| 模型 | 特点 | 适用场景 |
|------|------|----------|
| RNN | 基础版本 | 简单序列任务 |
| LSTM | 门控机制，长期记忆 | 长序列任务 |
| GRU | 简化版 LSTM | 轻量级任务 |
| Bi-RNN | 双向信息 | 需要上下文的任务 |

---

## 参考资料

- [Dive into Deep Learning - RNN](https://d2l.ai/chapter_recurrent-neural-networks/)
- [Colah's Understanding LSTM](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [PyTorch RNN Tutorial](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
