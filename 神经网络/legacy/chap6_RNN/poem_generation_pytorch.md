---
title: "chap6 唐诗生成"
tags: [神经网络, nndl]
---

# 唐诗生成（PyTorch）

用字符级 LSTM 在唐诗语料上训练自回归语言模型，并以指定首字生成诗句。

需要补全模型中：
1. `__init__` 里 `self.rnn_lstm` 的定义；
2. `forward` 里把 embedding 输入 LSTM 的一行。

```python
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

START_TOKEN = 'G'   # 单字符起始符
END_TOKEN   = 'E'   # 单字符结束符
PAD_TOKEN   = ' '   # 用空格作为 padding
```

## 数据预处理

按字符切分诗句，过滤含特殊符号、过长或过短的诗，建立 `字 → id` 词表。

```python
def process_poems(file_name):
    poems = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                _title, content = line.strip().split(':')
            except ValueError:
                continue
            content = content.replace(' ', '')
            bad_chars = '_(（《[' + START_TOKEN + END_TOKEN
            if any(c in content for c in bad_chars):
                continue
            if not 5 <= len(content) <= 80:
                continue
            poems.append(START_TOKEN + content + END_TOKEN)

    counter = collections.Counter()
    for p in poems:
        counter.update(p)
    # 把 PAD 放到 id=0
    vocab = [PAD_TOKEN] + [w for w, _ in counter.most_common() if w != PAD_TOKEN]
    word2id = {w: i for i, w in enumerate(vocab)}
    poems_ids = [[word2id[c] for c in p] for p in poems]
    return poems_ids, word2id, vocab

poems_ids, word2id, vocab = process_poems('./poems.txt')
PAD_ID = word2id[PAD_TOKEN]
print(f'诗句数: {len(poems_ids)}, 词表大小: {len(vocab)}')
```

## Dataset 与 DataLoader

变长序列用 `pad_sequence` 补到批内最大长度，padding 位置在 loss 中忽略。

```python
class PoemDataset(Dataset):
    def __init__(self, poems_ids):
        self.data = [torch.tensor(p, dtype=torch.long) for p in poems_ids]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]

def collate(batch):
    return pad_sequence(batch, batch_first=True, padding_value=PAD_ID)

dataset = PoemDataset(poems_ids)
loader  = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate)
```

## 模型

Embedding → LSTM → Linear。Linear 输出维度等于词表大小，得到的是每个位置上下一个字符的 logits。

**填空 1**：定义 `self.rnn_lstm`，要求是两层 LSTM，输入维度 `embedding_dim`，隐状态维度 `hidden_dim`，并且输入输出形状为 `(batch, seq, feature)`。

**填空 2**：把 `emb` 输入 `self.rnn_lstm`，得到序列输出 `out` 与新的隐状态 `hidden`。

```python
class PoemRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)

        # ??? 填空 1：定义 self.rnn_lstm
        # 提示：nn.LSTM，input_size=embedding_dim，hidden_size=hidden_dim，
        #       num_layers=num_layers，batch_first=True
        # self.rnn_lstm = ...

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)                 # (batch, seq, embedding_dim)

        # ??? 填空 2：把 emb 输入 self.rnn_lstm，得到 out 和 hidden
        # out, hidden = ...

        logits = self.fc(out)                   # (batch, seq, vocab_size)
        return logits, hidden
```

## 训练

Padding 位置的 token 用 `ignore_index=PAD_ID` 不计入 loss。

```python
def train(model, loader, num_epochs=10, lr=1e-3, clip=1.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(loader):
            batch = batch.to(device)
            inputs  = batch[:, :-1]
            targets = batch[:, 1:]
            logits, _ = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=PAD_ID,
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            total_loss += loss.item()
            if step % 100 == 0:
                print(f'epoch {epoch} step {step:4d} loss {loss.item():.4f}')
        print(f'== epoch {epoch} avg loss {total_loss/len(loader):.4f}')

model = PoemRNN(len(vocab)).to(device)
train(model, loader, num_epochs=10)
```

## 生成

以 `<起始符, 首字>` 为种子，自回归地一步步预测下一个字符直到遇到结束符。

```python
@torch.no_grad()
def gen_poem(model, start_char, max_len=80):
    model.eval()
    if start_char not in word2id:
        raise ValueError(f'首字 {start_char!r} 不在词表中')
    ids = [word2id[START_TOKEN], word2id[start_char]]
    x = torch.tensor([ids], device=device)
    logits, hidden = model(x)

    poem = start_char
    for _ in range(max_len):
        next_id = int(logits[0, -1].argmax())
        ch = vocab[next_id]
        if ch == END_TOKEN:
            break
        poem += ch
        x = torch.tensor([[next_id]], device=device)
        logits, hidden = model(x, hidden)
    return poem

for c in '日红山夜湖海月':
    print(c, '→', gen_poem(model, c))
```
