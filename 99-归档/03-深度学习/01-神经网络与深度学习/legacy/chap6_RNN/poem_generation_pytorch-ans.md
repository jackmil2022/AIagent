---
title: "chap6 唐诗生成参考答案"
tags: [神经网络, nndl]
---

# 唐诗生成（PyTorch）参考答案

```python
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

START_TOKEN = 'G'
END_TOKEN   = 'E'
PAD_TOKEN   = ' '
```

## 数据预处理

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
    vocab = [PAD_TOKEN] + [w for w, _ in counter.most_common() if w != PAD_TOKEN]
    word2id = {w: i for i, w in enumerate(vocab)}
    poems_ids = [[word2id[c] for c in p] for p in poems]
    return poems_ids, word2id, vocab

poems_ids, word2id, vocab = process_poems('./poems.txt')
PAD_ID = word2id[PAD_TOKEN]
print(f'诗句数 {len(poems_ids)}, 词表大小 {len(vocab)}')
```

## Dataset 与 DataLoader

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

关键：
- `nn.LSTM` 的 `batch_first=True`，让输入/输出形状是 `(batch, seq, feature)`。
- 把 `hidden=None` 直接传给 LSTM，PyTorch 会替你创建全零初始状态。

```python
class PoemRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.rnn_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        out, hidden = self.rnn_lstm(emb, hidden)
        return self.fc(out), hidden
```

## 训练

```python
def train(model, loader, num_epochs=10, lr=1e-3, clip=1.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        model.train()
        total = 0.0
        for step, batch in enumerate(loader):
            batch = batch.to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:]
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
            total += loss.item()
            if step % 100 == 0:
                print(f'epoch {epoch} step {step:4d} loss {loss.item():.4f}')
        print(f'== epoch {epoch} avg loss {total/len(loader):.4f}')

model = PoemRNN(len(vocab)).to(device)
train(model, loader, num_epochs=10)
```

## 生成

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
    print(c, '->', gen_poem(model, c))
```
