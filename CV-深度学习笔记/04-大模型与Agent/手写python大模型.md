---
module: "CV-深度学习笔记"
title: "手写python大模型"
tags: [CV, Deep-Learning, PyTorch]
---

# 手写python大模型

# 手写python大模型.py

```python
# 导入处理日期时间的模块
import datetime
# 导入操作系统接口模块，用于环境变量、路径等操作
import os
# 从 pathlib 库中导入 Path 类，用于进行跨平台的文件路径处理
from pathlib import Path

# 导入 PyTorch 库及其子模块
# PyTorch 主模块
import torch             
# 包含神经网络模块
import torch.nn as nn        
# 提供优化器功能
import torch.optim as optim          
# 导入 NumPy，用于科学计算（如数学函数）
import numpy as np                          

# 定义字符级别的分词器类（Tokenizer）
class CharTokenizer:
    def __init__(self, text):
        # 创建排序后的去重字符列表作为词汇表
        self.vocab = sorted(list(set(text)))       
        # 计算词汇表大小（字符数量）
        self.vocab_size = len(self.vocab)                  
        # 字符到索引的映射字典
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}  
        # 索引到字符的映射字典
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}  

    def encode(self, text):
        # 将输入文本的每个字符转换为索引
        return [self.char_to_idx[char] for char in text]             

    def decode(self, indices):
        # 将索引列表还原为字符串文本
        return ''.join([self.idx_to_char[idx] for idx in indices])   

# 定义位置编码模块，用于为序列中的每个位置添加唯一信息
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        # 调用父类构造函数
        super().__init__()              
        # 创建一个形状为 (max_len, d_model) 的全零张量
        pe = torch.zeros(max_len, d_model)                           
        # 创建位置索引 (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  
        # 计算分母频率项
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))  

        # 对偶数维度使用 sin 编码
        pe[:, 0::2] = torch.sin(position * div_term)               
         # 对奇数维度使用 cos 编码
        pe[:, 1::2] = torch.cos(position * div_term)                
        # 增加 batch 维度，形状变为 (1, max_len, d_model)
        pe = pe.unsqueeze(0)                      
        # 注册为 buffer，模型保存时包含但不更新
        self.register_buffer('pe', pe)                               

    def forward(self, x):
        # 截取对应长度位置编码，加到输入上
        return x + self.pe[:, :x.size(1)]                            

# 简化版 Transformer Decoder 层（本例未使用）
class SimpleDecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward, dropout):
        # 初始化父类
        super().__init__()   
        # 自注意力层
        self.self_atten = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)  
         # 第一个 LayerNorm
        self.norm1 = nn.LayerNorm(d_model)   
        # 前馈神经网络
        self.ffn = nn.Sequential(                                    
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        # 第二个 LayerNorm
        self.norm2 = nn.LayerNorm(d_model)    
        # Dropout 防止过拟合
        self.dropout = nn.Dropout(dropout)                           

    def forward(self, tgt, tgt_mask=None):
        # 执行自注意力
        attn_output, _ = self.self_atten(tgt, tgt, tgt, attn_mask=tgt_mask)  
        # 残差连接后加入 Dropout
        tgt = tgt + self.dropout(attn_output)           
        # 第一次归一化
        tgt = self.norm1(tgt)     
        # 前馈输出
        ffn_output = self.ffn(tgt)            
         # 再次残差连接
        tgt = tgt + self.dropout(tgt)    
        # 第二次归一化
        tgt = self.norm2(tgt)                                       
        return tgt

# 定义主模型：字符级语言模型
class SimleTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, num_layers, dim_feedforward, dropout, max_len):
        # 初始化父类
        super().__init__()               
        # 字符索引嵌入为 d_model 维向量
        self.embedding = nn.Embedding(vocab_size, d_model)      
        # 添加位置编码模块
        self.pos_encoder = PositionalEncoding(d_model, max_len)     
        # 单层解码器
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_head, dim_feedforward, dropout, batch_first=True)  
        # 堆叠多个解码层
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers) 
        # 最终线性层输出词表概率
        self.fc = nn.Linear(d_model, vocab_size)                    

        # 保存模型维度
        self.d_model = d_model       
        # 初始化模型参数
        self.init_weights()                                         

    def init_weights(self):
        # 设置初始化范围
        initrange = 0.1                  
        # 初始化嵌入层权重
        self.embedding.weight.data.uniform_(-initrange, initrange) 
        # 将线性层偏置设为 0
        self.fc.bias.data.zero_()    
        # 初始化线性层权重
        self.fc.weight.data.uniform_(-initrange, initrange)         

    def forward(self, src, tgt_mask=None):
        # 将输入索引嵌入并缩放
        src_emb = self.embedding(src) * np.sqrt(self.d_model)   
        # 加入位置编码
        src_emb = self.pos_encoder(src_emb)             
        # 使用零向量作为 Encoder 输出（Decoder-only）
        memory = torch.zeros_like(src_emb)                   
        # 解码器前向传播
        output = self.transformer_decoder(tgt=src_emb, memory=memory, tgt_mask=tgt_mask)  
        # 映射为词表大小输出
        output = self.fc(output)                                    
        return output

    def generate_square_mask(self, size):
        # 上三角掩码防止看到未来
        mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)  
        return mask

# 文本数据集定义：将文本转换为序列对用于训练
class TextDataset:
    def __init__(self, text, tokenizer, seq_len):
        # 保存分词器对象
        self.tokenizer = tokenizer      
        # 将文本转为字符索引列表
        self.indexed_text = tokenizer.encode(text)     
        # 设置序列长度
        self.seq_len = seq_len                                      

    def __len__(self):
        # 可构造的训练样本数量
        return len(self.indexed_text) - self.seq_len                

    def __getitem__(self, idx):
        # 输入序列
        input_indices = self.indexed_text[idx:idx + self.seq_len]   
        # 目标序列为输入向后移动一位
        target_indices = self.indexed_text[idx + 1:idx + self.seq_len + 1]  
        # 转换为张量
        return torch.tensor(input_indices), torch.tensor(target_indices)    

# 定义训练模型的主循环函数
def train_model(model, dataset, tokenizer, epochs, batch_size, seq_len, learning_rate, device):
    # 将模型移动到设备（CPU/GPU）
    model.to(device)      
    # 设置为训练模式
    model.train()                     
    # 使用 AdamW 优化器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)   
    # 使用交叉熵损失
    criterion = nn.CrossEntropyLoss()                               
    # 数据加载器
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, drop_last=True)  

    # 遍历每个 epoch
    for epoch in range(epochs):                                     
        total_loss = 0
        # 遍历每个 batch
        for batch_idx, (input_seq, target_seq) in enumerate(dataloader):  
            # 移动数据到设备
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)  
            # 生成因果掩码
            src_mask = model.generate_square_mask(seq_len).to(device)           

            # 清除梯度
            optimizer.zero_grad()          
            # 模型前向传播
            output = model(input_seq, tgt_mask=src_mask)       
            # 计算损失
            loss = criterion(output.view(-1, tokenizer.vocab_size), target_seq.view(-1))  

            # 反向传播
            loss.backward()     
            # 优化更新
            optimizer.step()                                        

            # 累积损失
            total_loss += loss.item()      
            # 每 100 批输出一次平均损失
            if batch_idx % 100 == 0:                                
                avg_loss = total_loss / (batch_idx + 1)
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}, Avg Loss:{avg_loss:.4f}')

        # 当前 epoch 平均损失
        avg_epoch_loss = total_loss / len(dataloader)               
        print(f'Epoch{epoch+1} finished, Average Loss:{avg_epoch_loss:.4f}')     

# 定义文本生成函数（给定起始字符生成序列）
def generate_text(model, tokenizer, start_text, max_len, device, temperature=1.0):
    # 移动模型到设备
    model.to(device)          
    # 设置为评估模式
    model.eval()                    
    # 将起始文本编码为索引列表
    input_indices = tokenizer.encode(start_text)                   
    # 添加 batch 维度并转为张量
    input_tensor = torch.tensor(input_indices).unsqueeze(0).to(device)  

    # 初始化生成结果列表
    generated_indices = input_indices.copy()                        

    # 不记录梯度，节省内存
    with torch.no_grad():      
        # 最长生成 max_len 个字符
        for _ in range(max_len):                        
            # 动态生成掩码
            src_mask = model.generate_square_mask(input_tensor.size(1)).to(device)  
            # 前向传播获取 logits
            output = model(input_tensor, tgt_mask=src_mask)         

            # 获取最后一个位置的输出向量
            last_output = output[:, -1, :]                          
            # 应用 softmax 获取概率分布
            output_probs = nn.functional.softmax(last_output / temperature, dim=-1)  
            # 从概率中采样下一个字符索引
            next_token_idx = torch.multinomial(output_probs, num_samples=1).item()  

            # 添加新字符
            generated_indices.append(next_token_idx)           
            # 更新输入张量
            input_tensor = torch.tensor([generated_indices]).to(device)  

    # 解码生成的索引为字符串
    generate_text = tokenizer.decode(generated_indices)             
    return generate_text

# 主程序入口
if __name__ == '__main__':
    # 输出当前时间
    print('加载数据', datetime.datetime.now())                      

    # 定义文本目录
    book_dir = Path(r'test')      
    # 初始化文本内容变量
    text_data = ''             
    # 遍历所有 txt 文件
    for book in book_dir.glob('*.txt'):                         
        # 读取文本并追加到 text_data
        text_data += book.read_text(encoding='utf-8')               

    # 打印文本总长度
    print('内容长度', len(text_data))       
    # 输出时间戳
    print('分词', datetime.datetime.now())       
    # 初始化字符分词器
    tokenizer = CharTokenizer(text_data)                            

    # 设置模型超参数
    vocab_size = tokenizer.vocab_size
    d_model = 128
    n_head = 8
    num_layers = 8
    dim_feedforward = 256
    dropout = 0.1
    seq_len = 32
    max_len = 512

    # 设置训练超参数
    epochs = 10
    batch_size = 32
    learning_rate = 0.001
    # 检查是否使用 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    # 初始化模型
    model = SimleTransformerLM(vocab_size, d_model, n_head, num_layers, dim_feedforward, dropout, max_len)  
    # 创建数据集对象
    dataset = TextDataset(text_data, tokenizer, seq_len)         
    # 输出开始训练时间
    print('训练', datetime.datetime.now())                          
    # 开始训练
    train_model(model, dataset, tokenizer, epochs, batch_size, seq_len, learning_rate, device)  

    # 输出测试开始时间
    print('测试', datetime.datetime.now())          
    # 设置起始文本
    start_text = '孙悟空'                                           
    # 文本生成
    generate_text = generate_text(model, tokenizer, start_text, max_len=100, device=device, temperature=1.0)  
    # 输出提示信息
    print('\nGenerated Text:')                                     
    # 打印生成的文本
    print(generate_text)                                           
```

```
加载数据 2025-06-30 13:18:55.233099
内容长度 2317
分词 2025-06-30 13:18:55.235095
训练 2025-06-30 13:18:55.268005
Epoch: 1/10, Batch: 0, Avg Loss:6.4988
Epoch1 finished, Average Loss:4.7539
Epoch: 2/10, Batch: 0, Avg Loss:3.0506
Epoch2 finished, Average Loss:1.8768
Epoch: 3/10, Batch: 0, Avg Loss:0.8697
Epoch3 finished, Average Loss:0.5106
Epoch: 4/10, Batch: 0, Avg Loss:0.3464
Epoch4 finished, Average Loss:0.2590
Epoch: 5/10, Batch: 0, Avg Loss:0.1770
Epoch5 finished, Average Loss:0.2028
Epoch: 6/10, Batch: 0, Avg Loss:0.1753
Epoch6 finished, Average Loss:0.1774
Epoch: 7/10, Batch: 0, Avg Loss:0.1516
Epoch7 finished, Average Loss:0.1606
Epoch: 8/10, Batch: 0, Avg Loss:0.1537
Epoch8 finished, Average Loss:0.1558
Epoch: 9/10, Batch: 0, Avg Loss:0.1285
Epoch9 finished, Average Loss:0.1462
Epoch: 10/10, Batch: 0, Avg Loss:0.1453
Epoch10 finished, Average Loss:0.1400
测试 2025-06-30 13:22:54.040832

Generated Text:
孙悟空在西方雨师山上挑战蝎子精，封印住西游三宝，只有获得蝎子精的“观音灵签”四个跟着说:“观音灵签”和把和尚去，垂才到此天山妖怪天通才到此宵。”才到贵处“，让唐僧上雷公僧说:“观音灵饮坐去，岂不得妖怪脑的神

```