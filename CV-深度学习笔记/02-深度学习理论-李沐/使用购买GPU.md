---
title: "使用购买GPU"
tags: [CV, Deep-Learning, PyTorch]
---

# 使用购买GPU

# 1. 确认GPU

```python
!nvidia-smi
```

```
Sat Apr 30 18:03:10 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 471.35       Driver Version: 471.35       CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ... WDDM  | 00000000:01:00.0  On |                  N/A |
| N/A   57C    P5    27W /  N/A |   2535MiB / 16384MiB |     10%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1556    C+G   Insufficient Permissions        N/A      |
|    0   N/A  N/A      2040    C+G   ...me\Application\chrome.exe    N/A      |
|    0   N/A  N/A      2876    C+G   C:\Windows\explorer.exe         N/A      |
|    0   N/A  N/A      3196    C+G   ...y\ShellExperienceHost.exe    N/A      |
|    0   N/A  N/A      9980    C+G   ...5n1h2txyewy\SearchApp.exe    N/A      |
|    0   N/A  N/A     12008    C+G   ...artMenuExperienceHost.exe    N/A      |
|    0   N/A  N/A     13280    C+G   ...cw5n1h2txyewy\LockApp.exe    N/A      |
|    0   N/A  N/A     14460    C+G   ...2txyewy\TextInputHost.exe    N/A      |
|    0   N/A  N/A     18708    C+G   Insufficient Permissions        N/A      |
|    0   N/A  N/A     20020      C   ...a\envs\py3.6.3\python.exe    N/A      |
|    0   N/A  N/A     20804    C+G   ...4__htrsf667h5kn2\AWCC.exe    N/A      |
|    0   N/A  N/A     21492    C+G   ...e\StoreExperienceHost.exe    N/A      |
|    0   N/A  N/A     22124    C+G   ...kzcwy\mcafee-security.exe    N/A      |
|    0   N/A  N/A     22540    C+G   ...y\AccountsControlHost.exe    N/A      |
|    0   N/A  N/A     23132    C+G   ...5n1h2txyewy\SearchApp.exe    N/A      |
|    0   N/A  N/A     26464    C+G   ...ekyb3d8bbwe\YourPhone.exe    N/A      |
+-----------------------------------------------------------------------------+

```

```python
import torch
from torch import nn

torch.device('cpu'), torch.cuda.device('cuda'), torch.cuda.device('cuda:1')
```

输出：
```
(device(type='cpu'),
 <torch.cuda.device at 0x2a59c8b2278>,
 <torch.cuda.device at 0x2a59c8b2588>)
```

```python
print(torch.cuda.device_count())
```

```
1

```

```python
def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回gpu"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]。"""
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]      
    return devices if devices else [torch.device('cpu')]
                    
print(try_gpu())
print(try_gpu(10))
print(try_all_gpus())
```

```
cuda:0
cpu
[device(type='cuda', index=0)]

```

```python
# 查询张量所在设备
X = torch.tensor([1,2,3])
print(X.device) # 默认在CPU内存上
```

```
cpu

```

```python
# 存储在GPU上
X = torch.ones(2,3,device=try_gpu())
X 
```

输出：
```
tensor([[1., 1., 1.],
        [1., 1., 1.]], device='cuda:0')
```

```python
# 在第二个GPU上创建一个随即张量
Y = torch.rand(2,3,device=try_gpu(1))
print(Y.device) # 没有1号GPU，则放到CPU上
Y
```

```
cpu

```

输出：
```
tensor([[0.3045, 0.4398, 0.9162],
        [0.5351, 0.1256, 0.8916]])
```

```python
# 要计算X+Y，我们需要决定在哪里执行这个操作
Z = X.cuda(0) # X+Y必须X和Y都在同一个GPU上
print(X)
print(Z)
```

```
tensor([[1., 1., 1.],
        [1., 1., 1.]], device='cuda:0')
tensor([[1., 1., 1.],
        [1., 1., 1.]], device='cuda:0')

```

```python
Y = torch.rand(2,3,device=try_gpu(0))
Y + Z
Z.cuda(0) is Z # 如果变量在0号GPU时，就返回True
```

输出：
```
True
```

```python
# 神经网络与GPU
net = nn.Sequential(nn.Linear(3,1)) # 创建神经网络时已经把权重初始化好了
net = net.to(device=try_gpu()) # 把所有参数在0号GPU上拷贝一份
X = torch.ones(2,3,device=try_gpu()) # X 在0号GPU上
net(X) # 所以前项运算所有元素都在0号GPU上运行
```

输出：
```
tensor([[0.7668],
        [0.7668]], device='cuda:0', grad_fn=<AddmmBackward0>)
```

```python
# 确认模型参数存储在同一个GPU上
net[0].weight.data.device
```

输出：
```
device(type='cuda', index=0)
```