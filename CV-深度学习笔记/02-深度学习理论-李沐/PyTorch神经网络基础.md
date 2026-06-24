---
title: "PyTorch神经网络基础"
tags: [CV, Deep-Learning, PyTorch]
---

# PyTorch神经网络基础

# 1. 层和块

① nn.Sequential 定义了一种特殊的Module。

```python
# 回顾一下多层感知机
import torch
from torch import nn
from torch.nn import functional as F
net = nn.Sequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
X = torch.rand(2,20)
net(X)
```

输出：
```
tensor([[-0.0214, -0.1789, -0.0700, -0.0238, -0.2697,  0.0381,  0.3078, -0.2082,
         -0.1502,  0.0433],
        [ 0.0200, -0.1466, -0.0633,  0.0031, -0.2042,  0.0993,  0.3137, -0.1206,
         -0.1057,  0.0434]], grad_fn=<AddmmBackward0>)
```

# 2. 自定义块

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()  # 调用父类的__init__函数
        self.hidden = nn.Linear(20,256)
        self.out = nn.Linear(256,10)
        
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))
    
# 实例化多层感知机的层，然后在每次调用正向传播函数调用这些层
net = MLP()
X = torch.rand(2,20)
net(X)
```

输出：
```
tensor([[-0.1600,  0.0363,  0.0851,  0.0364,  0.0189,  0.1590,  0.1519,  0.1299,
         -0.1382, -0.2075],
        [-0.1956,  0.0779, -0.0385, -0.0741,  0.0229,  0.0116,  0.1271,  0.0273,
         -0.0867, -0.0511]], grad_fn=<AddmmBackward0>)
```

# 3. 顺序块

```python
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for block in args:
            self._modules[block] = block # block 本身作为它的key，存在_modules里面的为层，以字典的形式
            
    def forward(self, X):
        for block in self._modules.values():
            print(block)
            X = block(X)
        return X
    
net = MySequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
X = torch.rand(2,20)
net(X)
```

```
Linear(in_features=20, out_features=256, bias=True)
ReLU()
Linear(in_features=256, out_features=10, bias=True)

```

输出：
```
tensor([[-0.0651,  0.0377, -0.0348, -0.0377,  0.1602,  0.0022, -0.0904,  0.1742,
         -0.0520,  0.0189],
        [-0.0192,  0.1056, -0.0497,  0.0301,  0.2464,  0.0126, -0.1700,  0.4147,
          0.0703, -0.0013]], grad_fn=<AddmmBackward0>)
```

# 4. 正向传播

```python
# 在正向传播函数中执行代码
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20,20),requires_grad=False)
        self.linear = nn.Linear(20,20)
    
    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight + 1))
        X = self.linear(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
    
net = FixedHiddenMLP()
X = torch.rand(2,20)
net(X)
```

输出：
```
tensor(0.3770, grad_fn=<SumBackward0>)
```

# 5. 混合组合块

```python
# 混合代培各种组合块的方法
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20,64),nn.ReLU(),
                                nn.Linear(64,32),nn.ReLU())
        self.linear = nn.Linear(32,16)
        
    def forward(self, X):
        return self.linear(self.net(X))
    
chimear = nn.Sequential(NestMLP(),nn.Linear(16,20),FixedHiddenMLP())
X = torch.rand(2,20)
chimear(X)
```

输出：
```
tensor(-0.1488, grad_fn=<SumBackward0>)
```

# 6. 参数管理

```python
# 首先关注具有单隐藏层的多层感知机
import torch
from torch import nn

net = nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,1))
X = torch.rand(size=(2,4))
print(net(X))
print(net[2].state_dict()) # 访问参数，net[2]就是最后一个输出层
print(type(net[2].bias)) # 目标参数
print(net[2].bias)
print(net[2].bias.data)
print(net[2].weight.grad == None) # 还没进行反向计算，所以grad为None
print(*[(name, param.shape) for name, param in net[0].named_parameters()])  # 一次性访问所有参数         
print(*[(name, param.shape) for name, param in net.named_parameters()])  # 0是第一层名字，1是ReLU，它没有参数
print(net.state_dict()['2.bias'].data) # 通过名字获取参数
```

```
tensor([[0.3941],
        [0.4224]], grad_fn=<AddmmBackward0>)
OrderedDict([('weight', tensor([[ 4.7564e-02, -5.3226e-02,  1.4919e-04, -2.8679e-01,  1.7408e-01,
          3.0859e-01, -1.2281e-01,  5.6171e-02]])), ('bias', tensor([0.3129]))])
<class 'torch.nn.parameter.Parameter'>
Parameter containing:
tensor([0.3129], requires_grad=True)
tensor([0.3129])
True
('weight', torch.Size([8, 4])) ('bias', torch.Size([8]))
('0.weight', torch.Size([8, 4])) ('0.bias', torch.Size([8])) ('2.weight', torch.Size([1, 8])) ('2.bias', torch.Size([1]))
tensor([0.3129])

```

# 7. 嵌套块

```python
# 从嵌套块收集参数
def block1():
    return nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,4),nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block{i}',block1()) # f'block{i}' 可以传一个字符串名字过来，block2可以嵌套四个block1                                      
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4,1))
print(rgnet(X))
print(rgnet)
```

```
tensor([[-0.1750],
        [-0.1750]], grad_fn=<AddmmBackward0>)
Sequential(
  (0): Sequential(
    (block0): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block1): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block2): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block3): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
  )
  (1): Linear(in_features=4, out_features=1, bias=True)
)

```

# 8 内置初始化

```python
net = nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,1))

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01) # 下划线表示把m.weight的值替换掉   
        nn.init.zeros_(m.bias)
        
net.apply(init_normal) # 会递归调用 直到所有层都初始化
print(net[0].weight.data[0])
print(net[0].bias.data[0])
```

```
tensor([ 0.0012, -0.0112, -0.0153,  0.0218])
tensor(0.)

```

```python
net = nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,1))

def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight,1)
        nn.init.zeros_(m.bias)
        
net.apply(init_constant)
print(net[0].weight.data[0]) 
print(net[0].bias.data[0])
```

```
tensor([1., 1., 1., 1.])
tensor(0.)

```

```python
# 对某些块应用不同的初始化
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)
        
net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```

```
tensor([ 0.0479, -0.1771,  0.5267, -0.0020])
tensor([[42., 42., 42., 42., 42., 42., 42., 42.]])

```

# 9. 参数替换

```python
# 自定义初始化
def my_init(m):
    if type(m) == nn.Linear:
        print("Init",*[(name, param.shape) for name, param in m.named_parameters()][0])  # 打印名字是啥，形状是啥       
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >=  5 # 这里*=的代码相当于先计算一个布尔矩阵(先判断>=)，然后再用布尔矩阵的对应元素去乘以原始矩阵的每个元素。保留绝对值大于5的权重，不是的话就设为0

net.apply(my_init)
print(net[0].weight[:2])
net[0].weight.data[:] += 1 # 参数替换
net[0].weight.data[0,0] = 42
print(net[0].weight.data[0])
```

```
Init weight torch.Size([8, 4])
Init weight torch.Size([1, 8])
tensor([[ 0.0000,  7.1240,  0.0000,  5.1135],
        [-8.6745, -7.3974,  0.0000, -0.0000]], grad_fn=<SliceBackward0>)
tensor([42.0000,  8.1240,  1.0000,  6.1135])

```

# 10. 参数绑定

```python
# 参数绑定
shared = nn.Linear(8,8)
net = nn.Sequential(nn.Linear(4,8),nn.ReLU(),shared,nn.ReLU(),shared,nn.ReLU(),nn.Linear(8,1))  # 第2个隐藏层和第3个隐藏层是share权重的，第一个和第四个是自己的  
net(X)
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0,0] = 100
print(net[2].weight.data[0] == net[4].weight.data[0])
```

```
tensor([True, True, True, True, True, True, True, True])
tensor([True, True, True, True, True, True, True, True])

```

# 11. 自定义层

```python
# 构造一个没有任何参数的自定义层
import torch
import torch.nn.functional as F
from torch import nn

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, X):
        return X - X.mean()
    
layer = CenteredLayer()
print(layer(torch.FloatTensor([1,2,3,4,5])))

# 将层作为组件合并到构建更复杂的模型中
net = nn.Sequential(nn.Linear(8,128),CenteredLayer())
Y = net(torch.rand(4,8))
print(Y.mean())

# 带参数的图层
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units,units)) # nn.Parameter使得这些参数加上了梯度    
        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
    
dense = MyLinear(5,3)
print(dense.weight)

# 使用自定义层直接执行正向传播计算
print(dense(torch.rand(2,5)))
# 使用自定义层构建模型
net = nn.Sequential(MyLinear(64,8),MyLinear(8,1))
print(net(torch.rand(2,64)))
```

```
tensor([-2., -1.,  0.,  1.,  2.])
tensor(-6.2864e-09, grad_fn=<MeanBackward0>)
Parameter containing:
tensor([[-2.8449,  0.1887,  0.7945],
        [ 0.4226,  1.6180, -0.5880],
        [-0.4794, -0.0817, -0.3648],
        [-0.1979,  0.8702, -0.3515],
        [-1.4943,  0.3618,  0.2969]], requires_grad=True)
tensor([[0.0000, 0.0000, 1.3957],
        [0.8225, 0.0000, 0.9089]])
tensor([[0.],
        [0.]])

```

# 12. 读写文件

```python
# 加载和保存张量
import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')
x2 = torch.load("x-file")
print(x2)

#存储一个张量列表，然后把它们读回内存
y = torch.zeros(4)
torch.save([x,y],'x-files')
x2, y2 = torch.load('x-files')
print(x2)
print(y2)

# 写入或读取从字符串映射到张量的字典
mydict = {'x':x,'y':y}
torch.save(mydict,'mydict')
mydict2 = torch.load('mydict')
print(mydict2)
```

```
tensor([0, 1, 2, 3])
tensor([0, 1, 2, 3])
tensor([0., 0., 0., 0.])
{'x': tensor([0, 1, 2, 3]), 'y': tensor([0., 0., 0., 0.])}

```

```python
# 加载和保存模型参数
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20,256)
        self.output = nn.Linear(256,10)
    
    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))
    
net = MLP()
X = torch.randn(size=(2,20))
Y = net(X)

# 将模型的参数存储为一个叫做"mlp.params"的文件
torch.save(net.state_dict(),'mlp.params')

# 实例化了原始多层感知机模型的一个备份。直接读取文件中存储的参数
clone = MLP() # 必须要先声明一下，才能导入参数
clone.load_state_dict(torch.load("mlp.params"))
print(clone.eval()) # eval()是进入测试模式

Y_clone = clone(X)
print(Y_clone == Y)
```

```
MLP(
  (hidden): Linear(in_features=20, out_features=256, bias=True)
  (output): Linear(in_features=256, out_features=10, bias=True)
)
tensor([[True, True, True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True, True, True]])

```