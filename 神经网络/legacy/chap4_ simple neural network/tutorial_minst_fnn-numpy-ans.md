---
title: "chap4 前馈神经网络numpy参考答案"
tags: [神经网络, nndl]
---

# 前馈神经网络（numpy 实现）参考答案

MNIST 分类。手写 Matmul / Relu / Softmax / Log 的前向与反向，并用 `torch.autograd` 对照梯度。

```python
import os
import numpy as np
import torchvision

def mnist_dataset():
    train_set = torchvision.datasets.MNIST(root='./mnist/', train=True,  download=True)
    test_set  = torchvision.datasets.MNIST(root='./mnist/', train=False, download=True)
    x       = train_set.data.numpy() / 255.0
    y       = train_set.targets.numpy()
    x_test  = test_set.data.numpy() / 255.0
    y_test  = test_set.targets.numpy()
    return (x, y), (x_test, y_test)
```

## 算子的前向与反向

```python
class Matmul:
    def __init__(self):
        self.mem = {}
    def forward(self, x, W):
        self.mem = {'x': x, 'W': W}
        return x @ W
    def backward(self, grad_y):
        x, W = self.mem['x'], self.mem['W']
        grad_x = grad_y @ W.T
        grad_W = x.T @ grad_y
        return grad_x, grad_W

class Relu:
    def __init__(self):
        self.mem = {}
    def forward(self, x):
        self.mem['x'] = x
        return np.where(x > 0, x, 0.0)
    def backward(self, grad_y):
        x = self.mem['x']
        return (x > 0).astype(grad_y.dtype) * grad_y

class Softmax:
    def __init__(self):
        self.epsilon = 1e-12
        self.mem = {}
    def forward(self, x):
        # 数值稳定：减去 max
        x_exp = np.exp(x - x.max(axis=1, keepdims=True))
        out = x_exp / (x_exp.sum(axis=1, keepdims=True) + self.epsilon)
        self.mem['out'] = out
        return out
    def backward(self, grad_y):
        s = self.mem['out']
        sisj = np.matmul(np.expand_dims(s, 2), np.expand_dims(s, 1))  # (N, c, c)
        tmp  = np.matmul(np.expand_dims(grad_y, 1), sisj).squeeze(1)   # (N, c)
        return -tmp + grad_y * s

class Log:
    def __init__(self):
        self.epsilon = 1e-12
        self.mem = {}
    def forward(self, x):
        self.mem['x'] = x
        return np.log(x + self.epsilon)
    def backward(self, grad_y):
        return grad_y / (self.mem['x'] + self.epsilon)
```

## 梯度对照（torch.autograd）

```python
import torch

x  = np.random.normal(size=[5, 6])
W1 = np.random.normal(size=[6, 5])
W2 = np.random.normal(size=[5, 6])
label = np.zeros_like(x)
label[0, 1] = 1; label[1, 0] = 1; label[2, 3] = 1; label[3, 5] = 1; label[4, 0] = 1

mul_h1, mul_h2 = Matmul(), Matmul()
relu, softmax, log = Relu(), Softmax(), Log()

h1       = mul_h1.forward(x, W1)
h1_relu  = relu.forward(h1)
h2       = mul_h2.forward(h1_relu, W2)
h2_soft  = softmax.forward(h2)
h2_log   = log.forward(h2_soft)

h2_log_grad   = log.backward(label)
h2_soft_grad  = softmax.backward(h2_log_grad)
h2_grad, W2g  = mul_h2.backward(h2_soft_grad)
h1_relu_grad  = relu.backward(h2_grad)
h1_grad, W1g  = mul_h1.backward(h1_relu_grad)

print('numpy grad on prob:')
print(h2_log_grad)

x_t  = torch.tensor(x)
W1_t = torch.tensor(W1, requires_grad=True)
W2_t = torch.tensor(W2, requires_grad=True)
label_t = torch.tensor(label)
prob = torch.softmax(torch.relu(x_t @ W1_t) @ W2_t, dim=-1)
loss = (label_t * torch.log(prob)).sum()
(prob_grad,) = torch.autograd.grad(loss, prob)
print('torch grad on prob:')
print(prob_grad.numpy())
```

## 模型与训练

```python
class myModel:
    def __init__(self):
        self.W1 = np.random.normal(size=[28*28 + 1, 100])
        self.W2 = np.random.normal(size=[100, 10])
        self.mul_h1, self.mul_h2 = Matmul(), Matmul()
        self.relu, self.softmax, self.log = Relu(), Softmax(), Log()

    def forward(self, x):
        x = x.reshape(-1, 28*28)
        x = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
        self.h1       = self.mul_h1.forward(x, self.W1)
        self.h1_relu  = self.relu.forward(self.h1)
        self.h2       = self.mul_h2.forward(self.h1_relu, self.W2)
        self.h2_soft  = self.softmax.forward(self.h2)
        self.h2_log   = self.log.forward(self.h2_soft)

    def backward(self, label):
        g = self.log.backward(-label)
        g = self.softmax.backward(g)
        g, self.W2_grad = self.mul_h2.backward(g)
        g = self.relu.backward(g)
        _, self.W1_grad = self.mul_h1.backward(g)
```

```python
def compute_loss(log_prob, labels):
    return np.mean(np.sum(-log_prob * labels, axis=1))

def compute_accuracy(log_prob, labels):
    return np.mean(np.argmax(log_prob, axis=1) == np.argmax(labels, axis=1))

def train_one_step(model, x, y, lr=1e-5):
    model.forward(x)
    model.backward(y)
    model.W1 -= lr * model.W1_grad
    model.W2 -= lr * model.W2_grad
    return compute_loss(model.h2_log, y), compute_accuracy(model.h2_log, y)

(x_tr, y_tr), (x_te, y_te) = mnist_dataset()
n_tr = x_tr.shape[0]
lab_tr = np.zeros((n_tr, 10)); lab_tr[np.arange(n_tr), y_tr] = 1
lab_te = np.zeros((x_te.shape[0], 10)); lab_te[np.arange(x_te.shape[0]), y_te] = 1

model = myModel()
for epoch in range(50):
    loss, acc = train_one_step(model, x_tr, lab_tr)
    print(f'epoch {epoch:2d}  loss {loss:8.4f}  acc {acc:.4f}')

model.forward(x_te)
print('test loss', compute_loss(model.h2_log, lab_te), 'test acc', compute_accuracy(model.h2_log, lab_te))
```
