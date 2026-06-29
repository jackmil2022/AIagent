---
title: "chap2 线性回归参考答案"
tags: [神经网络, nndl]
---

# 线性回归（参考答案）

包含三种基函数（identity / 多项式 / 高斯）以及两种求解方式（最小二乘闭式解 / 梯度下降）。

```python
import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    xys = []
    with open(filename, 'r') as f:
        for line in f:
            xys.append(list(map(float, line.strip().split())))
    xs, ys = zip(*xys)
    return np.asarray(xs), np.asarray(ys)
```

## 基函数

```python
def identity_basis(x):
    return np.expand_dims(x, axis=1)

def multinomial_basis(x, feature_num=10):
    x = np.expand_dims(x, axis=1)
    feat = [x ** i for i in range(1, feature_num + 1)]
    return np.concatenate(feat, axis=1)

def gaussian_basis(x, feature_num=10):
    centers = np.linspace(0, 25, feature_num)
    width = centers[1] - centers[0]
    x = np.expand_dims(x, axis=1)
    x = np.concatenate([x] * feature_num, axis=1)
    return np.exp(-0.5 * ((x - centers) / width) ** 2)
```

## 求解 w

两种方法都给出。默认走闭式解；要看梯度下降把 `method='gd'` 传进来。

```python
def main(x_train, y_train, basis_func=gaussian_basis, method='lsq'):
    phi0 = np.expand_dims(np.ones_like(x_train), axis=1)
    phi1 = basis_func(x_train)
    phi  = np.concatenate([phi0, phi1], axis=1)

    if method == 'lsq':
        # 最小二乘闭式解，等价于 (ΦᵀΦ)^-1 Φᵀy
        w = np.linalg.pinv(phi) @ y_train
    elif method == 'gd':
        dim = phi.shape[1]
        w = 0.05 * np.random.randn(dim)
        for _ in range(10000):
            grad = phi.T @ (phi @ w - y_train)
            w -= 0.001 * grad / (np.mean(np.abs(grad)) + 1e-12)
    else:
        raise ValueError(method)

    def f(x):
        phi0 = np.expand_dims(np.ones_like(x), axis=1)
        phi1 = basis_func(x)
        phi  = np.concatenate([phi0, phi1], axis=1)
        return phi @ w
    return f
```

## 评估

```python
def evaluate(ys, ys_pred):
    return np.sqrt(np.mean((ys - ys_pred) ** 2))

x_train, y_train = load_data('train.txt')
x_test,  y_test  = load_data('test.txt')

f = main(x_train, y_train, basis_func=gaussian_basis, method='lsq')
print('train std:', f'{evaluate(y_train, f(x_train)):.3f}')
print('test  std:', f'{evaluate(y_test,  f(x_test )):.3f}')

plt.plot(x_train, y_train, 'ro', markersize=3, label='train')
plt.plot(x_test,  f(x_test), 'k-', label='pred')
plt.xlabel('x'); plt.ylabel('y'); plt.title('Linear Regression')
plt.legend(); plt.show()
```
