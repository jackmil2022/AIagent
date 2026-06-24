---
title: "多GPU训练实现"
tags: [CV, Deep-Learning, PyTorch]
---

# 多GPU训练实现

# 1. 多GPU训练

```python
%matplotlib inline
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```python
W = torch.randn(size=(3,4)) # 从标准正态分布（均值为0，方差为1）中抽取的一组随机数。
scale = 0.01
print(W * scale)
```

```
tensor([[-0.0241, -0.0077,  0.0035,  0.0003],
        [-0.0109,  0.0084,  0.0114,  0.0123],
        [ 0.0036, -0.0087,  0.0080,  0.0018]])

```

```python
# 简单网络
scale = 0.01
W1 = torch.randn(size=(20,1,3,3)) * scale
b1 = torch.zeros(20)
W2 = torch.randn(size=(50,20,5,5)) * scale
b2 = torch.zeros(50)
W3 = torch.randn(size=(800,128)) * scale
b3 = torch.zeros(128)
W4 = torch.randn(size=(128,10)) * scale
b4 = torch.zeros(10)
params = [W1,b1,W2,b2,W3,b3,W4,b4]

def lenet(X, params):
    h1_conv = F.conv2d(input=X, weight=params[0],bias=params[1])
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2,2), stride=(2,2))
    h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation,kernel_size=(2,2),stride=(2,2))
    h2 = h2.reshape(h2.shape[0],-1)
    h3_linear = torch.mm(h2, params[4]) + params[5]
    h3 = F.relu(h3_linear)
    y_hat = torch.mm(h3, params[6]) + params[7]
    return y_hat

loss = nn.CrossEntropyLoss(reduction='none')
```

```python
# 向多个设备分发参数
def get_params(params, device):
    new_params = [p.clone().to(device) for p in params] # 把params中所有参数挪到GPU上  
    for p in new_params:
        p.requires_grad_() # 每一个参数都要计算梯度
    return new_params

new_params = get_params(params, d2l.try_gpu(0))
print('b1 weight：', new_params[1])
print('b1 grad：', new_params[1].grad)
```

```
b1 weight： tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       device='cuda:0', requires_grad=True)
b1 grad： None

```

```python
# allreduce函数将所有向量相加，并将结果广播给所有GPU
def allreduce(data): # 如果有四个GPU的话，data这个list里面就有四个元素
    for i in range(1, len(data)):
        data[0][:] += data[i].to(data[0].device) # 把其余三个GPU上元素，拷贝到0号GPU上
    for i in range(1, len(data)):
        data[i] = data[0].to(data[i].device) # 把相加后的结果复制回所有GPU上
        
data = [torch.ones((1,2),device=d2l.try_gpu(i)) * (i + 1) for i in range(2)]     
print('before allreduce:\n', data[0], '\n', data[1])
allreduce(data) # allreduce函数可以用于各个GPU的梯度加起来，然后各个GPU拿到合梯度
print('before allreduce:\n', data[0], '\n', data[1])
```

```
before allreduce:
 tensor([[1., 1.]], device='cuda:0') 
 tensor([[2., 2.]])
before allreduce:
 tensor([[3., 3.]], device='cuda:0') 
 tensor([[3., 3.]])

```

```python
# 将一个小批量数据均匀地分布在多个GPU上
data = torch.arange(20).reshape(4,5)
devices = [torch.device('cuda:0'),torch.device('cuda:1')]
print('input:',data)
print('load into',devices)
split = nn.parallel.scatter(data,devices) # 将data均匀的切开，放到不同的GPU上   
print('output:',split)
```

```
input: tensor([[ 0,  1,  2,  3,  4],
        [ 5,  6,  7,  8,  9],
        [10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19]])
load into [device(type='cuda', index=0), device(type='cuda', index=1)]

```

```python
def split_batch(X, y, devices):
    """将X和y拆分到多个设备上"""
    assert X.shape[0] == y.shape[0] # 确定样本数与标号数是一样的
    return (nn.parallel.scatter(X, devices), nn.parallel.scatter(y, devices)) # 样本与标号都均匀分布到不同GPU上    
```

```python
# 在一个小批量上实现多GPU训练
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices) # 把X，y分到不同GPU上
    # 拿到每一个GPU上的X_shard、y_shard以及对应的device_W
    # sum是对每一个GPU上所有样本的损失求和
    # ls返回的是每一个GPU上对应的损失
    ls = [loss(lenet(X_shard, device_W),y_shard).sum() 
          for X_shard, y_shard, device_W in zip(X_shards, y_shards, device_params)]                         
    for l in ls:
        l.backward() # 对每一个GPU的loss做负反馈，算梯度
    with torch.no_grad():
        for i in range(len(device_params[0])):
            # i是层，c是GPU
            allreduce([device_params[c][i].grad for c in range(len(devices))])
        for param in device_params: # 拿到所有梯度后，对每一个GPU做SGD
            d2l.sgd(param, lr, X.shape[0])
```

```python
# 训练
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)] # 创建多少个GPU
    device_params = [get_params(params, d) for d in devices] # 初始化的params复制到每个GPU上
    num_epochs = 10
    animator = d2l.Animator('epoch','test acc',xlim=[1,num_epochs])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            train_batch(X, y, device_params, devices, lr)
            torch.cuda.synchronize() # 等待所有的GPU算完
        timer.stop()
        animator.add(epoch+1,(d2l.evaluate_accuracy_gpu(
            lambda x: lenet(x,device_params[0]),test_iter,devices[0]),))  
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch'
         f'on {str(devices)}')   
```

```python
help(torch.cuda.synchronize)
```

```
Help on function synchronize in module torch.cuda:

synchronize(device:Union[torch.device, str, int, NoneType]=None) -> None
    Waits for all kernels in all streams on a CUDA device to complete.
    
    Args:
        device (torch.device or int, optional): device for which to synchronize.
            It uses the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).
```

```python
# 在单个GPU上运行
train(num_gpus=1, batch_size=256, lr=0.2)
```

```
test acc: 0.84, 3.7 sec/epochon [device(type='cuda', index=0)]

```

```python
# 增加为2个GPU
train(num_gpus=2, batch_size=256, lr=0.2) # 本电脑只有一个GPU，呜呜呜呜
```

# 2. 多GPU的简洁实现

```python
import torch
from torch import nn
from d2l import torch as d2l
```

```python
def resnet18(num_classes, in_channels=1):
    """稍加修改的ResNet-18模型"""
    def resnet_block(in_channels, out_channels, num_residuals, first_block=False):   
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(d2l.Residual(in_channels, out_channels, use_1x1conv=True,strides=2))    
            else:
                blk.append(d2l.Residual(out_channels,out_channels))
        return nn.Sequential(*blk)
        
    net = nn.Sequential(
        nn.Conv2d(in_channels,64,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())  
        
    net.add_module("resnet_block1", resnet_block(64,64,2,first_block=True))
    net.add_module("resnet_block2", resnet_block(64,128,2))
    net.add_module("resnet_block3", resnet_block(128,256,2))
    net.add_module("resnet_block4", resnet_block(256,512,2))
    net.add_module("resnet_avg_pool", nn.AdaptiveAvgPool2d((1,1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes)))  

    return net
    
net = resnet18(10)
devices = d2l.try_all_gpus()
```

```python
# 训练
def train(net, num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    
    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(m.weight, std=0.01)
            
    net.apply(init_weights)
    # nn.DataParallel会的是X切开并行到各个GPU上，并行算梯度，然后loss加起来，它重新定义了net的forward函数
    net = nn.DataParallel(net, device_ids=devices) # net会复制到每一个GPU上
    trainer = torch.optim.SGD(net.parameters(),lr)
    loss = nn.CrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch','test acc', xlim=[1, num_epochs])  
    for epoch in range(num_epochs):
        net.train()
        timer.start()
        for X, y in train_iter:
            trainer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        timer.stop()
        animator.add(epoch+1, (d2l.evaluate_accuracy_gpu(net, test_iter),))    
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch'
         f'on {str(devices)}')
```

```python
help(nn.DataParallel)
```

```
Help on class DataParallel in module torch.nn.parallel.data_parallel:

class DataParallel(torch.nn.modules.module.Module)
 |  Implements data parallelism at the module level.
 |  
 |  This container parallelizes the application of the given :attr:`module` by
 |  splitting the input across the specified devices by chunking in the batch
 |  dimension (other objects will be copied once per device). In the forward
 |  pass, the module is replicated on each device, and each replica handles a
 |  portion of the input. During the backwards pass, gradients from each replica
 |  are summed into the original module.
 |  
 |  The batch size should be larger than the number of GPUs used.
 |  
 |  .. warning::
 |      It is recommended to use :class:`~torch.nn.parallel.DistributedDataParallel`,
 |      instead of this class, to do multi-GPU training, even if there is only a single
 |      node. See: :ref:`cuda-nn-ddp-instead` and :ref:`ddp`.
 |  
 |  Arbitrary positional and keyword inputs are allowed to be passed into
 |  DataParallel but some types are specially handled. tensors will be
 |  **scattered** on dim specified (default 0). tuple, list and dict types will
 |  be shallow copied. The other types will be shared among different threads
 |  and can be corrupted if written to in the model's forward pass.
 |  
 |  The parallelized :attr:`module` must have its parameters and buffers on
 |  ``device_ids[0]`` before running this :class:`~torch.nn.DataParallel`
 |  module.
 |  
 |  .. warning::
 |      In each forward, :attr:`module` is **replicated** on each device, so any
 |      updates to the running module in ``forward`` will be lost. For example,
 |      if :attr:`module` has a counter attribute that is incremented in each
 |      ``forward``, it will always stay at the initial value because the update
 |      is done on the replicas which are destroyed after ``forward``. However,
 |      :class:`~torch.nn.DataParallel` guarantees that the replica on
 |      ``device[0]`` will have its parameters and buffers sharing storage with
 |      the base parallelized :attr:`module`. So **in-place** updates to the
 |      parameters or buffers on ``device[0]`` will be recorded. E.g.,
 |      :class:`~torch.nn.BatchNorm2d` and :func:`~torch.nn.utils.spectral_norm`
 |      rely on this behavior to update the buffers.
 |  
 |  .. warning::
 |      Forward and backward hooks defined on :attr:`module` and its submodules
 |      will be invoked ``len(device_ids)`` times, each with inputs located on
 |      a particular device. Particularly, the hooks are only guaranteed to be
 |      executed in correct order with respect to operations on corresponding
 |      devices. For example, it is not guaranteed that hooks set via
 |      :meth:`~torch.nn.Module.register_forward_pre_hook` be executed before
 |      `all` ``len(device_ids)`` :meth:`~torch.nn.Module.forward` calls, but
 |      that each such hook be executed before the corresponding
 |      :meth:`~torch.nn.Module.forward` call of that device.
 |  
 |  .. warning::
 |      When :attr:`module` returns a scalar (i.e., 0-dimensional tensor) in
 |      :func:`forward`, this wrapper will return a vector of length equal to
 |      number of devices used in data parallelism, containing the result from
 |      each device.
 |  
 |  .. note::
 |      There is a subtlety in using the
 |      ``pack sequence -> recurrent network -> unpack sequence`` pattern in a
 |      :class:`~torch.nn.Module` wrapped in :class:`~torch.nn.DataParallel`.
 |      See :ref:`pack-rnn-unpack-with-data-parallelism` section in FAQ for
 |      details.
 |  
 |  
 |  Args:
 |      module (Module): module to be parallelized
 |      device_ids (list of int or torch.device): CUDA devices (default: all devices)
 |      output_device (int or torch.device): device location of output (default: device_ids[0])
 |  
 |  Attributes:
 |      module (Module): the module to be parallelized
 |  
 |  Example::
 |  
 |      >>> net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
 |      >>> output = net(input_var)  # input_var can be on any device, including CPU
 |  
 |  Method resolution order:
 |      DataParallel
 |      torch.nn.modules.module.Module
 |      builtins.object
 |  
 |  Methods defined here:
 |  
 |  __init__(self, module, device_ids=None, output_device=None, dim=0)
 |      Initializes internal Module state, shared by both nn.Module and ScriptModule.
 |  
 |  forward(self, *inputs, **kwargs)
 |      Defines the computation performed at every call.
 |      
 |      Should be overridden by all subclasses.
 |      
 |      .. note::
 |          Although the recipe for forward pass needs to be defined within
 |          this function, one should call the :class:`Module` instance afterwards
 |          instead of this since the former takes care of running the
 |          registered hooks while the latter silently ignores them.
 |  
 |  gather(self, outputs, output_device)
 |  
 |  parallel_apply(self, replicas, inputs, kwargs)
 |  
 |  replicate(self, module, device_ids)
 |  
 |  scatter(self, inputs, kwargs, device_ids)
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from torch.nn.modules.module.Module:
 |  
 |  __call__ = _call_impl(self, *input, **kwargs)
 |  
 |  __delattr__(self, name)
 |      Implement delattr(self, name).
 |  
 |  __dir__(self)
 |      __dir__() -> list
 |      default dir() implementation
 |  
 |  __getattr__(self, name:str) -> Union[torch.Tensor, _ForwardRef('Module')]
 |  
 |  __repr__(self)
 |      Return repr(self).
 |  
 |  __setattr__(self, name:str, value:Union[torch.Tensor, _ForwardRef('Module')]) -> None
 |      Implement setattr(self, name, value).
 |  
 |  __setstate__(self, state)
 |  
 |  add_module(self, name:str, module:Union[_ForwardRef('Module'), NoneType]) -> None
 |      Adds a child module to the current module.
 |      
 |      The module can be accessed as an attribute using the given name.
 |      
 |      Args:
 |          name (string): name of the child module. The child module can be
 |              accessed from this module using the given name
 |          module (Module): child module to be added to the module.
 |  
 |  apply(self:~T, fn:Callable[[_ForwardRef('Module')], NoneType]) -> ~T
 |      Applies ``fn`` recursively to every submodule (as returned by ``.children()``)
 |      as well as self. Typical use includes initializing the parameters of a model
 |      (see also :ref:`nn-init-doc`).
 |      
 |      Args:
 |          fn (:class:`Module` -> None): function to be applied to each submodule
 |      
 |      Returns:
 |          Module: self
 |      
 |      Example::
 |      
 |          >>> @torch.no_grad()
 |          >>> def init_weights(m):
 |          >>>     print(m)
 |          >>>     if type(m) == nn.Linear:
 |          >>>         m.weight.fill_(1.0)
 |          >>>         print(m.weight)
 |          >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
 |          >>> net.apply(init_weights)
 |          Linear(in_features=2, out_features=2, bias=True)
 |          Parameter containing:
 |          tensor([[ 1.,  1.],
 |                  [ 1.,  1.]])
 |          Linear(in_features=2, out_features=2, bias=True)
 |          Parameter containing:
 |          tensor([[ 1.,  1.],
 |                  [ 1.,  1.]])
 |          Sequential(
 |            (0): Linear(in_features=2, out_features=2, bias=True)
 |            (1): Linear(in_features=2, out_features=2, bias=True)
 |          )
 |          Sequential(
 |            (0): Linear(in_features=2, out_features=2, bias=True)
 |            (1): Linear(in_features=2, out_features=2, bias=True)
 |          )
 |  
 |  bfloat16(self:~T) -> ~T
 |      Casts all floating point parameters and buffers to ``bfloat16`` datatype.
 |      
 |      .. note::
 |          This method modifies the module in-place.
 |      
 |      Returns:
 |          Module: self
 |  
 |  buffers(self, recurse:bool=True) -> Iterator[torch.Tensor]
 |      Returns an iterator over module buffers.
 |      
 |      Args:
 |          recurse (bool): if True, then yields buffers of this module
 |              and all submodules. Otherwise, yields only buffers that
 |              are direct members of this module.
 |      
 |      Yields:
 |          torch.Tensor: module buffer
 |      
 |      Example::
 |      
 |          >>> for buf in model.buffers():
 |          >>>     print(type(buf), buf.size())
 |          <class 'torch.Tensor'> (20L,)
 |          <class 'torch.Tensor'> (20L, 1L, 5L, 5L)
 |  
 |  children(self) -> Iterator[_ForwardRef('Module')]
 |      Returns an iterator over immediate children modules.
 |      
 |      Yields:
 |          Module: a child module
 |  
 |  cpu(self:~T) -> ~T
 |      Moves all model parameters and buffers to the CPU.
 |      
 |      .. note::
 |          This method modifies the module in-place.
 |      
 |      Returns:
 |          Module: self
 |  
 |  cuda(self:~T, device:Union[int, torch.device, NoneType]=None) -> ~T
 |      Moves all model parameters and buffers to the GPU.
 |      
 |      This also makes associated parameters and buffers different objects. So
 |      it should be called before constructing optimizer if the module will
 |      live on GPU while being optimized.
 |      
 |      .. note::
 |          This method modifies the module in-place.
 |      
 |      Args:
 |          device (int, optional): if specified, all parameters will be
 |              copied to that device
 |      
 |      Returns:
 |          Module: self
 |  
 |  double(self:~T) -> ~T
 |      Casts all floating point parameters and buffers to ``double`` datatype.
 |      
 |      .. note::
 |          This method modifies the module in-place.
 |      
 |      Returns:
 |          Module: self
 |  
 |  eval(self:~T) -> ~T
 |      Sets the module in evaluation mode.
 |      
 |      This has any effect only on certain modules. See documentations of
 |      particular modules for details of their behaviors in training/evaluation
 |      mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
 |      etc.
 |      
 |      This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.
 |      
 |      See :ref:`locally-disable-grad-doc` for a comparison between
 |      `.eval()` and several similar mechanisms that may be confused with it.
 |      
 |      Returns:
 |          Module: self
 |  
 |  extra_repr(self) -> str
 |      Set the extra representation of the module
 |      
 |      To print customized extra information, you should re-implement
 |      this method in your own modules. Both single-line and multi-line
 |      strings are acceptable.
 |  
 |  float(self:~T) -> ~T
 |      Casts all floating point parameters and buffers to ``float`` datatype.
 |      
 |      .. note::
 |          This method modifies the module in-place.
 |      
 |      Returns:
 |          Module: self
 |  
 |  get_buffer(self, target:str) -> 'Tensor'
 |      Returns the buffer given by ``target`` if it exists,
 |      otherwise throws an error.
 |      
 |      See the docstring for ``get_submodule`` for a more detailed
 |      explanation of this method's functionality as well as how to
 |      correctly specify ``target``.
 |      
 |      Args:
 |          target: The fully-qualified string name of the buffer
 |              to look for. (See ``get_submodule`` for how to specify a
 |              fully-qualified string.)
 |      
 |      Returns:
 |          torch.Tensor: The buffer referenced by ``target``
 |      
 |      Raises:
 |          AttributeError: If the target string references an invalid
 |              path or resolves to something that is not a
 |              buffer
 |  
 |  get_extra_state(self) -> Any
 |      Returns any extra state to include in the module's state_dict.
 |      Implement this and a corresponding :func:`set_extra_state` for your module
 |      if you need to store extra state. This function is called when building the
 |      module's `state_dict()`.
 |      
 |      Note that extra state should be pickleable to ensure working serialization
 |      of the state_dict. We only provide provide backwards compatibility guarantees
 |      for serializing Tensors; other objects may break backwards compatibility if
 |      their serialized pickled form changes.
 |      
 |      Returns:
 |          object: Any extra state to store in the module's state_dict
 |  
 |  get_parameter(self, target:str) -> 'Parameter'
 |      Returns the parameter given by ``target`` if it exists,
 |      otherwise throws an error.
 |      
 |      See the docstring for ``get_submodule`` for a more detailed
 |      explanation of this method's functionality as well as how to
 |      correctly specify ``target``.
 |      
 |      Args:
 |          target: The fully-qualified string name of the Parameter
 |              to look for. (See ``get_submodule`` for how to specify a
 |              fully-qualified string.)
 |      
 |      Returns:
 |          torch.nn.Parameter: The Parameter referenced by ``target``
 |      
 |      Raises:
 |          AttributeError: If the target string references an invalid
 |              path or resolves to something that is not an
 |              ``nn.Parameter``
 |  
 |  get_submodule(self, target:str) -> 'Module'
 |      Returns the submodule given by ``target`` if it exists,
 |      otherwise throws an error.
 |      
 |      For example, let's say you have an ``nn.Module`` ``A`` that
 |      looks like this:
 |      
 |      .. code-block::text
 |      
 |          A(
 |              (net_b): Module(
 |                  (net_c): Module(
 |                      (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
 |                  )
 |                  (linear): Linear(in_features=100, out_features=200, bias=True)
 |              )
 |          )
 |      
 |      (The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested
 |      submodule ``net_b``, which itself has two submodules ``net_c``
 |      and ``linear``. ``net_c`` then has a submodule ``conv``.)
 |      
 |      To check whether or not we have the ``linear`` submodule, we
 |      would call ``get_submodule("net_b.linear")``. To check whether
 |      we have the ``conv`` submodule, we would call
 |      ``get_submodule("net_b.net_c.conv")``.
 |      
 |      The runtime of ``get_submodule`` is bounded by the degree
 |      of module nesting in ``target``. A query against
 |      ``named_modules`` achieves the same result, but it is O(N) in
 |      the number of transitive modules. So, for a simple check to see
 |      if some submodule exists, ``get_submodule`` should always be
 |      used.
 |      
 |      Args:
 |          target: The fully-qualified string name of the submodule
 |              to look for. (See above example for how to specify a
 |              fully-qualified string.)
 |      
 |      Returns:
 |          torch.nn.Module: The submodule referenced by ``target``
 |      
 |      Raises:
 |          AttributeError: If the target string references an invalid
 |              path or resolves to something that is not an
 |              ``nn.Module``
 |  
 |  half(self:~T) -> ~T
 |      Casts all floating point parameters and buffers to ``half`` datatype.
 |      
 |      .. note::
 |          This method modifies the module in-place.
 |      
 |      Returns:
 |          Module: self
 |  
 |  load_state_dict(self, state_dict:'OrderedDict[str, Tensor]', strict:bool=True)
 |      Copies parameters and buffers from :attr:`state_dict` into
 |      this module and its descendants. If :attr:`strict` is ``True``, then
 |      the keys of :attr:`state_dict` must exactly match the keys returned
 |      by this module's :meth:`~torch.nn.Module.state_dict` function.
 |      
 |      Args:
 |          state_dict (dict): a dict containing parameters and
 |              persistent buffers.
 |          strict (bool, optional): whether to strictly enforce that the keys
 |              in :attr:`state_dict` match the keys returned by this module's
 |              :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
 |      
 |      Returns:
 |          ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
 |              * **missing_keys** is a list of str containing the missing keys
 |              * **unexpected_keys** is a list of str containing the unexpected keys
 |      
 |      Note:
 |          If a parameter or buffer is registered as ``None`` and its corresponding key
 |          exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
 |          ``RuntimeError``.
 |  
 |  modules(self) -> Iterator[_ForwardRef('Module')]
 |      Returns an iterator over all modules in the network.
 |      
 |      Yields:
 |          Module: a module in the network
 |      
 |      Note:
 |          Duplicate modules are returned only once. In the following
 |          example, ``l`` will be returned only once.
 |      
 |      Example::
 |      
 |          >>> l = nn.Linear(2, 2)
 |          >>> net = nn.Sequential(l, l)
 |          >>> for idx, m in enumerate(net.modules()):
 |                  print(idx, '->', m)
 |      
 |          0 -> Sequential(
 |            (0): Linear(in_features=2, out_features=2, bias=True)
 |            (1): Linear(in_features=2, out_features=2, bias=True)
 |          )
 |          1 -> Linear(in_features=2, out_features=2, bias=True)
 |  
 |  named_buffers(self, prefix:str='', recurse:bool=True) -> Iterator[Tuple[str, torch.Tensor]]
 |      Returns an iterator over module buffers, yielding both the
 |      name of the buffer as well as the buffer itself.
 |      
 |      Args:
 |          prefix (str): prefix to prepend to all buffer names.
 |          recurse (bool): if True, then yields buffers of this module
 |              and all submodules. Otherwise, yields only buffers that
 |              are direct members of this module.
 |      
 |      Yields:
 |          (string, torch.Tensor): Tuple containing the name and buffer
 |      
 |      Example::
 |      
 |          >>> for name, buf in self.named_buffers():
 |          >>>    if name in ['running_var']:
 |          >>>        print(buf.size())
 |  
 |  named_children(self) -> Iterator[Tuple[str, _ForwardRef('Module')]]
 |      Returns an iterator over immediate children modules, yielding both
 |      the name of the module as well as the module itself.
 |      
 |      Yields:
 |          (string, Module): Tuple containing a name and child module
 |      
 |      Example::
 |      
 |          >>> for name, module in model.named_children():
 |          >>>     if name in ['conv4', 'conv5']:
 |          >>>         print(module)
 |  
 |  named_modules(self, memo:Union[Set[_ForwardRef('Module')], NoneType]=None, prefix:str='', remove_duplicate:bool=True)
 |      Returns an iterator over all modules in the network, yielding
 |      both the name of the module as well as the module itself.
 |      
 |      Args:
 |          memo: a memo to store the set of modules already added to the result
 |          prefix: a prefix that will be added to the name of the module
 |          remove_duplicate: whether to remove the duplicated module instances in the result
 |          or not
 |      
 |      Yields:
 |          (string, Module): Tuple of name and module
 |      
 |      Note:
 |          Duplicate modules are returned only once. In the following
 |          example, ``l`` will be returned only once.
 |      
 |      Example::
 |      
 |          >>> l = nn.Linear(2, 2)
 |          >>> net = nn.Sequential(l, l)
 |          >>> for idx, m in enumerate(net.named_modules()):
 |                  print(idx, '->', m)
 |      
 |          0 -> ('', Sequential(
 |            (0): Linear(in_features=2, out_features=2, bias=True)
 |            (1): Linear(in_features=2, out_features=2, bias=True)
 |          ))
 |          1 -> ('0', Linear(in_features=2, out_features=2, bias=True))
 |  
 |  named_parameters(self, prefix:str='', recurse:bool=True) -> Iterator[Tuple[str, torch.nn.parameter.Parameter]]
 |      Returns an iterator over module parameters, yielding both the
 |      name of the parameter as well as the parameter itself.
 |      
 |      Args:
 |          prefix (str): prefix to prepend to all parameter names.
 |          recurse (bool): if True, then yields parameters of this module
 |              and all submodules. Otherwise, yields only parameters that
 |              are direct members of this module.
 |      
 |      Yields:
 |          (string, Parameter): Tuple containing the name and parameter
 |      
 |      Example::
 |      
 |          >>> for name, param in self.named_parameters():
 |          >>>    if name in ['bias']:
 |          >>>        print(param.size())
 |  
 |  parameters(self, recurse:bool=True) -> Iterator[torch.nn.parameter.Parameter]
 |      Returns an iterator over module parameters.
 |      
 |      This is typically passed to an optimizer.
 |      
 |      Args:
 |          recurse (bool): if True, then yields parameters of this module
 |              and all submodules. Otherwise, yields only parameters that
 |              are direct members of this module.
 |      
 |      Yields:
 |          Parameter: module parameter
 |      
 |      Example::
 |      
 |          >>> for param in model.parameters():
 |          >>>     print(type(param), param.size())
 |          <class 'torch.Tensor'> (20L,)
 |          <class 'torch.Tensor'> (20L, 1L, 5L, 5L)
 |  
 |  register_backward_hook(self, hook:Callable[[_ForwardRef('Module'), Union[Tuple[torch.Tensor, ...], torch.Tensor], Union[Tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, torch.Tensor]]) -> torch.utils.hooks.RemovableHandle
 |      Registers a backward hook on the module.
 |      
 |      This function is deprecated in favor of :meth:`~torch.nn.Module.register_full_backward_hook` and
 |      the behavior of this function will change in future versions.
 |      
 |      Returns:
 |          :class:`torch.utils.hooks.RemovableHandle`:
 |              a handle that can be used to remove the added hook by calling
 |              ``handle.remove()``
 |  
 |  register_buffer(self, name:str, tensor:Union[torch.Tensor, NoneType], persistent:bool=True) -> None
 |      Adds a buffer to the module.
 |      
 |      This is typically used to register a buffer that should not to be
 |      considered a model parameter. For example, BatchNorm's ``running_mean``
 |      is not a parameter, but is part of the module's state. Buffers, by
 |      default, are persistent and will be saved alongside parameters. This
 |      behavior can be changed by setting :attr:`persistent` to ``False``. The
 |      only difference between a persistent buffer and a non-persistent buffer
 |      is that the latter will not be a part of this module's
 |      :attr:`state_dict`.
 |      
 |      Buffers can be accessed as attributes using given names.
 |      
 |      Args:
 |          name (string): name of the buffer. The buffer can be accessed
 |              from this module using the given name
 |          tensor (Tensor or None): buffer to be registered. If ``None``, then operations
 |              that run on buffers, such as :attr:`cuda`, are ignored. If ``None``,
 |              the buffer is **not** included in the module's :attr:`state_dict`.
 |          persistent (bool): whether the buffer is part of this module's
 |              :attr:`state_dict`.
 |      
 |      Example::
 |      
 |          >>> self.register_buffer('running_mean', torch.zeros(num_features))
 |  
 |  register_forward_hook(self, hook:Callable[..., NoneType]) -> torch.utils.hooks.RemovableHandle
 |      Registers a forward hook on the module.
 |      
 |      The hook will be called every time after :func:`forward` has computed an output.
 |      It should have the following signature::
 |      
 |          hook(module, input, output) -> None or modified output
 |      
 |      The input contains only the positional arguments given to the module.
 |      Keyword arguments won't be passed to the hooks and only to the ``forward``.
 |      The hook can modify the output. It can modify the input inplace but
 |      it will not have effect on forward since this is called after
 |      :func:`forward` is called.
 |      
 |      Returns:
 |          :class:`torch.utils.hooks.RemovableHandle`:
 |              a handle that can be used to remove the added hook by calling
 |              ``handle.remove()``
 |  
 |  register_forward_pre_hook(self, hook:Callable[..., NoneType]) -> torch.utils.hooks.RemovableHandle
 |      Registers a forward pre-hook on the module.
 |      
 |      The hook will be called every time before :func:`forward` is invoked.
 |      It should have the following signature::
 |      
 |          hook(module, input) -> None or modified input
 |      
 |      The input contains only the positional arguments given to the module.
 |      Keyword arguments won't be passed to the hooks and only to the ``forward``.
 |      The hook can modify the input. User can either return a tuple or a
 |      single modified value in the hook. We will wrap the value into a tuple
 |      if a single value is returned(unless that value is already a tuple).
 |      
 |      Returns:
 |          :class:`torch.utils.hooks.RemovableHandle`:
 |              a handle that can be used to remove the added hook by calling
 |              ``handle.remove()``
 |  
 |  register_full_backward_hook(self, hook:Callable[[_ForwardRef('Module'), Union[Tuple[torch.Tensor, ...], torch.Tensor], Union[Tuple[torch.Tensor, ...], torch.Tensor]], Union[NoneType, torch.Tensor]]) -> torch.utils.hooks.RemovableHandle
 |      Registers a backward hook on the module.
 |      
 |      The hook will be called every time the gradients with respect to module
 |      inputs are computed. The hook should have the following signature::
 |      
 |          hook(module, grad_input, grad_output) -> tuple(Tensor) or None
 |      
 |      The :attr:`grad_input` and :attr:`grad_output` are tuples that contain the gradients
 |      with respect to the inputs and outputs respectively. The hook should
 |      not modify its arguments, but it can optionally return a new gradient with
 |      respect to the input that will be used in place of :attr:`grad_input` in
 |      subsequent computations. :attr:`grad_input` will only correspond to the inputs given
 |      as positional arguments and all kwarg arguments are ignored. Entries
 |      in :attr:`grad_input` and :attr:`grad_output` will be ``None`` for all non-Tensor
 |      arguments.
 |      
 |      For technical reasons, when this hook is applied to a Module, its forward function will
 |      receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
 |      of each Tensor returned by the Module's forward function.
 |      
 |      .. warning ::
 |          Modifying inputs or outputs inplace is not allowed when using backward hooks and
 |          will raise an error.
 |      
 |      Returns:
 |          :class:`torch.utils.hooks.RemovableHandle`:
 |              a handle that can be used to remove the added hook by calling
 |              ``handle.remove()``
 |  
 |  register_parameter(self, name:str, param:Union[torch.nn.parameter.Parameter, NoneType]) -> None
 |      Adds a parameter to the module.
 |      
 |      The parameter can be accessed as an attribute using given name.
 |      
 |      Args:
 |          name (string): name of the parameter. The parameter can be accessed
 |              from this module using the given name
 |          param (Parameter or None): parameter to be added to the module. If
 |              ``None``, then operations that run on parameters, such as :attr:`cuda`,
 |              are ignored. If ``None``, the parameter is **not** included in the
 |              module's :attr:`state_dict`.
 |  
 |  requires_grad_(self:~T, requires_grad:bool=True) -> ~T
 |      Change if autograd should record operations on parameters in this
 |      module.
 |      
 |      This method sets the parameters' :attr:`requires_grad` attributes
 |      in-place.
 |      
 |      This method is helpful for freezing part of the module for finetuning
 |      or training parts of a model individually (e.g., GAN training).
 |      
 |      See :ref:`locally-disable-grad-doc` for a comparison between
 |      `.requires_grad_()` and several similar mechanisms that may be confused with it.
 |      
 |      Args:
 |          requires_grad (bool): whether autograd should record operations on
 |                                parameters in this module. Default: ``True``.
 |      
 |      Returns:
 |          Module: self
 |  
 |  set_extra_state(self, state:Any)
 |      This function is called from :func:`load_state_dict` to handle any extra state
 |      found within the `state_dict`. Implement this function and a corresponding
 |      :func:`get_extra_state` for your module if you need to store extra state within its
 |      `state_dict`.
 |      
 |      Args:
 |          state (dict): Extra state from the `state_dict`
 |  
 |  share_memory(self:~T) -> ~T
 |      See :meth:`torch.Tensor.share_memory_`
 |  
 |  state_dict(self, destination=None, prefix='', keep_vars=False)
 |      Returns a dictionary containing a whole state of the module.
 |      
 |      Both parameters and persistent buffers (e.g. running averages) are
 |      included. Keys are corresponding parameter and buffer names.
 |      Parameters and buffers set to ``None`` are not included.
 |      
 |      Returns:
 |          dict:
 |              a dictionary containing a whole state of the module
 |      
 |      Example::
 |      
 |          >>> module.state_dict().keys()
 |          ['bias', 'weight']
 |  
 |  to(self, *args, **kwargs)
 |      Moves and/or casts the parameters and buffers.
 |      
 |      This can be called as
 |      
 |      .. function:: to(device=None, dtype=None, non_blocking=False)
 |         :noindex:
 |      
 |      .. function:: to(dtype, non_blocking=False)
 |         :noindex:
 |      
 |      .. function:: to(tensor, non_blocking=False)
 |         :noindex:
 |      
 |      .. function:: to(memory_format=torch.channels_last)
 |         :noindex:
 |      
 |      Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
 |      floating point or complex :attr:`dtype`\ s. In addition, this method will
 |      only cast the floating point or complex parameters and buffers to :attr:`dtype`
 |      (if given). The integral parameters and buffers will be moved
 |      :attr:`device`, if that is given, but with dtypes unchanged. When
 |      :attr:`non_blocking` is set, it tries to convert/move asynchronously
 |      with respect to the host if possible, e.g., moving CPU Tensors with
 |      pinned memory to CUDA devices.
 |      
 |      See below for examples.
 |      
 |      .. note::
 |          This method modifies the module in-place.
 |      
 |      Args:
 |          device (:class:`torch.device`): the desired device of the parameters
 |              and buffers in this module
 |          dtype (:class:`torch.dtype`): the desired floating point or complex dtype of
 |              the parameters and buffers in this module
 |          tensor (torch.Tensor): Tensor whose dtype and device are the desired
 |              dtype and device for all parameters and buffers in this module
 |          memory_format (:class:`torch.memory_format`): the desired memory
 |              format for 4D parameters and buffers in this module (keyword
 |              only argument)
 |      
 |      Returns:
 |          Module: self
 |      
 |      Examples::
 |      
 |          >>> linear = nn.Linear(2, 2)
 |          >>> linear.weight
 |          Parameter containing:
 |          tensor([[ 0.1913, -0.3420],
 |                  [-0.5113, -0.2325]])
 |          >>> linear.to(torch.double)
 |          Linear(in_features=2, out_features=2, bias=True)
 |          >>> linear.weight
 |          Parameter containing:
 |          tensor([[ 0.1913, -0.3420],
 |                  [-0.5113, -0.2325]], dtype=torch.float64)
 |          >>> gpu1 = torch.device("cuda:1")
 |          >>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
 |          Linear(in_features=2, out_features=2, bias=True)
 |          >>> linear.weight
 |          Parameter containing:
 |          tensor([[ 0.1914, -0.3420],
 |                  [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
 |          >>> cpu = torch.device("cpu")
 |          >>> linear.to(cpu)
 |          Linear(in_features=2, out_features=2, bias=True)
 |          >>> linear.weight
 |          Parameter containing:
 |          tensor([[ 0.1914, -0.3420],
 |                  [-0.5112, -0.2324]], dtype=torch.float16)
 |      
 |          >>> linear = nn.Linear(2, 2, bias=None).to(torch.cdouble)
 |          >>> linear.weight
 |          Parameter containing:
 |          tensor([[ 0.3741+0.j,  0.2382+0.j],
 |                  [ 0.5593+0.j, -0.4443+0.j]], dtype=torch.complex128)
 |          >>> linear(torch.ones(3, 2, dtype=torch.cdouble))
 |          tensor([[0.6122+0.j, 0.1150+0.j],
 |                  [0.6122+0.j, 0.1150+0.j],
 |                  [0.6122+0.j, 0.1150+0.j]], dtype=torch.complex128)
 |  
 |  to_empty(self:~T, *, device:Union[str, torch.device]) -> ~T
 |      Moves the parameters and buffers to the specified device without copying storage.
 |      
 |      Args:
 |          device (:class:`torch.device`): The desired device of the parameters
 |              and buffers in this module.
 |      
 |      Returns:
 |          Module: self
 |  
 |  train(self:~T, mode:bool=True) -> ~T
 |      Sets the module in training mode.
 |      
 |      This has any effect only on certain modules. See documentations of
 |      particular modules for details of their behaviors in training/evaluation
 |      mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
 |      etc.
 |      
 |      Args:
 |          mode (bool): whether to set training mode (``True``) or evaluation
 |                       mode (``False``). Default: ``True``.
 |      
 |      Returns:
 |          Module: self
 |  
 |  type(self:~T, dst_type:Union[torch.dtype, str]) -> ~T
 |      Casts all parameters and buffers to :attr:`dst_type`.
 |      
 |      .. note::
 |          This method modifies the module in-place.
 |      
 |      Args:
 |          dst_type (type or string): the desired type
 |      
 |      Returns:
 |          Module: self
 |  
 |  xpu(self:~T, device:Union[int, torch.device, NoneType]=None) -> ~T
 |      Moves all model parameters and buffers to the XPU.
 |      
 |      This also makes associated parameters and buffers different objects. So
 |      it should be called before constructing optimizer if the module will
 |      live on XPU while being optimized.
 |      
 |      .. note::
 |          This method modifies the module in-place.
 |      
 |      Arguments:
 |          device (int, optional): if specified, all parameters will be
 |              copied to that device
 |      
 |      Returns:
 |          Module: self
 |  
 |  zero_grad(self, set_to_none:bool=False) -> None
 |      Sets gradients of all model parameters to zero. See similar function
 |      under :class:`torch.optim.Optimizer` for more context.
 |      
 |      Args:
 |          set_to_none (bool): instead of setting to zero, set the grads to None.
 |              See :meth:`torch.optim.Optimizer.zero_grad` for details.
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors inherited from torch.nn.modules.module.Module:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)
 |  
 |  ----------------------------------------------------------------------
 |  Data and other attributes inherited from torch.nn.modules.module.Module:
 |  
 |  T_destination = ~T_destination
 |      Type variable.
 |      
 |      Usage::
 |      
 |        T = TypeVar('T')  # Can be anything
 |        A = TypeVar('A', str, bytes)  # Must be str or bytes
 |      
 |      Type variables exist primarily for the benefit of static type
 |      checkers.  They serve as the parameters for generic types as well
 |      as for generic function definitions.  See class Generic for more
 |      information on generic types.  Generic functions work as follows:
 |      
 |        def repeat(x: T, n: int) -> List[T]:
 |            '''Return a list containing n references to x.'''
 |            return [x]*n
 |      
 |        def longest(x: A, y: A) -> A:
 |            '''Return the longest of two strings.'''
 |            return x if len(x) >= len(y) else y
 |      
 |      The latter example's signature is essentially the overloading
 |      of (str, str) -> str and (bytes, bytes) -> bytes.  Also note
 |      that if the arguments are instances of some subclass of str,
 |      the return type is still plain str.
 |      
 |      At runtime, isinstance(x, T) and issubclass(C, T) will raise TypeError.
 |      
 |      Type variables defined with covariant=True or contravariant=True
 |      can be used do declare covariant or contravariant generic types.
 |      See PEP 484 for more details. By default generic types are invariant
 |      in all type variables.
 |      
 |      Type variables can be introspected. e.g.:
 |      
 |        T.__name__ == 'T'
 |        T.__constraints__ == ()
 |        T.__covariant__ == False
 |        T.__contravariant__ = False
 |        A.__constraints__ == (str, bytes)
 |  
 |  __annotations__ = {'__call__': typing.Callable[..., typing.Any], '_is_...
 |  
 |  dump_patches = False
```

```python
# 在单个GPU上训练网络
train(net, num_gpus=1, batch_size=256, lr=0.1)
```

```
test acc: 0.92, 27.5 sec/epochon [device(type='cuda', index=0)]

```

```python
# 使用2个GPU进行训练
train(net, num_gpus=2, batch_size=512, lr=0.2)
```