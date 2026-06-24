---
title: "合并行连接的网络GoogLeNet"
tags: [CV, Deep-Learning, PyTorch]
---

# 合并行连接的网络GoogLeNet

# 1. GoogLeNet




① 白色的卷积用来改变通道数，蓝色的卷积用来抽取信息。

② 最左边一条1X1卷积是用来抽取通道信息，其他的3X3卷积用来抽取空间信息。


① 输出相同的通道数，5X5比3X3的卷积层参数个数多，3X3比1X1卷积层的参数个数多。

② Inception块使用了大量1X1卷积层，使得参数相对单3X3、5X5卷积层更少。






① 1X7卷积层是看行一下空间信息，列信息不看，7X1是列看一下空间信息，行信息不看。





① 圈的大小表示耗内存的大小。


# 2. 总结


# 1. GoogLeNet（使用自定义）

```python
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs): # c1为第一条路的输出通道数、c2为第二条路的输出通道数     
        super(Inception, self).__init__(**kwargs) # python中*vars代表解包元组，**vars代表解包字典，通过这种语法可以传递不定参数。**kwage是将除了前面显式列出的参数外的其他参数, 以dict结构进行接收.                                                    
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0],c3[1],kernel_size=5,padding=2)
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels,c4,kernel_size=1)
        
    def forward(self, x):
        p1 = F.relu(self.p1_1(x))  # 第一条路的输出
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x)))) # 第二条路的输出
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1) # 批量大小的dim为0，通道数的dim为1，以通道数维度进行合并
```

```python
b1 = nn.Sequential(nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3),
                  nn.ReLU(),nn.MaxPool2d(kernel_size=3,stride=2,padding=1))  

b2 = nn.Sequential(nn.Conv2d(64,64,kernel_size=1),nn.ReLU(),
                  nn.Conv2d(64,192,kernel_size=3,padding=1),
                  nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

b3 = nn.Sequential(Inception(192,64,(96,128),(18,32),32),
                  Inception(256,128,(128,192),(32,96),64),
                  nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

b4 = nn.Sequential(Inception(480,192,(96,208),(16,48),64),
                  Inception(512,160,(112,224),(24,64),64),
                  Inception(512,128,(128,256),(24,64),64),
                  Inception(512,112,(144,288),(32,64),64),
                  Inception(528,256,(160,320),(32,128),128),
                  nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

b5 = nn.Sequential(Inception(832,256,(160,320),(32,128),128),
                   Inception(832,384,(192,384),(48,128),128),
                  nn.AdaptiveAvgPool2d((1,1)),nn.Flatten())

net = nn.Sequential(b1,b2,b3,b4,b5,nn.Linear(1024,10))
```

① 在实际的项目当中，我们往往预先只知道的是输入数据和输出数据的大小，而不知道核与步长的大小。

② 我们可以手动计算核的大小和步长的值。而自适应（Adaptive）能让我们从这样的计算当中解脱出来，只要我们给定输入数据和输出数据的大小，自适应算法能够自动帮助我们计算核的大小和每次移动的步长。

③ 相当于我们对核说，我已经给你输入和输出的数据了，你自己适应去吧。你要长多大，你每次要走多远，都由你自己决定，总之最后你的输出符合我的要求就行了。

④ 比如我们给定输入数据的尺寸是9， 输出数据的尺寸是3，那么自适应算法就能自动帮我们计算出，核的大小是3，每次移动的步长也是3，然后依据这些数据，帮我们创建好池化层。

```python
help(nn.AdaptiveAvgPool2d) # 对输入应用自适应平均池化，将feature map改为我们需要大小的输出。只需要给定输出特征图的大小就好，其中通道数前后不发生变化。                    
```

```
Help on class AdaptiveAvgPool2d in module torch.nn.modules.pooling:

class AdaptiveAvgPool2d(_AdaptiveAvgPoolNd)
 |  Applies a 2D adaptive average pooling over an input signal composed of several input planes.
 |  
 |  The output is of size H x W, for any input size.
 |  The number of output features is equal to the number of input planes.
 |  
 |  Args:
 |      output_size: the target output size of the image of the form H x W.
 |                   Can be a tuple (H, W) or a single H for a square image H x H.
 |                   H and W can be either a ``int``, or ``None`` which means the size will
 |                   be the same as that of the input.
 |  
 |  Shape:
 |      - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
 |      - Output: :math:`(N, C, S_{0}, S_{1})` or :math:`(C, S_{0}, S_{1})`, where
 |        :math:`S=\text{output\_size}`.
 |  
 |  Examples:
 |      >>> # target output size of 5x7
 |      >>> m = nn.AdaptiveAvgPool2d((5,7))
 |      >>> input = torch.randn(1, 64, 8, 9)
 |      >>> output = m(input)
 |      >>> # target output size of 7x7 (square)
 |      >>> m = nn.AdaptiveAvgPool2d(7)
 |      >>> input = torch.randn(1, 64, 10, 9)
 |      >>> output = m(input)
 |      >>> # target output size of 10x7
 |      >>> m = nn.AdaptiveAvgPool2d((None, 7))
 |      >>> input = torch.randn(1, 64, 10, 9)
 |      >>> output = m(input)
 |  
 |  Method resolution order:
 |      AdaptiveAvgPool2d
 |      _AdaptiveAvgPoolNd
 |      torch.nn.modules.module.Module
 |      builtins.object
 |  
 |  Methods defined here:
 |  
 |  forward(self, input:torch.Tensor) -> torch.Tensor
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
 |  ----------------------------------------------------------------------
 |  Data and other attributes defined here:
 |  
 |  __annotations__ = {'output_size': typing.Union[int, NoneType, typing.T...
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from _AdaptiveAvgPoolNd:
 |  
 |  __init__(self, output_size:Union[int, NoneType, Tuple[Union[int, NoneType], ...]]) -> None
 |      Initializes internal Module state, shared by both nn.Module and ScriptModule.
 |  
 |  extra_repr(self) -> str
 |      Set the extra representation of the module
 |      
 |      To print customized extra information, you should re-implement
 |      this method in your own modules. Both single-line and multi-line
 |      strings are acceptable.
 |  
 |  ----------------------------------------------------------------------
 |  Data and other attributes inherited from _AdaptiveAvgPoolNd:
 |  
 |  __constants__ = ['output_size']
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
 |  dump_patches = False
```

```python
# 为了使Fashion-MNIST上的训练短小精悍，我们将输入的高和宽从224降到96
X = torch.rand(size=(1,1,96,96))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)
```

```
Sequential output shape:	 torch.Size([1, 64, 24, 24])
Sequential output shape:	 torch.Size([1, 192, 12, 12])
Sequential output shape:	 torch.Size([1, 480, 6, 6])
Sequential output shape:	 torch.Size([1, 832, 3, 3])
Sequential output shape:	 torch.Size([1, 1024])
Linear output shape:	 torch.Size([1, 10])

```

```python
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2,
                                           padding=1))

b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))
```

```python
lr, num_epochs, batch_size =0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size,resize=96)  
d2l.train_ch6(net,train_iter,test_iter,num_epochs,lr,d2l.try_gpu())
```

```
loss 0.252, train acc 0.903, test acc 0.889
1518.0 examples/sec on cuda:0

```