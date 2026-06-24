---
title: "课程4 第2周 作业题 Keras入门与残差网络搭建"
tags: [CV, Deep-Learning, PyTorch]
---

# 课程4 第2周 作业题 Keras入门与残差网络搭建

# ***Keras入门***

# 0. 要解决的问题

① 下一次放假的时候，你决定和你的五个朋友一起度过一个星期。

② 这是一个非常好的房子，在附近有很多事情要做，但最重要的好处是每个人在家里都会感到快乐，所以任何想进入房子的人都必须证明他们目前的幸福状态。

<center>Figure 1：The Happy House</center>

① 作为一个深度学习的专家，为了确保“快乐才开门”规则得到严格的应用，你将建立一个算法，它使用来自前门摄像头的图片来检查这个人是否快乐，只有在人高兴的时候，门才会打开。

② 你收集了你的朋友和你自己的照片，被前门的摄像头拍了下来，数据集已经标记好了。

<center>Figure 2：Dataset Labels</center>

# 1. 导入包

① 为什么我们要使用Keras框架呢？

② Keras是为了使深度学习工程师能够很快地建立和实验不同的模型的框架，正如TensorFlow是一个比Python更高级的框架，Keras是一个更高层次的框架，并提供了额外的抽象方法。

③ 最关键的是Keras能够以最短的时间让想法变为现实。

④ 然而，Keras比底层框架更具有限制性，所以有一些非常复杂的模型可以在TensorFlow中实现，但在Keras中却没有（没有更多困难）。

⑤ 话虽如此，Keras对许多常见模型都能正常运行。

```python
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

%matplotlib inline
```

```
Using TensorFlow backend.
D:\11_Anaconda\envs\py3.6.3\lib\site-packages\requests\__init__.py:104: RequestsDependencyWarning: urllib3 (1.26.11) or chardet (5.0.0)/charset_normalizer (2.0.12) doesn't match a supported version!
  RequestsDependencyWarning)

```

⑥ 正如你所看到的，我们已经从Keras中导入了很多功能，只需直接调用它们即可轻松使用它们。
 - 比如：X = Input(…) 或者X = ZeroPadding2D(…).

# 2. 导入数据集

```python
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()          

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape)) 
```

```
number of training examples = 600
number of test examples = 150
X_train shape: (600, 64, 64, 3)
Y_train shape: (600, 1)
X_test shape: (150, 64, 64, 3)
Y_test shape: (150, 1)

```

# 3. 使用Keras框架构建模型

① Keras非常适合快速制作模型，它可以在很短的时间内建立一个很优秀的模型，举个例子：

```python
def model(input_shape):
    """
    模型大纲
    """
    # 定义一个 tensor 的 placeholder，维度为 input_shape
    X_input = Input(input_shape)
    
    # 使用 0 填充：X_input的周围填充 0
    X = ZeroPadding2D((3,3))(X_input)
    
    # 对X使用 CONV -> BN -> RELU 块
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    
    # 最大值池化层
    X = MaxPooling2D((2,2),name="max_pool")(X)
    
    # 降维，矩阵转化为向量 + 全连接层
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)
    
    # 创建模型，讲话创建一个模型的实体，我们可以用它来训练、测试。
    model = Model(inputs = X_input, outputs = X, name='HappyModel')
    
    return model
```

② Keras框架使用的变量名和我们以前使用的numpy和TensorFlow变量不一样。
 - 它不是在前向传播的每一步上创建新变量（比如X, Z1, A1, Z2, A2,…）以便于不同层之间的计算。
 - 在Keras中，我们使用X覆盖了所有的值，没有保存每一层结果，我们只需要最新的值，唯一例外的就是X_input，我们将它分离出来是因为它是输入的数据，我们要在最后的创建模型那一步中用到。

```python
def HappyModel(input_shape):
    """
    实现一个检测笑容的模型
    
    参数：
        input_shape - 输入的数据的维度
    返回：
        model - 创建的Keras的模型
        
    """
    
    #你可以参考和上面的大纲
    X_input = Input(input_shape)

    #使用0填充：X_input的周围填充0
    X = ZeroPadding2D((3, 3))(X_input)

    #对X使用 CONV -> BN -> RELU 块
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    #最大值池化层
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    #降维，矩阵转化为向量 + 全连接层
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    #创建模型，讲话创建一个模型的实体，我们可以用它来训练、测试。
    model = Model(inputs=X_input, outputs=X, name='HappyModel')

    return model
```

③ 现在我们已经设计好了我们的模型了，要训练并测试模型我们需要这么做：

1. 创建一个模型实体。
2. 编译模型，可以使用这个语句：model.compile(optimizer = "...", loss = "...", metrics = ["accuracy"])。
3. 训练模型：model.fit(x = ..., y = ..., epochs = ..., batch_size = ...)。
4. 评估模型：model.evaluate(x = ..., y = ...)。

```python
# 创建一个模型实体
happy_model = HappyModel(X_train.shape[1:])
# 编译模型
happy_model.compile("adam","binary_crossentropy", metrics=['accuracy'])
# 训练模型
# 请注意，此操作会花费你大约 6-10 分钟。
happy_model.fit(X_train, Y_train, epochs=40, batch_size=50)
# 评估模型
preds = happy_model.evaluate(X_test, Y_test, batch_size=32, verbose=1, sample_weight=None)
print ("误差值 = " + str(preds[0]))
print ("准确度 = " + str(preds[1]))
```

```
Epoch 1/40
600/600 [==============================] - 5s 9ms/step - loss: 1.1980 - acc: 0.6383
Epoch 2/40
600/600 [==============================] - 5s 9ms/step - loss: 0.2719 - acc: 0.8700
Epoch 3/40
600/600 [==============================] - 5s 9ms/step - loss: 0.1740 - acc: 0.9333
Epoch 4/40
600/600 [==============================] - 5s 8ms/step - loss: 0.1296 - acc: 0.9533
Epoch 5/40
600/600 [==============================] - 5s 8ms/step - loss: 0.1109 - acc: 0.9600
Epoch 6/40
600/600 [==============================] - 5s 9ms/step - loss: 0.0961 - acc: 0.9700
Epoch 7/40
600/600 [==============================] - 5s 9ms/step - loss: 0.0814 - acc: 0.9750
Epoch 8/40
600/600 [==============================] - 5s 9ms/step - loss: 0.0766 - acc: 0.9750
Epoch 9/40
600/600 [==============================] - 5s 9ms/step - loss: 0.0554 - acc: 0.9850
Epoch 10/40
600/600 [==============================] - 5s 9ms/step - loss: 0.0425 - acc: 0.9917
Epoch 11/40
600/600 [==============================] - 6s 10ms/step - loss: 0.0438 - acc: 0.9883
Epoch 12/40
600/600 [==============================] - 6s 9ms/step - loss: 0.0442 - acc: 0.9867
Epoch 13/40
600/600 [==============================] - 5s 9ms/step - loss: 0.0407 - acc: 0.9900
Epoch 14/40
600/600 [==============================] - 5s 9ms/step - loss: 0.0311 - acc: 0.9950
Epoch 15/40
600/600 [==============================] - 6s 9ms/step - loss: 0.0359 - acc: 0.9933
Epoch 16/40
600/600 [==============================] - 5s 9ms/step - loss: 0.0390 - acc: 0.9867
Epoch 17/40
600/600 [==============================] - 5s 9ms/step - loss: 0.0385 - acc: 0.9883
Epoch 18/40
600/600 [==============================] - 5s 9ms/step - loss: 0.0270 - acc: 0.9933
Epoch 19/40
600/600 [==============================] - 5s 9ms/step - loss: 0.0344 - acc: 0.9867
Epoch 20/40
600/600 [==============================] - 5s 9ms/step - loss: 0.0200 - acc: 0.9967
Epoch 21/40
600/600 [==============================] - 6s 9ms/step - loss: 0.0319 - acc: 0.9883
Epoch 22/40
600/600 [==============================] - 5s 9ms/step - loss: 0.0219 - acc: 0.9983
Epoch 23/40
600/600 [==============================] - 5s 9ms/step - loss: 0.0280 - acc: 0.9933
Epoch 24/40
600/600 [==============================] - 5s 9ms/step - loss: 0.0259 - acc: 0.9933
Epoch 25/40
600/600 [==============================] - 5s 9ms/step - loss: 0.0172 - acc: 0.9967
Epoch 26/40
600/600 [==============================] - 6s 9ms/step - loss: 0.0165 - acc: 0.9967
Epoch 27/40
600/600 [==============================] - 5s 9ms/step - loss: 0.0157 - acc: 0.9950
Epoch 28/40
600/600 [==============================] - 5s 9ms/step - loss: 0.0139 - acc: 0.9983
Epoch 29/40
600/600 [==============================] - 5s 9ms/step - loss: 0.0240 - acc: 0.9933
Epoch 30/40
600/600 [==============================] - 5s 9ms/step - loss: 0.0189 - acc: 0.9933
Epoch 31/40
600/600 [==============================] - 6s 9ms/step - loss: 0.0189 - acc: 0.9950
Epoch 32/40
600/600 [==============================] - 6s 9ms/step - loss: 0.0152 - acc: 0.9950
Epoch 33/40
600/600 [==============================] - 6s 9ms/step - loss: 0.0136 - acc: 0.9967
Epoch 34/40
600/600 [==============================] - 6s 9ms/step - loss: 0.0123 - acc: 0.9967
Epoch 35/40
600/600 [==============================] - 5s 9ms/step - loss: 0.0239 - acc: 0.9950
Epoch 36/40
600/600 [==============================] - 6s 9ms/step - loss: 0.0160 - acc: 0.9933
Epoch 37/40
600/600 [==============================] - 6s 10ms/step - loss: 0.0115 - acc: 1.0000
Epoch 38/40
600/600 [==============================] - 6s 10ms/step - loss: 0.0085 - acc: 0.9983
Epoch 39/40
600/600 [==============================] - 6s 9ms/step - loss: 0.0089 - acc: 0.9983
Epoch 40/40
600/600 [==============================] - 5s 9ms/step - loss: 0.0178 - acc: 0.9950
150/150 [==============================] - 1s 4ms/step
误差值 = 0.07078419178724289
准确度 = 0.9466666706403096

```

① 只要准确度大于75%就算正常，如果你的准确度没有大于75%，可以尝试以下方法来达到此目的：

 - 你可以尝试改变模型：

   - X = Conv2D(32, (3, 3), strides = (1, 1), name = 'conv0')(X)

   - X = BatchNormalization(axis = 3, name = 'bn0')(X)

   - X = Activation('relu')(X)
   
 - 你可以在每个块后面使用最大值池化层，它将会减少宽、高的维度。
 
 - 改变优化器，这里我们使用的是Adam
 
 - 如果模型难以运行，并且遇到了内存不够的问题，那么就降低batch_size(12通常是一个很好的折中方案)
 
 - 运行更多代，直到看到有良好效果的时候。

② 即使你已经达到了75%的准确度，你也可以继续优化你的模型，以获得更好的结果。

# 4. 测试你的图片

① 因为对这些数据进行训练的模型可能或不能处理你自己的图片，但是你可以试一试嘛：

```python
#网上随便找的图片，侵删
img_path = 'datasets/test_happy.png' 

img_origianl = image.load_img(img_path)
imshow(img_origianl)

img = image.load_img(img_path, target_size=(64, 64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

print(happy_model.predict(x))
```

```
[[1.]]

```

# 5. 其他一些有用的功能

 - model.summary()：打印出你的每一层的大小细节
 - plot_model() : 绘制出布局图

```python
happy_model.summary()
```

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 64, 64, 3)         0         
_________________________________________________________________
zero_padding2d_1 (ZeroPaddin (None, 70, 70, 3)         0         
_________________________________________________________________
conv0 (Conv2D)               (None, 64, 64, 32)        4736      
_________________________________________________________________
bn0 (BatchNormalization)     (None, 64, 64, 32)        128       
_________________________________________________________________
activation_1 (Activation)    (None, 64, 64, 32)        0         
_________________________________________________________________
max_pool (MaxPooling2D)      (None, 32, 32, 32)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 32768)             0         
_________________________________________________________________
fc (Dense)                   (None, 1)                 32769     
=================================================================
Total params: 37,633
Trainable params: 37,569
Non-trainable params: 64
_________________________________________________________________

```

# ***残差网络搭建***

# 0. 要解决的问题

① 这里我们将学习怎样使用残差网络构建一个非常深的卷积网络。

② 理论上越深的网络越能够实现越复杂的功能，但是在实际上却非常难以训练。

③ 残差网络就是为了解决深网络的难以训练的问题的。

④ 在本文章中，我们将：

 - 实现基本的残差块。
 - 将这些残差块放在一起，实现并训练用于图像分类的神经网络。
 - 本次实验将使用Keras框架

# 1. 导入库

① 在解决问题之前，我们先来导入库函数：

```python
import numpy as np
import tensorflow as tf
import cv2

from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform

import pydot
from IPython.display import SVG
import scipy.misc
from matplotlib.pyplot import imshow
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

import resnets_utils
```

```
D:\11_Anaconda\envs\py3.6.3\lib\site-packages\requests\__init__.py:104: RequestsDependencyWarning: urllib3 (1.26.11) or chardet (5.0.0)/charset_normalizer (2.0.12) doesn't match a supported version!
  RequestsDependencyWarning)
Using TensorFlow backend.

```

# 2. 深层网络的麻烦

① 上周，我们构建了第一个卷积神经网络。

② 最近几年，卷积神经网络变得越来越深，从几层（例如AlexNet）到超过一百层。

① 使用深层网络最大的好处就是它能够完成很复杂的功能，它能够从边缘（浅层）到非常复杂的特征（深层）中不同的抽象层次的特征中学习。

② 然而，使用比较深的网络通常没有什么好处，一个特别大的麻烦就在于训练的时候会产生梯度消失，非常深的网络通常会有一个梯度信号，该信号会迅速的消退，从而使得梯度下降变得非常缓慢。

③ 更具体的说，在梯度下降的过程中，当你从最后一层回到第一层的时候，你在每个步骤上乘以权重矩阵，因此梯度值可以迅速的指数式地减少到0（在极少数的情况下会迅速增长，造成梯度爆炸）。

① 在训练的过程中，你可能会看到开始几层的梯度的大小（或范数）迅速下降到0，如下图：


<center>Figure 1：梯度消失</center>

② 为了解决这个问题，我们将构建残差网络。

# 3. 构建残差网络

① 在残差网络中，一个“捷径（shortcut）”或者说“跳跃连接（skip connection）”允许梯度直接反向传播到更浅的层，如下图：


<center>图2：残差网络中跳跃连接的残差块示意</center>

① 图像左边是神经网络的主路，图像右边是添加了一条捷径的主路，通过这些残差块堆叠在一起，可以形成一个非常深的网络。

② 我们在视频中可以看到使用捷径的方式使得每一个残差块能够很容易学习到恒等式功能，这意味着我们可以添加很多的残差块而不会损害训练集的表现。

③ 残差块有两种类型，主要取决于输入输出的维度是否相同，下面我们来看看吧~

## 3.1 恒等块

① 恒等块是残差网络使用的的标准块，对应于输入的激活值（比如$a^{[l]}$）与输出激活值（比如$a^{[l+1]}$）具有相同的维度。

② 为了具象化残差块的不同步骤，我们来看看下面的图吧~


<center>图3：恒等块</center>

<center>使用的是跳跃连接，幅度为两层</center>

① 上图中，上面的曲线路径是“捷径”，下面的直线路径是主路径。

② 在上图中，我们依旧把CONV2D 与 ReLU包含到了每个步骤中，为了提升训练的速度，我们在每一步也把数据进行了归一化（BatchNorm），不要害怕这些东西，因为Keras框架已经实现了这些东西，调用BatchNorm只需要一行代码。

① 在实践中，我们要做一个更强大的版本：跳跃连接会跳过3个隐藏层而不是两个，就像下图：


<center>图4：恒等块</center>

<center>使用的是跳跃连接，幅度为三层</center>

② 下面是各个步骤：

1. 主路径的第一部分：

 - 第一个CONV2D具有形状为（1,1）和步幅为（1,1）的$F_1$个滤波器。其填充为“valid”，其名称应为conv_name_base + '2a'。使用0作为随机初始化的种子。
 - 第一个BatchNorm标准化通道轴。它的名字应该是bn_name_base + '2a'。
 - 然后应用ReLU激活函数。

2. 主路径的第二部分：

 - 第二个CONV2D具有形状为$（f,f）$的步幅为（1,1）的$F_2$个滤波器。其填充为“same”，其名称应为conv_name_base + '2b'。使用0作为随机初始化的种子。
 - 第二个BatchNorm标准化通道轴。它的名字应该是bn_name_base + '2b'。
 - 然后应用ReLU激活函数。

3. 主路径的第三部分：

 - 第三个CONV2D具有形状为（1,1）和步幅为（1,1）的$F_3$个滤波器。其填充为“valid”，其名称应为conv_name_base + '2c'。使用0作为随机初始化的种子。
 - 第三个BatchNorm标准化通道轴。它的名字应该是bn_name_base + '2c'。请注意，此组件中没有ReLU激活函数。

4. 最后一步：

 - 将shortcut和输入添加在一起。
 - 然后应用ReLU激活函数。

```python
def identity_block(X, f, filters, stage, block):
    """
    实现图3的恒等块
    
    参数：
        X - 输入的tensor类型的数据，维度为( m, n_H_prev, n_W_prev, n_H_prev )
        f - 整数，指定主路径中间的CONV窗口的维度
        filters - 整数列表，定义了主路径每层的卷积层的过滤器数量
        stage - 整数，根据每层的位置来命名每一层，与block参数一起使用。
        block - 字符串，据每层的位置来命名每一层，与stage参数一起使用。
        
    返回：
        X - 恒等块的输出，tensor类型，维度为(n_H, n_W, n_C)
    
    """
    
    #定义命名规则
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base   = "bn"  + str(stage) + block + "_branch"
    
    #获取过滤器
    F1, F2, F3 = filters
    
    #保存输入数据，将会用于为主路径添加捷径
    X_shortcut = X
    
    #主路径的第一部分
    ##卷积层
    X = Conv2D(filters=F1, kernel_size=(1,1), strides=(1,1) ,padding="valid",
               name=conv_name_base+"2a", kernel_initializer=glorot_uniform(seed=0))(X)
    ##归一化
    X = BatchNormalization(axis=3,name=bn_name_base+"2a")(X)
    ##使用ReLU激活函数
    X = Activation("relu")(X)
    
    #主路径的第二部分
    ##卷积层
    X = Conv2D(filters=F2, kernel_size=(f,f),strides=(1,1), padding="same",
               name=conv_name_base+"2b", kernel_initializer=glorot_uniform(seed=0))(X)
    ##归一化
    X = BatchNormalization(axis=3,name=bn_name_base+"2b")(X)
    ##使用ReLU激活函数
    X = Activation("relu")(X)
    
    
    #主路径的第三部分
    ##卷积层
    X = Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding="valid",
               name=conv_name_base+"2c", kernel_initializer=glorot_uniform(seed=0))(X)
    ##归一化
    X = BatchNormalization(axis=3,name=bn_name_base+"2c")(X)
    ##没有ReLU激活函数
    
    #最后一步：
    ##将捷径与输入加在一起
    X = Add()([X,X_shortcut])
    ##使用ReLU激活函数
    X = Activation("relu")(X)
    
    return X
```

```python
tf.reset_default_graph()

with tf.Session() as test:
    np.random.seed(1)
    A_prev = tf.placeholder("float", [3, 4, 4, 6])
    X = np.random.randn(3, 4, 4, 6)
    A = identity_block(A_prev, f = 2, filters = [2, 4, 6], stage = 1, block = 'a')
    test.run(tf.global_variables_initializer())
    out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
    print("out = " + str(out[0][1][1][0]))
```

```
out = [0.19716819 0.         1.3561226  2.1713073  0.         1.3324987 ]

```

## 3.2 卷积块

① 我们已经实现了残差网络的恒等块，现在，残差网络的卷积块是另一种类型的残差块，它适用于输入输出的维度不一致的情况，它不同于上面的恒等块，与之区别在于，捷径中有一个CONV2D层，如下图：


<center>图5：卷积块</center>

② 捷径中的卷积层将把输入$x$卷积为不同的维度，因此在主路径最后那里需要适配捷径中的维度。

③ 比如：把激活值中的宽高减少2倍，我们可以使用1x1的卷积，步伐为2。捷径上的卷积层不使用任何非线性激活函数，它的主要作用是仅仅应用（学习后的）线性函数来减少输入的维度，以便在后面的加法步骤中的维度相匹配。

④ 卷积块的细节如下：

1. 主路径的第一部分：

 - 第一个卷积层有$F_1$个过滤器，其维度为（1，1），步伐为（s，s），使用“valid”的填充方式，命名规则为conv_name_base + '2a'。
 - 第一个规范层是通道的轴归一化，其命名规则为bn_name_base + '2a'。
 - 使用ReLU激活函数，它没有命名规则也没有超参数。

2. 主路径的第二部分：

 - 第二个卷积层有$F_2$个过滤器，其维度为（f ff，f ff），步伐为（1 11，1 11），使用“same”的填充方式，命名规则为conv_name_base + '2b'
 - 第二个BatchNorm标准化通道轴。它的名字应该是bn_name_base + '2b'。
 - 然后应用ReLU激活函数。

3. 主路径的第三部分：

 - 第三个卷积层有$F_3$个过滤器，其维度为（1 11，1 11），步伐为（s ss，s ss），使用“valid”的填充方式，命名规则为conv_name_base + '2c'。
 - 第三个规范层是通道的轴归一化，其命名规则为bn_name_base + '2c'。
 - 没有激活函数。

4. 捷径：

 - 此卷积层有$F_3$个过滤器，其维度为（1 11，1 11），步伐为（s ss，s ss），使用“valid”的填充方式，命名规则为conv_name_base + '1'。
 - 此规范层是通道的轴归一化，其命名规则为bn_name_base + '1'。

5. 最后一步：

 - 将捷径与输入加在一起。
 - 使用ReLU激活函数。

```python
def convolutional_block(X, f, filters, stage, block, s=2):
    """
    实现图5的卷积块
    
    参数：
        X - 输入的tensor类型的变量，维度为( m, n_H_prev, n_W_prev, n_C_prev)
        f - 整数，指定主路径中间的CONV窗口的维度
        filters - 整数列表，定义了主路径每层的卷积层的过滤器数量
        stage - 整数，根据每层的位置来命名每一层，与block参数一起使用。
        block - 字符串，据每层的位置来命名每一层，与stage参数一起使用。
        s - 整数，指定要使用的步幅
    
    返回：
        X - 卷积块的输出，tensor类型，维度为(n_H, n_W, n_C)
    """
    
    #定义命名规则
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base   = "bn"  + str(stage) + block + "_branch"
    
    #获取过滤器数量
    F1, F2, F3 = filters
    
    #保存输入数据
    X_shortcut = X
    
    #主路径
    ##主路径第一部分
    X = Conv2D(filters=F1, kernel_size=(1,1), strides=(s,s), padding="valid",
               name=conv_name_base+"2a", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name_base+"2a")(X)
    X = Activation("relu")(X)
    
    ##主路径第二部分
    X = Conv2D(filters=F2, kernel_size=(f,f), strides=(1,1), padding="same",
               name=conv_name_base+"2b", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name_base+"2b")(X)
    X = Activation("relu")(X)
    
    ##主路径第三部分
    X = Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding="valid",
               name=conv_name_base+"2c", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name_base+"2c")(X)
    
    #捷径
    X_shortcut = Conv2D(filters=F3, kernel_size=(1,1), strides=(s,s), padding="valid",
               name=conv_name_base+"1", kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3,name=bn_name_base+"1")(X_shortcut)
    
    #最后一步
    X = Add()([X,X_shortcut])
    X = Activation("relu")(X)
    
    return X
```

```python
tf.reset_default_graph()

with tf.Session() as test:
    np.random.seed(1)
    A_prev = tf.placeholder("float",[3,4,4,6])
    X = np.random.randn(3,4,4,6)
    
    A = convolutional_block(A_prev,f=2,filters=[2,4,6],stage=1,block="a")
    test.run(tf.global_variables_initializer())
    
    out = test.run([A],feed_dict={A_prev:X,K.learning_phase():0})
    print("out = " + str(out[0][1][1][0]))
    
    test.close()
```

```
out = [0.09018463 1.2348979  0.46822023 0.03671762 0.         0.65516603]

```

## 3.3 构建残差网络(50层)

① 我们已经做完所需要的所有残差块了，下面这个图就描述了神经网络的算法细节，图中的"ID BLOCK"是指标准的恒等块，"ID BLOCK X3"是指把三个恒等块放在一起。


<center>图6：ResNet50 模型</center>

② 此ResNet-50模型的详细结构是：

1. 对输入数据进行0填充，padding =（3,3）
2. stage1：
 - 卷积层有64个过滤器，其维度为（7，7），步伐为（2，2），命名为“conv1”。
 - 规范层（BatchNorm）对输入数据进行通道轴归一化。
 - 最大值池化层使用一个（3，3）的窗口和（2，2）的步伐。
 
3. stage2：
 - 卷积块使用f=3个大小为[64，64，256]的过滤器，f=3，s=1,block=“a”。
 - 2个恒等块使用三个大小为[64，64，256]的过滤器，f=3，block=“b”、“c”。
 
4. stage3：
 - 卷积块使用f=3个大小为[128,128,512]的过滤器，f=3，s=2,block=“a”。
 - 3个恒等块使用三个大小为[128,128,512]的过滤器，f=3，block=“b”、“c”、“d”。
 
5. stage4：
 - 卷积块使用f=3个大小为[256,256,1024]的过滤器，f=3，s=2,block=“a”。
 - 5个恒等块使用三个大小为[256,256,1024]的过滤器，f=3，block=“b”、“c”、“d”、“e”、“f”。
 
6. stage5：
 - 卷积块使用f=3个大小为[512,512,2048]的过滤器，f=3，s=2,block=“a”。
 - 2个恒等块使用三个大小为[256,256,2048]的过滤器，f=3，block=“b”、“c”。
 
7. 均值池化层使用维度为（2,2）的窗口，命名为“avg_pool”。
8. 展开操作没有任何超参数以及命名。
9. 全连接层（密集连接）使用softmax激活函数，命名为"fc" + str(classes)。

```python
def ResNet50(input_shape=(64,64,3),classes=6):
    """
    实现ResNet50
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
    
    参数：
        input_shape - 图像数据集的维度
        classes - 整数，分类数
        
    返回：
        model - Keras框架的模型
        
    """
    
    #定义tensor类型的输入数据
    X_input = Input(input_shape)
    
    #0填充
    X = ZeroPadding2D((3,3))(X_input)
    
    #stage1
    X = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), name="conv1",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name="bn_conv1")(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(pool_size=(3,3), strides=(2,2))(X)
    
    #stage2
    X = convolutional_block(X, f=3, filters=[64,64,256], stage=2, block="a", s=1)
    X = identity_block(X, f=3, filters=[64,64,256], stage=2, block="b")
    X = identity_block(X, f=3, filters=[64,64,256], stage=2, block="c")
    
    #stage3
    X = convolutional_block(X, f=3, filters=[128,128,512], stage=3, block="a", s=2)
    X = identity_block(X, f=3, filters=[128,128,512], stage=3, block="b")
    X = identity_block(X, f=3, filters=[128,128,512], stage=3, block="c")
    X = identity_block(X, f=3, filters=[128,128,512], stage=3, block="d")
    
    #stage4
    X = convolutional_block(X, f=3, filters=[256,256,1024], stage=4, block="a", s=2)
    X = identity_block(X, f=3, filters=[256,256,1024], stage=4, block="b")
    X = identity_block(X, f=3, filters=[256,256,1024], stage=4, block="c")
    X = identity_block(X, f=3, filters=[256,256,1024], stage=4, block="d")
    X = identity_block(X, f=3, filters=[256,256,1024], stage=4, block="e")
    X = identity_block(X, f=3, filters=[256,256,1024], stage=4, block="f")
    
    #stage5
    X = convolutional_block(X, f=3, filters=[512,512,2048], stage=5, block="a", s=2)
    X = identity_block(X, f=3, filters=[512,512,2048], stage=5, block="b")
    X = identity_block(X, f=3, filters=[512,512,2048], stage=5, block="c")
    
    #均值池化层
    X = AveragePooling2D(pool_size=(2,2),padding="same")(X)
    
    #输出层
    X = Flatten()(X)
    X = Dense(classes, activation="softmax", name="fc"+str(classes),
              kernel_initializer=glorot_uniform(seed=0))(X)
    
    
    #创建模型
    model = Model(inputs=X_input, outputs=X, name="ResNet50")
    
    return model
```

③ 然后我们对模型做实体化和编译工作：

```python
model = ResNet50(input_shape=(64,64,3),classes=6)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
```

④ 现在模型已经准备好了，接下来就是加载训练集进行训练。


<center>图7：手势数据集</center>

```python
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = resnets_utils.load_dataset()

# Normalize image vectors
X_train = X_train_orig / 255.
X_test = X_test_orig / 255.

# Convert training and test labels to one hot matrices
Y_train = resnets_utils.convert_to_one_hot(Y_train_orig, 6).T
Y_test = resnets_utils.convert_to_one_hot(Y_test_orig, 6).T

print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))
```

```
number of training examples = 1080
number of test examples = 120
X_train shape: (1080, 64, 64, 3)
Y_train shape: (1080, 6)
X_test shape: (120, 64, 64, 3)
Y_test shape: (120, 6)

```

⑤ 运行模型两代，batch=32，每代大约3分钟左右。

```python
model.fit(X_train,Y_train,epochs=2,batch_size=32)
```

```
Epoch 1/2
1080/1080 [==============================] - 80s 74ms/step - loss: 2.7403 - acc: 0.3176
Epoch 2/2
1080/1080 [==============================] - 74s 69ms/step - loss: 1.5414 - acc: 0.5361

```

输出：
```
<keras.callbacks.History at 0x28a139eac50>
```

① 在$\frac{1}{2}$Epoch中，loss在1\~5之间算正常，acc在0.2\~0.5之间算正常，你的结果和我的不一致也算正常。

② 在$\frac{2}{2}$Epoch中，loss在1\~5之间算正常，acc在0.2\~0.5之间算正常，你可以看到损失在下降，准确率在上升。

```python
preds = model.evaluate(X_test,Y_test)

print("误差值 = " + str(preds[0]))
print("准确率 = " + str(preds[1]))
```

```
120/120 [==============================] - 2s 21ms/step
误差值 = 13.431746482849121
准确率 = 0.16666666716337203

```

① 在完成这个任务之后，如果愿意的话，您还可以选择继续训练RESNET。

② 当我们训练20代时，我们得到了更好的性能，但是在得在CPU上训练需要一个多小时。

③ 使用GPU的话，博主已经在手势数据集上训练了自己的RESNET50模型的权重，你可以使用下面的代码载并运行博主的训练模型，加载模型可能需要1min。

```python
# 加载模型
model = load_model("datasets/ResNet50.h5")
```

```python
preds = model.evaluate(X_test,Y_test)
print("误差值 = " + str(preds[0]))
print("准确率 = " + str(preds[1]))
```

```
120/120 [==============================] - 3s 23ms/step
误差值 = 0.10854307860136032
准确率 = 0.9666666626930237

```

# 4. 查看网络的节点大小

```python
model.summary()
```

```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 64, 64, 3)    0                                            
__________________________________________________________________________________________________
zero_padding2d_1 (ZeroPadding2D (None, 70, 70, 3)    0           input_1[0][0]                    
__________________________________________________________________________________________________
conv1 (Conv2D)                  (None, 32, 32, 64)   9472        zero_padding2d_1[0][0]           
__________________________________________________________________________________________________
bn_conv1 (BatchNormalization)   (None, 32, 32, 64)   256         conv1[0][0]                      
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 32, 32, 64)   0           bn_conv1[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 15, 15, 64)   0           activation_1[0][0]               
__________________________________________________________________________________________________
res2a_branch2a (Conv2D)         (None, 15, 15, 64)   4160        max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
bn2a_branch2a (BatchNormalizati (None, 15, 15, 64)   256         res2a_branch2a[0][0]             
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 15, 15, 64)   0           bn2a_branch2a[0][0]              
__________________________________________________________________________________________________
res2a_branch2b (Conv2D)         (None, 15, 15, 64)   36928       activation_2[0][0]               
__________________________________________________________________________________________________
bn2a_branch2b (BatchNormalizati (None, 15, 15, 64)   256         res2a_branch2b[0][0]             
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 15, 15, 64)   0           bn2a_branch2b[0][0]              
__________________________________________________________________________________________________
res2a_branch2c (Conv2D)         (None, 15, 15, 256)  16640       activation_3[0][0]               
__________________________________________________________________________________________________
res2a_branch1 (Conv2D)          (None, 15, 15, 256)  16640       max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
bn2a_branch2c (BatchNormalizati (None, 15, 15, 256)  1024        res2a_branch2c[0][0]             
__________________________________________________________________________________________________
bn2a_branch1 (BatchNormalizatio (None, 15, 15, 256)  1024        res2a_branch1[0][0]              
__________________________________________________________________________________________________
add_1 (Add)                     (None, 15, 15, 256)  0           bn2a_branch2c[0][0]              
                                                                 bn2a_branch1[0][0]               
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 15, 15, 256)  0           add_1[0][0]                      
__________________________________________________________________________________________________
res2b_branch2a (Conv2D)         (None, 15, 15, 64)   16448       activation_4[0][0]               
__________________________________________________________________________________________________
bn2b_branch2a (BatchNormalizati (None, 15, 15, 64)   256         res2b_branch2a[0][0]             
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 15, 15, 64)   0           bn2b_branch2a[0][0]              
__________________________________________________________________________________________________
res2b_branch2b (Conv2D)         (None, 15, 15, 64)   36928       activation_5[0][0]               
__________________________________________________________________________________________________
bn2b_branch2b (BatchNormalizati (None, 15, 15, 64)   256         res2b_branch2b[0][0]             
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 15, 15, 64)   0           bn2b_branch2b[0][0]              
__________________________________________________________________________________________________
res2b_branch2c (Conv2D)         (None, 15, 15, 256)  16640       activation_6[0][0]               
__________________________________________________________________________________________________
bn2b_branch2c (BatchNormalizati (None, 15, 15, 256)  1024        res2b_branch2c[0][0]             
__________________________________________________________________________________________________
add_2 (Add)                     (None, 15, 15, 256)  0           bn2b_branch2c[0][0]              
                                                                 activation_4[0][0]               
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 15, 15, 256)  0           add_2[0][0]                      
__________________________________________________________________________________________________
res2c_branch2a (Conv2D)         (None, 15, 15, 64)   16448       activation_7[0][0]               
__________________________________________________________________________________________________
bn2c_branch2a (BatchNormalizati (None, 15, 15, 64)   256         res2c_branch2a[0][0]             
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 15, 15, 64)   0           bn2c_branch2a[0][0]              
__________________________________________________________________________________________________
res2c_branch2b (Conv2D)         (None, 15, 15, 64)   36928       activation_8[0][0]               
__________________________________________________________________________________________________
bn2c_branch2b (BatchNormalizati (None, 15, 15, 64)   256         res2c_branch2b[0][0]             
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 15, 15, 64)   0           bn2c_branch2b[0][0]              
__________________________________________________________________________________________________
res2c_branch2c (Conv2D)         (None, 15, 15, 256)  16640       activation_9[0][0]               
__________________________________________________________________________________________________
bn2c_branch2c (BatchNormalizati (None, 15, 15, 256)  1024        res2c_branch2c[0][0]             
__________________________________________________________________________________________________
add_3 (Add)                     (None, 15, 15, 256)  0           bn2c_branch2c[0][0]              
                                                                 activation_7[0][0]               
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 15, 15, 256)  0           add_3[0][0]                      
__________________________________________________________________________________________________
res3a_branch2a (Conv2D)         (None, 8, 8, 128)    32896       activation_10[0][0]              
__________________________________________________________________________________________________
bn3a_branch2a (BatchNormalizati (None, 8, 8, 128)    512         res3a_branch2a[0][0]             
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 8, 8, 128)    0           bn3a_branch2a[0][0]              
__________________________________________________________________________________________________
res3a_branch2b (Conv2D)         (None, 8, 8, 128)    147584      activation_11[0][0]              
__________________________________________________________________________________________________
bn3a_branch2b (BatchNormalizati (None, 8, 8, 128)    512         res3a_branch2b[0][0]             
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 8, 8, 128)    0           bn3a_branch2b[0][0]              
__________________________________________________________________________________________________
res3a_branch2c (Conv2D)         (None, 8, 8, 512)    66048       activation_12[0][0]              
__________________________________________________________________________________________________
res3a_branch1 (Conv2D)          (None, 8, 8, 512)    131584      activation_10[0][0]              
__________________________________________________________________________________________________
bn3a_branch2c (BatchNormalizati (None, 8, 8, 512)    2048        res3a_branch2c[0][0]             
__________________________________________________________________________________________________
bn3a_branch1 (BatchNormalizatio (None, 8, 8, 512)    2048        res3a_branch1[0][0]              
__________________________________________________________________________________________________
add_4 (Add)                     (None, 8, 8, 512)    0           bn3a_branch2c[0][0]              
                                                                 bn3a_branch1[0][0]               
__________________________________________________________________________________________________
activation_13 (Activation)      (None, 8, 8, 512)    0           add_4[0][0]                      
__________________________________________________________________________________________________
res3b_branch2a (Conv2D)         (None, 8, 8, 128)    65664       activation_13[0][0]              
__________________________________________________________________________________________________
bn3b_branch2a (BatchNormalizati (None, 8, 8, 128)    512         res3b_branch2a[0][0]             
__________________________________________________________________________________________________
activation_14 (Activation)      (None, 8, 8, 128)    0           bn3b_branch2a[0][0]              
__________________________________________________________________________________________________
res3b_branch2b (Conv2D)         (None, 8, 8, 128)    147584      activation_14[0][0]              
__________________________________________________________________________________________________
bn3b_branch2b (BatchNormalizati (None, 8, 8, 128)    512         res3b_branch2b[0][0]             
__________________________________________________________________________________________________
activation_15 (Activation)      (None, 8, 8, 128)    0           bn3b_branch2b[0][0]              
__________________________________________________________________________________________________
res3b_branch2c (Conv2D)         (None, 8, 8, 512)    66048       activation_15[0][0]              
__________________________________________________________________________________________________
bn3b_branch2c (BatchNormalizati (None, 8, 8, 512)    2048        res3b_branch2c[0][0]             
__________________________________________________________________________________________________
add_5 (Add)                     (None, 8, 8, 512)    0           bn3b_branch2c[0][0]              
                                                                 activation_13[0][0]              
__________________________________________________________________________________________________
activation_16 (Activation)      (None, 8, 8, 512)    0           add_5[0][0]                      
__________________________________________________________________________________________________
res3c_branch2a (Conv2D)         (None, 8, 8, 128)    65664       activation_16[0][0]              
__________________________________________________________________________________________________
bn3c_branch2a (BatchNormalizati (None, 8, 8, 128)    512         res3c_branch2a[0][0]             
__________________________________________________________________________________________________
activation_17 (Activation)      (None, 8, 8, 128)    0           bn3c_branch2a[0][0]              
__________________________________________________________________________________________________
res3c_branch2b (Conv2D)         (None, 8, 8, 128)    147584      activation_17[0][0]              
__________________________________________________________________________________________________
bn3c_branch2b (BatchNormalizati (None, 8, 8, 128)    512         res3c_branch2b[0][0]             
__________________________________________________________________________________________________
activation_18 (Activation)      (None, 8, 8, 128)    0           bn3c_branch2b[0][0]              
__________________________________________________________________________________________________
res3c_branch2c (Conv2D)         (None, 8, 8, 512)    66048       activation_18[0][0]              
__________________________________________________________________________________________________
bn3c_branch2c (BatchNormalizati (None, 8, 8, 512)    2048        res3c_branch2c[0][0]             
__________________________________________________________________________________________________
add_6 (Add)                     (None, 8, 8, 512)    0           bn3c_branch2c[0][0]              
                                                                 activation_16[0][0]              
__________________________________________________________________________________________________
activation_19 (Activation)      (None, 8, 8, 512)    0           add_6[0][0]                      
__________________________________________________________________________________________________
res3d_branch2a (Conv2D)         (None, 8, 8, 128)    65664       activation_19[0][0]              
__________________________________________________________________________________________________
bn3d_branch2a (BatchNormalizati (None, 8, 8, 128)    512         res3d_branch2a[0][0]             
__________________________________________________________________________________________________
activation_20 (Activation)      (None, 8, 8, 128)    0           bn3d_branch2a[0][0]              
__________________________________________________________________________________________________
res3d_branch2b (Conv2D)         (None, 8, 8, 128)    147584      activation_20[0][0]              
__________________________________________________________________________________________________
bn3d_branch2b (BatchNormalizati (None, 8, 8, 128)    512         res3d_branch2b[0][0]             
__________________________________________________________________________________________________
activation_21 (Activation)      (None, 8, 8, 128)    0           bn3d_branch2b[0][0]              
__________________________________________________________________________________________________
res3d_branch2c (Conv2D)         (None, 8, 8, 512)    66048       activation_21[0][0]              
__________________________________________________________________________________________________
bn3d_branch2c (BatchNormalizati (None, 8, 8, 512)    2048        res3d_branch2c[0][0]             
__________________________________________________________________________________________________
add_7 (Add)                     (None, 8, 8, 512)    0           bn3d_branch2c[0][0]              
                                                                 activation_19[0][0]              
__________________________________________________________________________________________________
activation_22 (Activation)      (None, 8, 8, 512)    0           add_7[0][0]                      
__________________________________________________________________________________________________
res4a_branch2a (Conv2D)         (None, 4, 4, 256)    131328      activation_22[0][0]              
__________________________________________________________________________________________________
bn4a_branch2a (BatchNormalizati (None, 4, 4, 256)    1024        res4a_branch2a[0][0]             
__________________________________________________________________________________________________
activation_23 (Activation)      (None, 4, 4, 256)    0           bn4a_branch2a[0][0]              
__________________________________________________________________________________________________
res4a_branch2b (Conv2D)         (None, 4, 4, 256)    590080      activation_23[0][0]              
__________________________________________________________________________________________________
bn4a_branch2b (BatchNormalizati (None, 4, 4, 256)    1024        res4a_branch2b[0][0]             
__________________________________________________________________________________________________
activation_24 (Activation)      (None, 4, 4, 256)    0           bn4a_branch2b[0][0]              
__________________________________________________________________________________________________
res4a_branch2c (Conv2D)         (None, 4, 4, 1024)   263168      activation_24[0][0]              
__________________________________________________________________________________________________
res4a_branch1 (Conv2D)          (None, 4, 4, 1024)   525312      activation_22[0][0]              
__________________________________________________________________________________________________
bn4a_branch2c (BatchNormalizati (None, 4, 4, 1024)   4096        res4a_branch2c[0][0]             
__________________________________________________________________________________________________
bn4a_branch1 (BatchNormalizatio (None, 4, 4, 1024)   4096        res4a_branch1[0][0]              
__________________________________________________________________________________________________
add_8 (Add)                     (None, 4, 4, 1024)   0           bn4a_branch2c[0][0]              
                                                                 bn4a_branch1[0][0]               
__________________________________________________________________________________________________
activation_25 (Activation)      (None, 4, 4, 1024)   0           add_8[0][0]                      
__________________________________________________________________________________________________
res4b_branch2a (Conv2D)         (None, 4, 4, 256)    262400      activation_25[0][0]              
__________________________________________________________________________________________________
bn4b_branch2a (BatchNormalizati (None, 4, 4, 256)    1024        res4b_branch2a[0][0]             
__________________________________________________________________________________________________
activation_26 (Activation)      (None, 4, 4, 256)    0           bn4b_branch2a[0][0]              
__________________________________________________________________________________________________
res4b_branch2b (Conv2D)         (None, 4, 4, 256)    590080      activation_26[0][0]              
__________________________________________________________________________________________________
bn4b_branch2b (BatchNormalizati (None, 4, 4, 256)    1024        res4b_branch2b[0][0]             
__________________________________________________________________________________________________
activation_27 (Activation)      (None, 4, 4, 256)    0           bn4b_branch2b[0][0]              
__________________________________________________________________________________________________
res4b_branch2c (Conv2D)         (None, 4, 4, 1024)   263168      activation_27[0][0]              
__________________________________________________________________________________________________
bn4b_branch2c (BatchNormalizati (None, 4, 4, 1024)   4096        res4b_branch2c[0][0]             
__________________________________________________________________________________________________
add_9 (Add)                     (None, 4, 4, 1024)   0           bn4b_branch2c[0][0]              
                                                                 activation_25[0][0]              
__________________________________________________________________________________________________
activation_28 (Activation)      (None, 4, 4, 1024)   0           add_9[0][0]                      
__________________________________________________________________________________________________
res4c_branch2a (Conv2D)         (None, 4, 4, 256)    262400      activation_28[0][0]              
__________________________________________________________________________________________________
bn4c_branch2a (BatchNormalizati (None, 4, 4, 256)    1024        res4c_branch2a[0][0]             
__________________________________________________________________________________________________
activation_29 (Activation)      (None, 4, 4, 256)    0           bn4c_branch2a[0][0]              
__________________________________________________________________________________________________
res4c_branch2b (Conv2D)         (None, 4, 4, 256)    590080      activation_29[0][0]              
__________________________________________________________________________________________________
bn4c_branch2b (BatchNormalizati (None, 4, 4, 256)    1024        res4c_branch2b[0][0]             
__________________________________________________________________________________________________
activation_30 (Activation)      (None, 4, 4, 256)    0           bn4c_branch2b[0][0]              
__________________________________________________________________________________________________
res4c_branch2c (Conv2D)         (None, 4, 4, 1024)   263168      activation_30[0][0]              
__________________________________________________________________________________________________
bn4c_branch2c (BatchNormalizati (None, 4, 4, 1024)   4096        res4c_branch2c[0][0]             
__________________________________________________________________________________________________
add_10 (Add)                    (None, 4, 4, 1024)   0           bn4c_branch2c[0][0]              
                                                                 activation_28[0][0]              
__________________________________________________________________________________________________
activation_31 (Activation)      (None, 4, 4, 1024)   0           add_10[0][0]                     
__________________________________________________________________________________________________
res4d_branch2a (Conv2D)         (None, 4, 4, 256)    262400      activation_31[0][0]              
__________________________________________________________________________________________________
bn4d_branch2a (BatchNormalizati (None, 4, 4, 256)    1024        res4d_branch2a[0][0]             
__________________________________________________________________________________________________
activation_32 (Activation)      (None, 4, 4, 256)    0           bn4d_branch2a[0][0]              
__________________________________________________________________________________________________
res4d_branch2b (Conv2D)         (None, 4, 4, 256)    590080      activation_32[0][0]              
__________________________________________________________________________________________________
bn4d_branch2b (BatchNormalizati (None, 4, 4, 256)    1024        res4d_branch2b[0][0]             
__________________________________________________________________________________________________
activation_33 (Activation)      (None, 4, 4, 256)    0           bn4d_branch2b[0][0]              
__________________________________________________________________________________________________
res4d_branch2c (Conv2D)         (None, 4, 4, 1024)   263168      activation_33[0][0]              
__________________________________________________________________________________________________
bn4d_branch2c (BatchNormalizati (None, 4, 4, 1024)   4096        res4d_branch2c[0][0]             
__________________________________________________________________________________________________
add_11 (Add)                    (None, 4, 4, 1024)   0           bn4d_branch2c[0][0]              
                                                                 activation_31[0][0]              
__________________________________________________________________________________________________
activation_34 (Activation)      (None, 4, 4, 1024)   0           add_11[0][0]                     
__________________________________________________________________________________________________
res4e_branch2a (Conv2D)         (None, 4, 4, 256)    262400      activation_34[0][0]              
__________________________________________________________________________________________________
bn4e_branch2a (BatchNormalizati (None, 4, 4, 256)    1024        res4e_branch2a[0][0]             
__________________________________________________________________________________________________
activation_35 (Activation)      (None, 4, 4, 256)    0           bn4e_branch2a[0][0]              
__________________________________________________________________________________________________
res4e_branch2b (Conv2D)         (None, 4, 4, 256)    590080      activation_35[0][0]              
__________________________________________________________________________________________________
bn4e_branch2b (BatchNormalizati (None, 4, 4, 256)    1024        res4e_branch2b[0][0]             
__________________________________________________________________________________________________
activation_36 (Activation)      (None, 4, 4, 256)    0           bn4e_branch2b[0][0]              
__________________________________________________________________________________________________
res4e_branch2c (Conv2D)         (None, 4, 4, 1024)   263168      activation_36[0][0]              
__________________________________________________________________________________________________
bn4e_branch2c (BatchNormalizati (None, 4, 4, 1024)   4096        res4e_branch2c[0][0]             
__________________________________________________________________________________________________
add_12 (Add)                    (None, 4, 4, 1024)   0           bn4e_branch2c[0][0]              
                                                                 activation_34[0][0]              
__________________________________________________________________________________________________
activation_37 (Activation)      (None, 4, 4, 1024)   0           add_12[0][0]                     
__________________________________________________________________________________________________
res4f_branch2a (Conv2D)         (None, 4, 4, 256)    262400      activation_37[0][0]              
__________________________________________________________________________________________________
bn4f_branch2a (BatchNormalizati (None, 4, 4, 256)    1024        res4f_branch2a[0][0]             
__________________________________________________________________________________________________
activation_38 (Activation)      (None, 4, 4, 256)    0           bn4f_branch2a[0][0]              
__________________________________________________________________________________________________
res4f_branch2b (Conv2D)         (None, 4, 4, 256)    590080      activation_38[0][0]              
__________________________________________________________________________________________________
bn4f_branch2b (BatchNormalizati (None, 4, 4, 256)    1024        res4f_branch2b[0][0]             
__________________________________________________________________________________________________
activation_39 (Activation)      (None, 4, 4, 256)    0           bn4f_branch2b[0][0]              
__________________________________________________________________________________________________
res4f_branch2c (Conv2D)         (None, 4, 4, 1024)   263168      activation_39[0][0]              
__________________________________________________________________________________________________
bn4f_branch2c (BatchNormalizati (None, 4, 4, 1024)   4096        res4f_branch2c[0][0]             
__________________________________________________________________________________________________
add_13 (Add)                    (None, 4, 4, 1024)   0           bn4f_branch2c[0][0]              
                                                                 activation_37[0][0]              
__________________________________________________________________________________________________
activation_40 (Activation)      (None, 4, 4, 1024)   0           add_13[0][0]                     
__________________________________________________________________________________________________
res5a_branch2a (Conv2D)         (None, 2, 2, 512)    524800      activation_40[0][0]              
__________________________________________________________________________________________________
bn5a_branch2a (BatchNormalizati (None, 2, 2, 512)    2048        res5a_branch2a[0][0]             
__________________________________________________________________________________________________
activation_41 (Activation)      (None, 2, 2, 512)    0           bn5a_branch2a[0][0]              
__________________________________________________________________________________________________
res5a_branch2b (Conv2D)         (None, 2, 2, 512)    2359808     activation_41[0][0]              
__________________________________________________________________________________________________
bn5a_branch2b (BatchNormalizati (None, 2, 2, 512)    2048        res5a_branch2b[0][0]             
__________________________________________________________________________________________________
activation_42 (Activation)      (None, 2, 2, 512)    0           bn5a_branch2b[0][0]              
__________________________________________________________________________________________________
res5a_branch2c (Conv2D)         (None, 2, 2, 2048)   1050624     activation_42[0][0]              
__________________________________________________________________________________________________
res5a_branch1 (Conv2D)          (None, 2, 2, 2048)   2099200     activation_40[0][0]              
__________________________________________________________________________________________________
bn5a_branch2c (BatchNormalizati (None, 2, 2, 2048)   8192        res5a_branch2c[0][0]             
__________________________________________________________________________________________________
bn5a_branch1 (BatchNormalizatio (None, 2, 2, 2048)   8192        res5a_branch1[0][0]              
__________________________________________________________________________________________________
add_14 (Add)                    (None, 2, 2, 2048)   0           bn5a_branch2c[0][0]              
                                                                 bn5a_branch1[0][0]               
__________________________________________________________________________________________________
activation_43 (Activation)      (None, 2, 2, 2048)   0           add_14[0][0]                     
__________________________________________________________________________________________________
res5b_branch2a (Conv2D)         (None, 2, 2, 512)    1049088     activation_43[0][0]              
__________________________________________________________________________________________________
bn5b_branch2a (BatchNormalizati (None, 2, 2, 512)    2048        res5b_branch2a[0][0]             
__________________________________________________________________________________________________
activation_44 (Activation)      (None, 2, 2, 512)    0           bn5b_branch2a[0][0]              
__________________________________________________________________________________________________
res5b_branch2b (Conv2D)         (None, 2, 2, 512)    2359808     activation_44[0][0]              
__________________________________________________________________________________________________
bn5b_branch2b (BatchNormalizati (None, 2, 2, 512)    2048        res5b_branch2b[0][0]             
__________________________________________________________________________________________________
activation_45 (Activation)      (None, 2, 2, 512)    0           bn5b_branch2b[0][0]              
__________________________________________________________________________________________________
res5b_branch2c (Conv2D)         (None, 2, 2, 2048)   1050624     activation_45[0][0]              
__________________________________________________________________________________________________
bn5b_branch2c (BatchNormalizati (None, 2, 2, 2048)   8192        res5b_branch2c[0][0]             
__________________________________________________________________________________________________
add_15 (Add)                    (None, 2, 2, 2048)   0           bn5b_branch2c[0][0]              
                                                                 activation_43[0][0]              
__________________________________________________________________________________________________
activation_46 (Activation)      (None, 2, 2, 2048)   0           add_15[0][0]                     
__________________________________________________________________________________________________
res5c_branch2a (Conv2D)         (None, 2, 2, 512)    1049088     activation_46[0][0]              
__________________________________________________________________________________________________
bn5c_branch2a (BatchNormalizati (None, 2, 2, 512)    2048        res5c_branch2a[0][0]             
__________________________________________________________________________________________________
activation_47 (Activation)      (None, 2, 2, 512)    0           bn5c_branch2a[0][0]              
__________________________________________________________________________________________________
res5c_branch2b (Conv2D)         (None, 2, 2, 512)    2359808     activation_47[0][0]              
__________________________________________________________________________________________________
bn5c_branch2b (BatchNormalizati (None, 2, 2, 512)    2048        res5c_branch2b[0][0]             
__________________________________________________________________________________________________
activation_48 (Activation)      (None, 2, 2, 512)    0           bn5c_branch2b[0][0]              
__________________________________________________________________________________________________
res5c_branch2c (Conv2D)         (None, 2, 2, 2048)   1050624     activation_48[0][0]              
__________________________________________________________________________________________________
bn5c_branch2c (BatchNormalizati (None, 2, 2, 2048)   8192        res5c_branch2c[0][0]             
__________________________________________________________________________________________________
add_16 (Add)                    (None, 2, 2, 2048)   0           bn5c_branch2c[0][0]              
                                                                 activation_46[0][0]              
__________________________________________________________________________________________________
activation_49 (Activation)      (None, 2, 2, 2048)   0           add_16[0][0]                     
__________________________________________________________________________________________________
average_pooling2d_1 (AveragePoo (None, 1, 1, 2048)   0           activation_49[0][0]              
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 2048)         0           average_pooling2d_1[0][0]        
__________________________________________________________________________________________________
fc6 (Dense)                     (None, 6)            12294       flatten_1[0][0]                  
==================================================================================================
Total params: 23,600,006
Trainable params: 23,546,886
Non-trainable params: 53,120
__________________________________________________________________________________________________

```