---
title: thought_DL
date: 2025-05-31 14:00:00 +0800
categories: [BLOG, DL]
tags: [DL]     # TAG names should always be lowercase
math: true
mermaid: true
---

# 激活函数

激活函数会将输入信号的总和，例如$a = w1\cdot{x1} + w2\cdot{x2} + b$，转换为输出信号。激活函数必须使用非线性函数，如阶跃函数，sigmoid函数或者ReLu函数，深度学习的核心是==多层嵌套的非线性变换==。每一层的神经元通过非线性激活函数（如 ReLU、Sigmoid 等）对输入进行非线性映射，再通过多层叠加形成高度复杂的复合函数。

## 常见的激活函数

sigmoid 函数是一种常见的激活函数，其数学表达式为：$sigmoid(x) = \dfrac{1}{1+e^{-x}}$。从函数形式来看，它能够将任意实数输入映射到 (0,1) 区间内，当输入值趋近于正无穷时，函数值趋近于 1；当输入值趋近于负无穷时，函数值趋近于 0，其函数曲线呈现出光滑的 S 形。

sigmoid函数可以引入非线性变换，使网络具备学习复杂非线性关系的能力。不过，sigmoid 函数存在一些缺点，比如当输入值的绝对值较大时，函数的导数趋近于 0，容易导致梯度消失问题，影响网络的训练效率。

```python
#sigmoid函数
def sigmoid(x):
    return 1/(1 + np.exp(-x))
```

在ReLU函数中, 当输入 *x*≥0 时，输出 *x*；当 *x*<0 时，输出 0。

```python
#ReLU函数
def ReLU(x):
    return max(x,0)
```

ReLU函数解决了梯度消失问题，Sigmoid在输入绝对值较大时导数趋近于 0（梯度消失），而 ReLU 在 *x*>0 时导数恒为 1，梯度可稳定传播。同时，ReLU运算仅需判断符号和取最大值，无需指数运算（对比 Sigmoid 的 *e*−*x* 计算），更适合大规模网络。

```python
#softmax函数
def softmax_imporved(x):
    x1 = x - np.max(x)
    return np.exp(x1)/np.exp(x1).sum()
```

softmax 函数是深度学习中常用的激活函数，主要用于多分类问题，将模型的输出转换为概率分布。在代码实现中，softmax_imporved(x)先通过`x1 = x - np.max(x)`对输入进行平移，避免计算指数时因数值过大导致溢出（例如，若 *x* 中元素很大，`e**x`可能超出计算机数值范围。最终输出的每个元素值在 (0,1) 之间，且所有元素之和为 1，符合概率分布的性质。

# 学习算法的实现

通过配置合适的神经网络参数数量和层数，通过梯度下降等优化方法，逐渐找到最适合的权重和偏置参数，以便逐渐拟合训练数据的过程即称为学习。

 **步骤1：**在训练数据集中选取mini-batch，将其输入神经网络并获得输出，再根据其输出计算损失函数

**步骤2：**根据损失函数计算各个***权重参数的梯度***

 **步骤3：**将权重参数沿着梯度进行微小更新

**步骤4：**反复重复步骤1-3

## mini_batch_study

mini_batch学习 由于许多时候训练数据数据量极为庞大，因此以全部数据为对象计算损失函数是不现实的，因此，我们随即从全部数据中取出一部分，作为全部数据的""近似"，神经网络的学习也是从 训练数据中选出一批数据（称为mini_batch）,然后对每个mini_batch进行学习。这种学习方式称为mini_batch学习

```python
#选取mini-batch
train_size = x_train.shape[0]
batch_size = 100
batch_mask = np.random.choice(train_size,batch_size) #随机选择100个数作为索引
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]#以随机选择的100个索引取出对应的数据和标签
```

使用predict函数计算预测结果y

```python
#如果t不为one-hot表示的mini_batch版交叉熵误差计算
def cross_entropy_error_mb_no(y,t):
    if y.ndim == 1: 
        y.reshape(1,y.size) 
        t.reshape(1,t.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size),t] + 1e-7))/batch_size#计算平均交叉熵误差
#使用one-hot表示的标签用于计算交叉熵误差也就是计算对应正确解的神经网络输出结果的ln值
#因此使用y[np.arange(batch_size),t](NumPy中,可以使用数组作为索引,这被称为高级索引)
#取出神经网络输出y中对应正确解t的结果计算即可
```



# Gradient Calculation

## 1d_params_matrix

```python
def numerical_gradient_1d(f,x): #当x为一维数组时
    h = 1e-4
    grad = np.zeros_like(x) #生成和x形状相同的数组

    for idx in range(x.size):
        temp_val = x[idx] #将数组的真实值暂时储存在一个变量内
        
        #计算f(x + h)
        x[idx] = temp_val + h #利用暂存的真实值更改x[idx]
        fxh1 = f(x) #更改x[idx]后，计算函数值
        
        #计算f(x - h)
        x[idx] = temp_val - h 
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2)/(2*h)
        x[idx] = temp_val #将真实值还原回x数组中
    return grad
```

## 2d_matrix

```python
def numerical_gradient(f, x):#f(x)接受二维数组作为输入,这里的x实际上是权重参数矩阵
    h = 1e-4
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        it.iternext() 
    return grad
```

> np.nditer是NumPy中用于高效多维数组迭代的核心工具，在数值计算场景中常用于遍历数组元素。以下从五个维度解析其用法：
>
> **关键参数解析**
>
> ***flags：控制迭代行为***
>
> multi_index 生成多维索引（用于定位元素位置）
>
> external_loop 合并维度优化迭代
>
> c_index/f_index 指定C或Fortran内存顺序
>
> ***op_flags：设置操作权限***
>
> readonly (默认)
>
> readwrite 允许修改元素值
>
> writeonly 仅写入模式

![image-20250607184113976](/assets/image-20250607184113976.png)

## a simple two_layers neural network

![image-20250607184008242](/assets/image-20250607184008242.png)

## construct a class of Two_layers_NN

```python
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        #初始化权重params
        self.params = {} #将权重参数保存在字典内
        self.params['W1'] = weight_init_std * \
                            np.random.randn(input_size,hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)
        
    def predict(self,x): #输出预测结果
        W1,W2 = self.params['W1'],self.params['W2']
        b1,b2 = self.params['b1'],self.params['b2']
        a1 = np.dot(x,W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,W2) + b2
        y = softmax_imporved(a2)
        return y
        
    # x:输入数据, t:监督数据
    def loss(self, x, t): #计算损失函数
        y = self.predict(x)
        return cross_entropy_error(y,t)#已经除以batch_size求得平均交叉熵误差,t已转换为正确解表示
        
        
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self,x,t):
        loss_W = lambda W: self.loss(x,t)#计算出平均交叉熵误差
        grads = {}
#使用数值微分的方法计算参数W1的梯度，由于参数W1的形状为(input_size,hidden_size),因此会计算出一个形状为(input_size,hidden_size)的梯度矩阵，
#因而迭代器要迭代input_size * hidden_size次才能计算出W1的梯度，每一次循环还包括计算b1,W2,b2的梯度，因此即使现代cpu算力强大，速度仍然非常慢
        grads['W1'] = numerical_gradient(loss_W,self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W,self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W,self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W,self.params['b2'])
        return grads
```

## implement a two_layers_network

```python
#读取训练集和测试集
train_path = "/Users/luomeng/Desktop/syncspace/train.csv"
test_path = "/Users/luomeng/Desktop/syncspace/test.csv"
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)#设置合适的隐藏层数

x_train_1 = train_data.iloc[:, 1:].values 
x_train = x_train_1/255 #归一化避免数值溢出
t_train_1 = train_data.iloc[:, 0].values
encoder = OneHotEncoder(sparse_output=False)#初始化one-hot编码器
t_train = encoder.fit_transform(t_train_1.reshape(-1, 1))#将标签转换为one-hot表示

x_test_1 = test_data.iloc[:, 1:].values
x_test = x_test_1/255
t_test_1 = test_data.iloc[:, 0].values
t_test = encoder.transform(t_test_1.reshape(-1, 1))   
```

```python
#设置超参数
iters_num = 100  # 适当设定循环的次数
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

iter_per_epoch = max(train_size / batch_size, 1)
#epoch指将整个训练数据集完整地过一遍模型的过程
```

```python
#获取mini_batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #计算梯度
    grad = network.numerical_gradient(x_batch,t_batch)
    
    # 更新参数
    for key in ('W1','b1','W2','b2'):
        network.params[key] -= learning_rate * grad[key]
```





# 优化方法

## Momentum优化法

普通梯度下降法每次更新的方向完全由当前的梯度决定。如果损失函数的曲面是一个狭长的椭圆形，梯度下降法可能会在陡峭方向上来回震荡，导致收敛速度变慢。

动量法通过引入动量 $$v_t$$，在更新时会“记住”之前的更新方向，从而在陡峭方向上减少震荡，在平缓方向上加速移动。

```math
v_t = \mu v_{t-1} - \eta  \frac{\partial L}{\partial W}
```

可以把$\mu$理解为斜坡的摩擦力，在物体离开陡峭梯度时承担使"速度"逐渐减缓的作用（实际上承担"记忆"的作用）

动量法的更新公式可以展开成一个累积的形式：

$$
v_t = -\eta \nabla f(\theta_t) - \eta \mu \nabla f(\theta_{t-1}) - \eta \mu^2 \nabla f(\theta_{t-2}) - \dots
$$

动量法的更新实际上是对**过去的梯度进行加权平均**，权重由动量因子 *μ* 决定。（指数加权平均）

- 当前的梯度$ \nabla f(\theta_t)$权重最大。
- 越早的梯度权重越小（因为被$\mu^k$衰减）。

```python
class Momentum:
    def __init__(self, lr=0.01, momentum = 0.9):
        self.v = None
        self.lr = lr
        self.momentum = momentum
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        
        for key in grads.keys():
            self.v[key] = self.v[key] * self.momentum - self.lr * grads[key]
            params[key] += self.v[key]
```

## AdaGrad

AdaGrad通过对过往梯度的记录和numpy数组运算的特性，使得偏导数大的参数学习率下降更快，偏导数小的参数学习率下降更慢

```python
class AdaGrad:
  def __init__(self,lr = 0.01):
    self.h = None
    self.lr = lr
    
  def update(self,params,grads):
    if self.h is None:
      self.h = {}
      for key,val in params.items():
        self.h[key] = np.zeros_like(val)
		for key in params.keys():
      self.h[key] = self.h[key] + grads[key] * grads[key] 
      grads[key] -= self.lr * grads[key] / np.sqrt(self.h[key] + 1e-7)
      
```

