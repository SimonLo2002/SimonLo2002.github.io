---
title: thought_math
date: 2025-05-31 14:00:00 +0800
categories: [BLOG, DL]
tags: [math]     # TAG names should always be lowercase
math: true
mermaid: true
---

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
def numerical_gradient(f, x):#f(x)接受二维数组作为输入
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
