---
title: thought_DL
date: 2025-05-25 11:00:00 +0800
categories: [BLOG, DL]
tags: [dl]     # TAG names should always be lowercase
math: true
mermaid: true
---

# Momentum优化法

普通梯度下降法每次更新的方向完全由当前的梯度决定。如果损失函数的曲面是一个狭长的椭圆形（如下图所示），梯度下降法可能会在陡峭方向上来回震荡，导致收敛速度变慢。

动量法通过引入动量 $$v_t$$，在更新时会“记住”之前的更新方向，从而在陡峭方向上减少震荡，在平缓方向上加速移动。

动量法的更新公式可以展开成一个累积的形式：

$$
v_t = -\eta \nabla f(\theta_t) - \eta \mu \nabla f(\theta_{t-1}) - \eta \mu^2 \nabla f(\theta_{t-2}) - \dots
$$

动量法的更新实际上是对**过去的梯度进行加权平均**，权重由动量因子 *μ* 决定。（指数加权平均）

- 当前的梯度$$ \nabla f(\theta_t)$$权重最大。
- 越早的梯度权重越小（因为被$$\mu^k$$衰减）。
