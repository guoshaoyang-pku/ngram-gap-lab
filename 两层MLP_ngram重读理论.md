# 两层 MLP 中 n-gram 偏置、数据重读与 train–test gap

本文只保留一个最小模型。真正需要跟踪的量只有：

- $\delta b_h$：上下文 $h$ 对应的 n-gram 有效偏置；
- $\xi_h$：该上下文的有限样本经验误差；
- $m$：这个上下文被读取并更新的次数。

其余符号都视为固定参数或已知量。为避免符号过多，本文把 $h$ 固定为一个具体上下文；因此真正随训练变化的只有 $\delta b_h$，而 $\xi_h$ 是固定训练集决定的目标，$m$ 是读取次数。$x$、$p_x$、$A$、$b$、$v_h$、$L_0$、$k$ 和 $\eta$ 都只作为固定量出现；$r$ 与 $n_h$ 只用于把实验中的 epoch 换算成 $m$。

---

## 1. 一句话结论

均匀重读不会改变训练数据的平均目标，但会让同一个 n-gram 上下文被重复更新。于是它对应的有效偏置 $\delta b_h$ 越来越接近有限样本经验误差 $\xi_h$：

$$
\text{重复读取}
\Longrightarrow
\delta b_h\text{ 拟合有限样本误差 }\xi_h
\Longrightarrow
L_{\mathrm{train}}\text{ 下降}
\quad\text{而}\quad
L_{\mathrm{test}}\text{ 不同步下降}.
$$

因此 train–test gap 会增加，但在固定学习率的最简 GD 模型中会逐渐饱和。

---

## 2. 数据：一阶 Markov toy data

设当前输入 token 为 $x$。你的 toy data 是一阶 Markov 数据，因此真实的下一个 token 分布只由 $x$ 决定，记为 $p_x$。更早的历史在真实分布中不再提供额外信息。

现在固定一个具体的 n-gram 上下文 $h$。它的最后一个 token 也是 $x$，所以真实分布仍然是 $p_x$。

但在有限训练集里，$h$ 只出现了有限次。它的经验分布通常不是恰好 $p_x$，而是

$$
p_x+\xi_h.
$$

这里 $\xi_h$ 的定义是“有限样本经验误差”：

$$
\xi_h
=
\text{上下文 }h\text{ 在训练集中的经验分布}
-
\text{真实分布 }p_x.
$$

$\xi_h$ 是由有限数据的随机抽样产生的固定量，不是额外注入的噪声，也不是模型参数：

- 换一份训练集，$\xi_h$ 会改变；
- 固定同一份训练集后，重复读取时看到的是同一个 $\xi_h$；
- $h$ 出现次数越少，$\xi_h$ 通常越大；例如上面的 4 次观察产生了 $\pm1/4$ 的误差，而观察次数更多时，经验比例通常会更接近真实比例。

独立 test 数据仍来自真实分布 $p_x$，不会重复出现训练集中特定的 $\xi_h$。有限 test 集也会有自己的抽样误差，但它与训练集的 $\xi_h$ 独立；本文为简化推导，使用独立 test 分布的期望，即直接使用 $p_x$。

### 一个具体的小例子

为了只看数字，令词表为 $\{1,2,3,4,5\}$，固定当前 token $x=2$，并取一个具体 bigram 上下文

$$
h=(1,2).
$$

假设真实 Markov 规则规定：从 token $2$ 出发，下一个 token 只可能是 $3$ 或 $5$，且概率相同：

$$
p_x(3)=\frac12,
\qquad
p_x(5)=\frac12.
$$

现在有 4 条很短的训练序列：

$$
[1,2,3,4,5],
\qquad
[1,2,3,3,5],
\qquad
[1,2,5,4,5],
\qquad
[1,2,3,5,4].
$$

每条序列开头都出现同一个上下文 $h=(1,2)$。这 4 次出现后面的 token 依次为 $[3,3,5,3]$，所以训练集经验分布为

$$
\widehat p_h(3)=\frac34,
\qquad
\widehat p_h(5)=\frac14.
$$

于是该上下文的有限样本经验误差为

$$
\xi_h(3)
=\widehat p_h(3)-p_x(3)
=\frac34-\frac12
=\frac14,
$$

$$
\xi_h(5)
=\widehat p_h(5)-p_x(5)
=\frac14-\frac12
=-\frac14.
$$

因此可以把这个例子写成向量：

$$
\xi_h
=\left(\frac14,-\frac14\right)
$$

（坐标顺序为目标 $3,5$）。它的含义不是“数据被加了一个 $+1/4$ 或 $-1/4$ 的外部噪声”，而是：在这个有限训练集里，$h$ 后面看到 $3$ 的比例比真实概率高了 $1/4$，看到 $5$ 的比例比真实概率低了 $1/4$。

如果把同一个训练集完整重读，4 条序列和后继列表 $[3,3,5,3]$ 都不变，所以每一轮看到的仍是同一个 $\xi_h$。独立 test 样本的期望仍由 $(1/2,1/2)$ 给出，而不会自动变成训练集中的 $(3/4,1/4)$。

---

## 3. n-gram 为什么可以看成偏置

两层 MLP 写成

$$
z=W_2\phi(W_1x+b_1)+b_2.
$$

取恒等激活 $\phi(x)=x$，得到

$$
z=W_2W_1x+W_2b_1+b_2.
$$

记

$$
A=W_2W_1,
\qquad
b=W_2b_1+b_2,
$$

于是普通 MLP 为

$$
z=Ax+b.
$$

在当前代码的 `input` 注入方式中，n-gram 表给上下文 $h$ 一个向量。记这个向量为 $v_h$，则 MLP 实际接收的输入是

$$
x+v_h.
$$

输出变成

$$
\begin{aligned}
z_h
&=A(x+v_h)+b\\
&=Ax+b+Av_h.
\end{aligned}
$$

因此定义

$$
\boxed{
\delta b_h:=Av_h
}
$$

就可以写成

$$
\boxed{
z_h=Ax+b+\delta b_h.
}
$$

这就是“n-gram 偏置”：

- $b$ 是所有样本共享的普通偏置；
- $\delta b_h$ 只在上下文 $h$ 出现时生效；
- 不同上下文有不同的 $\delta b_h$，所以 n-gram 表提供了逐上下文的记忆自由度。

为了得到最简单的公式，下面暂时把 $A$、$b$ 和 n-gram 向量到输出的增益都视为固定。只研究 $\delta b_h$ 如何被 GD 写入。

---

## 4. 用一个二次损失描述“拟合有限样本经验误差”

只分析输出 logits 中一个有效方向，并把该方向上的 n-gram 偏置仍记作 $\delta b_h$，把相应的有限样本经验误差仍记作 $\xi_h$。

在 MLP 已经学会主要 Markov 规律的附近，训练损失可以用最简单的二次式近似。把与 $\delta b_h$ 无关的常数吸收到 $L_0$ 中，写成：

$$
L_{\mathrm{train}}(\delta b_h)
=L_0-k\xi_h\delta b_h+\frac{k}{2}(\delta b_h)^2.
$$

其中：

- $L_0$ 是与这个 n-gram 记忆无关的基准损失；
- $k>0$ 是固定曲率参数；
- $\xi_h$ 是训练集希望该上下文偏置拟合的目标。

这个式子表达的意思很直接：训练集会奖励 $\delta b_h$ 接近 $\xi_h$。它等价于

$$
L_{\mathrm{train}}(\delta b_h)
=L_0+\frac{k}{2}(\delta b_h-\xi_h)^2-\frac{k}{2}\xi_h^2.
$$

最后一项与 $\delta b_h$ 无关，只是为了让 $\delta b_h=0$ 时的训练损失取为 $L_0$。

对独立 test 数据，训练集特有的经验误差不存在，因此对应的二次近似是

$$
L_{\mathrm{test}}(\delta b_h)
=L_0+\frac{k}{2}(\delta b_h)^2.
$$

test loss 希望 n-gram 偏置保持在零，因为真实 Markov 规律已经由共享的 $Ax+b$ 表示；额外的上下文偏置只是在拟合有限训练样本造成的经验误差。

因此，在 $\delta b_h=0$ 时，train 和 test 都从同一个基准 $L_0$ 开始；两者的区别只来自有限样本经验误差项 $-k\xi_h\delta b_h$。两式的最低点分别是：

$$
\begin{aligned}
L_{\mathrm{train}}&\text{ 的最低点在 }\delta b_h=\xi_h,\\
L_{\mathrm{test}}&\text{ 的最低点在 }\delta b_h=0.
\end{aligned}
$$

---

## 5. GD 更新公式

对训练损失求导：

$$
\frac{\partial L_{\mathrm{train}}}{\partial\delta b_h}
=k(\delta b_h-\xi_h).
$$

用学习率 $\eta$ 的 GD 更新：

$$
\boxed{
\delta b_h^{(m+1)}
=\delta b_h^{(m)}
-\eta k\left(\delta b_h^{(m)}-\xi_h\right)
}
$$

整理为

$$
\delta b_h^{(m+1)}
=
(1-\eta k)\delta b_h^{(m)}
+\eta k\xi_h.
$$

这里的 $m$ 只表示读取次数：

- $m=0$：还没有读取这个上下文；
- $m=1$：读到一次并更新一次；
- $m=2$：读到两次并更新两次；
- 以此类推。

如果从未写入的表开始，取

$$
\delta b_h^{(0)}=0.
$$

---

## 6. 读取 $m$ 次后的公式

前几步为

$$
\delta b_h^{(1)}=\eta k\xi_h,
$$

$$
\delta b_h^{(2)}
=(1-\eta k)\eta k\xi_h+\eta k\xi_h.
$$

继续展开，得到等比数列：

$$
\delta b_h^{(m)}
=\eta k\xi_h
\sum_{s=0}^{m-1}(1-\eta k)^s.
$$

使用等比数列求和公式（$s$ 只是求和用的临时下标，不是新的动态变量）：

$$
\sum_{s=0}^{m-1}(1-\eta k)^s
=
\frac{1-(1-\eta k)^m}{\eta k},
$$

得到

$$
\boxed{
\delta b_h^{(m)}
=
\left[1-(1-\eta k)^m\right]\xi_h.
}
$$

这就是重复读取公式。

当

$$
0<\eta k<1
$$

时，模型每次只向 $\xi_h$ 移动一部分，因此

$$
\delta b_h^{(m)}\longrightarrow\xi_h
\qquad(m\longrightarrow\infty).
$$

所以重复读取的作用不是把偏置直接乘上重复次数，而是让它逐步逼近有限样本经验误差。

---

## 7. gap 随重复次数的趋势

把读取 $m$ 次后的偏置代回两种损失。由于 $\delta b_h^{(0)}=0$，相对于 $m=0$，train loss 的变化为

$$
\Delta L_{\mathrm{train}}^{(m)}
=
-k\left[1-(1-\eta k)^m\right]\xi_h^2
+\frac{k}{2}\left[1-(1-\eta k)^m\right]^2\xi_h^2.
$$

test loss 的变化为

$$
\Delta L_{\mathrm{test}}^{(m)}
=
\frac{k}{2}\left[1-(1-\eta k)^m\right]^2\xi_h^2.
$$

定义 gap 为

$$
G=L_{\mathrm{test}}-L_{\mathrm{train}}.
$$

因此，gap 相对于初始 gap 的增加量为

$$
\boxed{
\Delta G^{(m)}
=
k\left[1-(1-\eta k)^m\right]\xi_h^2.
}
$$

在 $0<\eta k<1$ 时，

$$
\Delta G^{(m+1)}-\Delta G^{(m)}
=
k\eta k(1-\eta k)^m\xi_h^2>0.
$$

所以：

1. gap 随读取次数增加；
2. 每次新增读取带来的 gap 增量越来越小；
3. gap 最终趋于有限平台：

$$
\Delta G^{(m)}\longrightarrow k\xi_h^2.
$$

最简 GD 理论预测的是“单调增大、逐渐饱和”，不是无限增大。

---

## 8. 为什么 train loss 下降而 test loss 不一定下降

从上面的两个损失可以直接看出：

$$
L_{\mathrm{train}}\text{ 希望 }\delta b_h\to\xi_h,
$$

而

$$
L_{\mathrm{test}}\text{ 希望 }\delta b_h\to0.
$$

因此当 GD 让 $\delta b_h$ 从 $0$ 向 $\xi_h$ 移动时：

- train loss 先明显下降，因为模型开始记住训练集中的 $\xi_h$；
- test loss 的额外项 $k(\delta b_h)^2/2$ 上升，因为 test 中没有同一个有限样本经验误差；
- 两者之差，也就是 gap，增大。

若 $\xi_h=0$，则

$$
\delta b_h^{(m)}=0,
\qquad
\Delta G^{(m)}=0.
$$

这说明没有有限样本经验误差时，单纯重读不会产生这种 n-gram 过拟合 gap。

---

## 9. 重读如何对应到 $m$

若一个 n-gram 上下文 $h$ 在唯一训练数据中出现 $n_h$ 次，每完整读取一遍数据，它就平均被更新 $n_h$ 次。

如果完整数据被读取 $r$ 遍，则近似有

$$
m=rn_h.
$$

这里 $n_h$ 和 $r$ 只是把实验设置换算成读取次数的参数；核心动态变量仍然只有 $\delta b_h$、$\xi_h$ 和 $m$。

均匀重读时，训练集中的经验分布没有改变：复制 $r$ 次只会同时把所有计数乘以 $r$，归一化后的平均目标不变。它改变的是同一个 $\xi_h$ 被 GD 看到的次数 $m$。

---

## 10. 与实际代码的对应

当前 `code/train_mlp.py` 的 `input` 注入可以用下面的最简关系表示：

$$
x\longmapsto x+v_h
\longmapsto
Ax+b+Av_h
=Ax+b+\delta b_h.
$$

实际代码中 n-gram 表的每一行就是一个可学习的 $v_h$；同一个上下文再次出现时，会重新取出同一行，并通过当前 loss 产生梯度。

本文为了得到闭式公式，使用普通 GD，并把 $A$ 固定。实际代码使用 RMSProp 更新 n-gram 表、AdamW 更新 MLP，因此具体曲线不一定严格等于上面的幂律形式，但“同一上下文反复出现、同一有限样本经验误差反复写入、gap 随 replay 增长”的机制不变。

---

## 11. 最终总结

对一个固定 n-gram 上下文 $h$：

1. 有限训练集产生固定的经验误差 $\xi_h$；
2. n-gram 表通过输入向量 $v_h$ 产生有效偏置 $\delta b_h=Av_h$；
3. train loss 推动 $\delta b_h$ 拟合 $\xi_h$；
4. test loss 不包含这个训练集特有的 $\xi_h$；
5. GD 每读取一次就更新一次 $\delta b_h$；
6. 读取 $m$ 次后：

$$
\boxed{
\delta b_h^{(m)}
=
\left[1-(1-\eta k)^m\right]\xi_h
}
$$

7. 因而 gap 增加量为

$$
\boxed{
\Delta G^{(m)}
=
k\left[1-(1-\eta k)^m\right]\xi_h^2
}
$$

在固定 GD 参数下，gap 单调增加但逐渐饱和。低频上下文的 $\xi_h$ 通常更大，因此它们对总 gap 的贡献也更大。
