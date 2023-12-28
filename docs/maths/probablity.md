# Probablity

# Chapter 1


<hr>

# Chapter 2 & 4

## 2.1. 几种重要分布

> 主要介绍0-1分布、二项分布、泊松分布、均匀分布、指数分布、正态分布。

### 2.1.1 离散型随机变量

#### 2.1.1.1 0-1分布（0-1 distribution）
记作：
$$ X \sim b(1,p) $$

概率质量函数：(n次试验中正好得到k次成功的概率/分布律)

$$ P\{x=k\} = p^k(1-p)^{n-k} \quad (k=0,1) $$

<br>

#### 2.1.1.2 二项分布（Binomial distribution）
**Intro: 伯努利试验（Bernoulli trial）**
&emsp;&emsp; 只有两种可能结果（“成功”或“失败”）的单次随机试验，即对于一个随机变量X而言，$ Pr[X=1]=p, \quad Pr[X=0]=1-p $
&emsp;&emsp; 一个伯努利过程（Bernoulli process, n重伯努利试验）是由重复出现独立但是相同分布的伯努利试验组成，例如抛硬币十次，而此时呈现之结果将呈现二项分布。

记作：
$$ X \sim b(N,p) $$

概率质量函数：

$$ P\{x=k\} = C_n^kp^k(1-p)^{n-k} $$

<br>

#### 2.1.1.3 泊松分布（Poisson distribution）

期望 $E(X) = \lambda$

记作：

$$ X \sim P(\lambda) \quad or \quad X \sim \pi(\lambda) $$

概率质量函数：

$$ P\{x=k\} = \frac{\lambda^k}{k!}e^{-\lambda} \quad (k=0,1,2..., \lambda>0) $$

##### 2.1.1.3.1 泊松定理

&emsp;&emsp;&emsp;&emsp; 若存在常数 $\lambda>0$ ， 当 $n \to +\infty$ 时，有 $np_n \to \lambda$ ，则

$$ {\lim_{n \to +\infty}} C_n^kp_n^k(1-p_n)^{n-k} = \frac{\lambda ^k}{k!}e^{-k} \quad (k=0,1,...,n)$$

##### 2.1.1.3.2 二项分布的泊松近似计算公式

&emsp;&emsp;&emsp;&emsp; 当 $n \geq 50, p \leq 0.05$ 时，可令 $\lambda = np$ ，使用泊松分布公式近似计算二项分布的概率。

<br>

### 2.1.2 连续型随机变量


#### 2.1.2.1 均匀分布（Uniform distribution）


#### 2.1.2.2 指数分布（Exponential distribution）


#### 2.1.2.3 正态分布（Normal distribution，又名高斯分布，Gaussian distribution）



<hr>

# Chapter 3




<hr>

# Chapter 5

## 5.1. Chebyshev不等式

> 主要用于在不知道分布类型时进行估算

## 5.2. 依概率收敛

## 5.3. 大数定律

### 5.3.1 Chebyshev大数定律

&emsp;&emsp;**定义：** 设 ${X_i}(i=1,2,...)$ 为随机变量序列，令 $Y_n = \frac{1}{n} {\sum\limits_{i=1}^{n}}X_i$ ，若存在常数列 $a_1, a_2, ...$ ，对任意 $\varepsilon >0$ ，有

$$ \lim_{n \to + \infty}P\left\{ |Y_n-a_n| < \varepsilon \right\} = 1 ,$$

则称随机变量序列 ${X_i}$ 服从<font color="red">**大数定律**</font>。


&emsp;&emsp;设 ${X_i}(i=1,2,...)$ 为两两相互独立的随机变量序列，且数学期望 $E(X_i)$ 存在，方差 $D(X_i) \leq c(i=1,2,...)$ ， $c$ 为常数，令 $Y_n = \frac{1}{n} {\sum\limits^{n}_{i=1}X_i}$ ，则对于任意的正数 $\varepsilon$ ，有 

$${\lim_{n \to +\infty}}P\left\{ |Y_n - E\left( Y_n \right)| < \varepsilon \right\} = 1 .$$

特别地，若数学期望 $E(X_i) = \mu, (i=1,2,...),$ 则 $Y_n \xrightarrow{P} \mu, (n \rightarrow \infty) .$

### 5.3.2 Bornoulli大数定律

&emsp;&emsp;设在 $n$ 次独立重复试验中，事件 $A$ 以概率 $p$ 发生了 $m$ 次，则 $\frac{m}{n} \xrightarrow{P}p, (n\rightarrow\infty).$

$Proof:$

&emsp;&emsp;设 $X_i=\left\{
\begin{aligned}
1,& 第i次实验发生A \\
0,& 第i次实验不发生A\\
\end{aligned}
\right. (i=1,2,...,n)$ ，

&emsp;&emsp;则 $X_i \sim b(1,p), E(X_i) = p, D(X_i) = p(1-p)$ 存在， $\{X_i\}$ 相互独立，且事件 $A$ 发生的次数 $m = X_1 + X_2 + ... + X_n$ ，由切比雪夫大数定律得，

$$ Y_n = \frac{1}{n} {\sum\limits_{i=1}^{n}} X_i = \frac{m}{n} \xrightarrow{P} \mu = E(X_i) = p $$

，即 $\frac{m}{n} \xrightarrow{P} p, (n \rightarrow \infty).$


### 5.3.3 Khinchin大数定律

&emsp;&emsp;设 $X_1,X_2,...$ 是<font color="red">**独立同分布**</font>的随机变量序列，若数学期望 $E(X_i) = \mu (i=1,2,...)$ 存在，则
$$ Y_n = \frac{1}{n} {\sum\limits_{i=1}^{n}}X_i \xrightarrow{P} \mu (n \rightarrow \infty). $$

<font color="deepskyblue">**注：**</font>
&emsp;&emsp;设随机变量序列 $ X_1,X_2,... $ <font color="red">**相互独立，均与$X$同分布**</font>，且数学期望 $E(X), E(X^k)$ 均存在，则 $n \rightarrow \infty$ 时有
$$ \frac{1}{n}\sum_{i=1}^{n}X_i \xrightarrow{P} E(X), $$

$$ \frac{1}{n}\sum_{i=1}^{n}X_i^k \xrightarrow{P} E(X^k). $$

## 5.4. 中心极限定理

> 研究和的概率

### 5.4.1 依分布收敛



<br>

### 5.4.2 几个常用的中心极限定理

#### 5.4.2.1 Liapunov中心极限定理




#### 5.4.2.2 Levy-Lindeberg中心极限定理（独立同分布中心极限定理）

&emsp;&emsp;设 $\{\xi_n\}$ 为独立同分布的 随机变量序列，若 $E\xi_k = \mu < \infty, D\xi_k = \sigma^2 < \infty, k=1,2,...$ ，则 $\{\xi_n\}$ 满足中心极限定理。

根据上述定理，当 $n$ 充分大时，
$$ p\{\sum_{i=1}^{n} \xi_i \leq x\} \approx \Phi_0(\frac{x-n\mu}{\sqrt{n}\sigma}) $$

<br>

<font color="gray">例：将一颗骰子连掷100次，则点数之和不少于500的概率是多少？

解：设 $\xi_k$ 为第 $k$ 次掷出的点数， $k=1,2,...,100$ ，则 $\xi_1, ..., \xi_{100}$ 独立同分布。</font>

#### 5.4.2.3 De Moivre-Laplace中心极限定理

&emsp;&emsp;设随机变量 $\xi_n(n=1,2,...)$ 服从参数为 $n, p(0<p<1)$ 的二项分布，则：
$$ \frac{\xi_n - np}{\sqrt{npq}} \xrightarrow{w} \xi \sim N(0,1) (q = 1-p).$$

$Proof:$

&emsp;&emsp;设 $X_i=\left\{
\begin{aligned}
1,& 第i次试验事件A发生 \\
0,& 第i次试验事件A不发生\\
\end{aligned}
\right. (i=1,2,...,n)$ ，

则 $$ E(X_i) = p, \quad D(X_i) = p(1-p), \quad \xi_n = {\sum\limits_{i=1}^{n}X_i} $$
由中心极限定理，结论得证。


#### 5.4.2.4 Laplace定理



<br>

<hr>

# Chapter 7

## 7.1 总体、个体和样本

### 7.1.1 数理统计

&emsp;&emsp;概率论是数理统计的基础，而数理统计是概率论的重要应用。
&emsp;&emsp;数理统计的主要内容：参数估计、假设检验、相关分析、试验设计、非参数统计、过程统计等。

### 7.1.2 总体和个体

&emsp;&emsp;当样本容量有限但很大时，可认为是无限总体。

### 7.1.3 样本

&emsp;&emsp;抽取 $n$ 个个体，设这 $n$ 个个体数量指标为 $(X_1, X_2, ..., X_n)$，则称它为容量为n的样本。

### 7.1.4 样本的联合分布

&emsp;&emsp;设总体 $X$ 的分布函数为 $F(x)$ ，则样本 $(X_1, X_2, ..., X_n)$ 的联合分布函数为
$$ F(x_1,x_2,...,x_n) = \prod_{i=1}^{n}F(x_i) $$

&emsp;&emsp;设 $X$ 为<font color="red">**连续型总体**</font>，其概率密度函数为 $f(x)$ ，则样本 $(X_1, X_2, ..., X_n)$ 的<font color="red">**联合概率密度函数**</font>为
$$ f(x_1,x_2,...,x_n) = \prod_{i=1}^{n}f(x_i) $$

&emsp;&emsp;设 $X$ 为<font color="red">**离散型总体**</font>，其分布律为 $P\{X=a_i\} = p,(i=1,2,...,k)$ ，则样本 $(X_1, X_2, ..., X_n)$ 的<font color="red">**联合分布律**</font>为
$$ P\{X_1=x_1,X_2=x_2,...,X_n=x_n\} = \prod_{i=1}^{n}P(X_i=x_i) $$
其中 $(x_1, x_2, ..., x_i)$ 为一组样本值。

<br>

## 7.2 统计量

### 7.2.1 定义

### 7.2.2 常见的统计量--样本矩

&emsp;&emsp;**定义2**&emsp;&emsp;设 $X_1, X_2, ..., X_n$ 是来自总体 $X$ 的一组样本， $x_1, x_2, ..., x_n$ 是这组样本的观察值。

#### 7.2.2.1 样本均值
$$ \bar X = \frac{1}{n}\sum_{i=1}^{n}X_i $$
注：在原样本均值为 $\mu$, 方差为 $\sigma$ 的情况下， $\bar X$ 的期望为 $\mu$, 方差为 $\frac{\sigma^2}{n}$

#### 7.2.2.2 样本方差
$$ S^2 = \frac{1}{n-1}\sum_{i=1}^{n}(X_i - \bar X)^2 $$
注：均值为 $\sigma^2$

#### 7.2.2.3 样本标准差
$$ S = \sqrt{\frac{1}{n-1}\sum_[i=1]^{n}(X_i - \bar X)^2} $$

#### 7.2.2.4 样本 $k$ 阶原点矩
$$ A_k = \frac{1}{n}\sum_{i=1}^{n}X_i^k \quad (k=1,2,...) $$

注： $A_1 = \bar X$

#### 7.2.2.5 样本 $k$ 阶中心矩
$$ B_k = \frac{1}{n}\sum_{i=1}^{k}(X_i - \bar X)^K \quad (k=1,2,...) $$
注： $B_2 = \frac{n-1}{n}S^2$

### 7.2.3 样本矩与总体矩的关系

&emsp;&emsp;**定理1**&emsp;&emsp;设 $X_1, X_2,...,X_n$ 是来自总体 $X$ 的一组样本，若总体 $X$ 的 $k$ 阶原点矩存在，即设 $\mu_k = E(X^k)$ ，且 $E(X) = \mu, D(X) = \sigma^2$ ，设 $(X_1,X_2,...,X_n)$ 是来自总体 $X$ 的一组简单随机样本，
**则样本矩具有如下性质：**
$$E(\bar X) = \mu, D(\bar X) = \frac{\sigma^2}{n};$$
$$E(S^2) = \sigma^2;$$
$$E(A_k) = \mu_k.$$

## 7.3 抽样分布

### 7.3.1 抽样分布的定义
&emsp;&emsp;<font color="red">**统计量的分布**</font>称为抽样分布

### 7.3.2 $\chi ^2$ 分布

&emsp;&emsp;设 $X_1,X_2,...,X_n$ 为来自总体 $N(0,1)$ 的样本，则统计量
$$\chi^2 = X_1^2+X_2^2+...+X_n^2$$
&emsp;&emsp;服从<font color="red">**自由度为** $n$ </font>的 $\chi^2$ 分布，简记为 $\chi^2 \sim \chi^2(n).$

&emsp;&emsp;其中**自由度**为上式右端包含独立变量的个数。


### 7.3.3 $t$ 分布



### 7.3.4 几个常用统计量的分布
&emsp;&emsp;设总体 $X$ 服从正态分布 $X \sim N(\mu, \sigma^2)$ ，其中 $(X_1, X_2,..., X_n)$ 是来自总体 $X$ 的简单随机样本， $\bar X$ 是样本均值，$S^2$ 是样本均值，则：

$(1) \quad \bar X \sim N(\mu, \frac{\sigma^2}{n}), 即\frac{\bar X - \mu}{\sigma / \sqrt{n}} \sim N(0, 1);$

$(2) \quad \bar X 与 S^2 相互独立;$

$(3) \quad \frac{(n-1)S^2}{\sigma^2} = \frac{1}{\sigma^2}{\sum\limits_{i=1}^{n}}(X_i - \bar X)^2 \sim \chi^2(n-1);$

$变式：(3) \quad \frac{1}{\sigma^2}{\sum\limits_{i=1}^{n}}(X_i - \mu)^2 = {\sum\limits_{i=1}^{n}}(\frac{X_i - \mu}{\sigma})^2 \sim \chi^2(n);$

$(4) \quad \frac{\bar X - \mu}{S / \sqrt{n}} \sim t(n-1).$


<hr>


# Chapter 8  参数估计

## 8.1 点估计

### 8.1.1 矩估计

例2：设总体 $X$ 服从参数为 $\lambda > 0$ 的泊松分布， $X_1, X_2,...,X_n$ 是取自 $X$ 的样本，样本值分别为 $x_1, x_2,..., x_n.$ 求参数 $\lambda$ 的矩估计。
解：


### 8.1.2 极大似然估计

## 8.2 区间估计




