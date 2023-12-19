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