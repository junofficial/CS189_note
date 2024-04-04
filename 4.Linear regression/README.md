# 1. From MLE to least squares linear regression

1. Some linear algebra on the MLE objective
   - for this setup: $$arg\underset{w}\max -\frac{1}{2 \sigma^2} \sum_{i=1}^{N} (\mathbf{w}^T x_i - y_i)^2 = arg\underset{w}\min \sum_{i=1}^{N} (\mathbf{w}^T x_i - y_i)^2$$
   - this looks like a $l_2-norm$ squared : $||v||_{2}^{2} = \sum_i v_i^2 = V^T V$

2. Solving least squares linear regression
   - objective : $arg\underset{w}\min {||Xw-y||}_2^2 = arg\underset{w}\min {(Xw-y)}^T(Wx-y)$
     $= arg\underset{w}\min w^T X^T X w - 2 y^T X w + y^T y$
   - take gradient and set equal to zero :
   - $\nabla_w : 2 X^T X w - 2 X^T y$
   - $set\ to\ 0\ : X^T X w_{MLE} = X ^ T y \Rightarrow w_{MLE} = {(X^T X)}^{-1} X^T y$
   - Hessian check :
   - $\nabla_w^2 : 2 X^T X\ is\ PSD\ for\ any\ v,$
     $= v^T(2 X^T X) v = 2(Xv)^T Xv = 2{||Xv||}_2^2 \ge 0$

# 2. Other MLE formulations for linear regression
1. Other MLE formulations examples
   - 1. if i.i.d. gaussian noise
          - Least square Linear Regression으로 해결
   - 2. if Laplace distribution noise
          - Least absolute Deviation Linear Regression으로 해결
   - 3. if not i.i.d.
          - Weighted Linear Regression
 

2. From MLE to least abolute deviation
   - Assume that the ouput given the input is generated i.i.d. as 
   $Y|X \sim Laplace (\hat{\mathbf{w}}^T X , \sigma)$
   - we can see in terms of output probabilities.
     <p align="center">$p_{\theta}(y|x)= \mathcal{N}(y;\mathbf{w}x+b,{\sigma}^2)$</p>

# 2. MLE for machine learning

1. MLE : the basic(non machine learning) setup
   - 주어진 데이터가 $\mathcal{D} = ${ $\mathbf{x}_1, \ldots , \mathbf{x}_N$ }
   - Assume $\mathcal{D}$ was sampled from a member of this family : $X_1, \ldots , X_N \overset{i.i.d.}{\sim} P_{\hat{\theta}}$
   - The goal is to recover $\hat{\theta}$
   - objective/definition : $$\theta_{MLE} = arg\underset{\theta \leftarrow \Theta}\max p_{\theta}(\infty) = arg\underset{\theta \leftarrow \Theta}\max\prod_{i=1}^N p_{\theta}(x_i)$$

2. MLE : for a univatiate Gaussian
   - Assume that each data point is generated i.i.d. as $X_i \sim \mathcal{N}(\hat{\mu},\hat{\sigma}^2)$
   - Given dataset : $\mathcal{D} = ${ $x_1, \ldots , x_N$ }
   - our goal : $$\theta_{MLE} = [\mu_{MLE}, \sigma_{MLE}^2] = arg\underset{\mu, \sigma}\max\prod_{i=1}^N \mathcal{N}(x_i;\mu,\sigma^2)$$
   - 하지만 이 모든 곱을 어떻게 미분을 통해 최소값을 찾을까(0으로 만들까?) => Idea is to take log
   - After taking log : $$arg\underset{\mu, \sigma}\max\prod_{i=1}^N \mathcal{N}(x_i;\mu,\sigma^2) = arg\underset{\mu, \sigma}\max\log\prod_{i=1}^N \mathcal{N}(x_i;\mu,\sigma^2)$$
   - so the final expression for the MLE for univariate Gaussian : $$arg\underset{\mu, \sigma}\max \sum_i[-\frac{1}{2} \frac{(x_i-\mu)^2}{\sigma^2} -\frac{1}{2}\log{\sigma}^2 + C]$$
   - In final expression we wish to find :  $$arg\underset{\mu, \sigma}\max -\frac{1}{2}\sum_i^N (x_i-\mu)^2 -\frac{N}{2}\log{\sigma}^2$$

3. As $\theta = [\mu,\sigma^2]$ we are going to find $\mu_{MLE}, \sigma_{MLE}^2$ using derivatives
   <p align="center">$${\partial\over\partial \mu} : \sum_i \frac{x_i-\mu_{MLE}}{\sigma^2} = 0 \Rightarrow \mu_{MLE} : \frac{\sum_i x_i}{N}$$ </p>
   <p align="center">$${\partial\over\partial \sigma^2} : \frac{1}{(2\sigma_{MLE}^2)^2} \sum_i {(x_i-\mu)}^2 - \frac{N}{2\sigma_{MLE}^2} = 0$$ </p>
   <p align="center">$$\Rightarrow \sigma_{MLE}^2 = \frac{\sum_i (x_i - \mu)^2}{N}$$</p>

4. MLE : in the limit of infinite data
   - what happens as the number of data points $N \rightarrow \infty$?
   <p align="center">$$arg\underset{\theta}\max\sum_{i=1}^k \log p_{\theta}(x_i)$$ </p>
   <p align="center">$$\frac{1}{N} \sum_i \log p_{\theta}(x_i) \overset{N \rightarrow \infty}\rightarrow \mathbb{E}_{p_{\hat{\theta}}}[\log p_{\theta}(X)]$$(Monte Carlo estimation) </p>

5. MLE and information theory
   - recall the definition of cross-entropy : $H(p,q) = \mathbb{E}_p[-\log q(x)]$
   - change p to $p_{\hat{\theta}}$ and q to $p_{\theta}$ then:
   <p align="center">$$H(p_{\hat{\theta}},p_{\theta}) = \mathbb{E}_{p_{\hat{\theta}}}[-\log p_{\theta}(X)] \approx \frac{1}{N} \sum_{i=1}^N -\log p_{\theta}(x_i)$$</p>
   - Also cross-entropy to KLD
   <p align="center">$D_{KL}(p_{\hat{\theta}} \parallel p_{\theta}) = H(p_{\hat{\theta}},p_{\theta}) - H(p_{\hat{\theta}})$</p>

# 3. MLE for regression/classification
1. Regression and classification
   - given data : $\mathcal{D} = ${ $(\mathbf{x}_1,y_1), \ldots , (\mathbf{x}_N,y_N)$ }
   - assume a set of distributions on $(\mathbf{x},y)$ : $p_{\theta} : \theta \in \Theta, p_{\theta}(x,y) = p(x)p_{\theta}(y|x)$
   - the parameter $\theta$ only dictate the conditional distribution of y given x
   - the objective, definition are same : 
    <p align="center">$$\theta_{MLE} = arg\underset{\theta \leftarrow \Theta}\max p_{\theta}(\infty) = arg\underset{\theta \leftarrow \Theta}\max\prod_{i=1}^N p_{\theta}(x_i)$$</p>

2. Example : "Least squares" linear regression(최소제곱 선형회귀)
   - data : $\mathcal{D} = ${ $(\mathbf{x}_1,y_1), \ldots , (\mathbf{x}_N,y_N)$ }
   - $Y|X \sim \mathcal{N}(\hat{\mathbf{w}}^T X + \hat{b}, {\sigma}^2)$
   - $Y = \hat{\mathbf{w}}^T X + \hat{b} + \varepsilon \leftarrow \varepsilon \sim \mathcal{N}(0,{\sigma}^2)$(noise)
  
3. $\theta$에 대해서 결정되는 것들을 구분($\theta$에 의해서 결정되는 input이 아니면 고려하지 않아도 됨)
   - the object is : $$arg\underset{\theta \leftarrow \Theta}\max \sum_{i=1}^{N} \log \mathcal{N}(y_i ; \mathbf{w}^T x_i + b))$$
     $$=arg\underset{\theta}\max \sum_i \frac{1}{2 \sigma^2}(\mathbf{w}^T x_i + b - y_i)^2 + constant w.r.t. \theta$$