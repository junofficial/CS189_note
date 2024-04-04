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
   $$Y|X \sim Laplace (\hat{\mathbf{w}}^T X , \sigma)$$
   - where Laplace
   <p align="center">$(y ; \mu, \sigma) = \frac{1}{2 \sigma} exp$ { $-\frac{|y-\mu|}{\sigma}$ }</p>
   - So, the argmax is
   $$arg\underset{w}\max\sum_{i=1}^N log\ Laplace(y_i\ ; \mathbf{w}^T x;\ \sigma)$$
   $$arg\underset{w}\max\sum_{i=1}^N -\frac{1}{\sigma} \underset{(least\ absolute\ deviation)}{|y_i - \mathbf{w}^T x_i|} + constant$$

3. Least absolute deviations vs Least squares
   - 전자는 linear의 절대값 후자는 제곱으로 계산
   - 전자의 경우에는 0으로 가까워 지는것을 확인 할 수 있지만 후자의 경우 반복적인 최적화 알고리즘으로 사용으로 확인하기 어려움
   - 후자의 경우 제곱을 하다보니 outliers에 민감하고 전자는 이러한 outlier에 비교적 강인한 모습
   - 이럴 때 나오는 가우시안 분포의 모습 => heavy tailed distribution
  
4. From MLE to weighted linear regression
   - Assume that output given input is generated independently(not i.i.d.) => 독립이지만 같은 확률분포에서 나온 값이 아님
     $$Y_i|X_i \sim \mathcal{N}(\mathbf{w}^T x_i, \underset{noise}{\sigma^2})$$
   - The objective is:
     $$arg\underset{w}\max\sum_{i=1}^N log\ \mathcal{N}(y_i ;\mathbf{w}^T x_i, \sigma^2)$$
     $$= arg\underset{w}\max\sum_{i=1}^N - \frac{1}{2 \sigma^2}(\mathbf{w}^T x_i - y_i)^2 + const$$
