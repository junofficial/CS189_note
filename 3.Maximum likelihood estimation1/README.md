# 1. MLE

1. Loss functions and objectives

<p align="center"><img src="https://github.com/junofficial/CS189_note/assets/124868359/d803adde-2ff3-4838-b4a5-ab21f07410e5" width="200" height="200"/></p>

   - data : $\{(x_i, y_i)\} _{i=1}^{N}$
   - model : $f_{\theta}(x) = y$
   - linear model : $f_{\theta} (x) = \mathbf{w}^T x + b$
   - loss function : $l(y,\hat{y}) = (y-\hat{y})^2$
   - objective : find parameters that minimize the average loss
     $\theta^* = arg\underset{\theta}\min \frac{1}{N} \sum_{i=1}^{N} l(y_i,f_{\theta}(x_i))$

2. The maximum likelihood principle
   - 1. requires probability distribution controlled by model parameters.
      - focus on probabilistic models
   - 2. assume that some setting of parameters generated the observed data
   - 3. We can recover these parameters through optimization by maximizing the likelihood of the observed data.
 

3. Probabilistic models
   - What does a probabilistic model look like?
     <p align="center">$f_{\theta}(x)= \mathcal{N}(\mathbf{w}x+b,{\sigma}^2)$</p>
   - we can see in terms of output probabilities.
     <p align="center">$p_{\theta}(y|x)= \mathcal{N}(y;\mathbf{w}x+b,{\sigma}^2)$</p>

# 2. MLE for machine learning

1. MLE : the basic(non machine learning) setup
   - 주어진 데이터가 $\mathcal{D} = ${ $\mathbf{x}_1, \ldots , \mathbf{x}_N$ }
   - Assume $\mathcal{D}$ was sampled from a member of this family : $X_1, \ldots , X_N \overset{i.i.d.}{\sim} P_{\hat{\theta}}$
   - The goal is to recover $\hat{\theta}$
   - objective/definition : $\theta_{MLE} = arg\underset{\theta \leftarrow \Theta}\max p_{\theta}(\infty) = arg\underset{\theta \leftarrow \Theta}\max\prod_{i=1}^N p_{\theta}(x_i)$

2. MLE : for a univatiate Gaussian
   - Assume that each data point is generated i.i.d. as $X_i \sim \mathcal{N}(\hat{\mu},\hat{\sigma}^2)$
   - Given dataset : $\mathcal{D} = ${ $x_1, \ldots , x_N$ }
   - our goal : $\theta_{MLE} = [\mu_{MLE},]$










   

4. Matrix
   - Square matrix : $n=m$
   - Symmetric matrix : $a_{ij} = a_{ji}, A = A^T$
   - Positive semidefinite matrix : a square, symmetric matrix A for which $X^{T}AX \geq 0,\ for \ all \ X$
   - trace of square matrix $tr(A) = \sum_i a{ii}$
   - Forbenius norm : $\|A\|  = \sqrt{\sum_{i}\sum_{j} |a_{ij}|^2} = \sqrt{tr(A^{T}A)}$


# 3. Eigenvalue and eigenvectors

1. Eigenvalue and eigenvectors
   - 수업 중 설명이 많이 없어서 추가로 공부 진행하였고 공부했던 사이트 링크 첨부하겠습니다.
   - https://darkpgmr.tistory.com/105
2. Singularvalue and singularvectors
   - 마찬가지로 링크 첨부해두겠습니다. 설명이 자세해서 공부할 때는 좋을 것 같습니다.
   - https://darkpgmr.tistory.com/106

# 4. Vector calculus review

1. Gradients
   - 함수를 편미분한 값을 원소로 하는 벡터, f(x) = y 에서 f(x)는 n-dimension column vector이고 이때의 y는 scholar값
   - x와y로 이루어진 함수에서 함수 값 z가 두 값의 변화에 따라 어떻게 변하는지를 나타냄
   - 최적화 문제에서는 이 z값을 최소화 하는 지점 p(x,y)를 찾는 것
   - First order cordinate를 이용하면 $\nabla g(x,y) = 0$
  
2. Hessian Matrix
   - 2차미분함수(이계도함수)를 행렬로 표현한 것
   - $H_{ij} = \frac{\partial ^2 f}{\partial {w_i} \partial {w_j}}$
   - i와 i의 변수로 함수 f를 편미분한 이계도함수를 (i,j)의 원소로 가지고 있는 행렬
   - 연속함수이려면 대칭행렬
   - $Q(h) = h^T H(f) h = h^T Q \mathit{\Lambda} Q^T h \ = \ (Q^T) h^T A Q^T h$
   - 이 때 $u \ = \ Q^T h$
   - $Q(u) = \lambda_1 u_1^2 + \lambda_2 u_2^2 + \cdots + \lambda_n u_n^2$
   - 고유값을 통해 임계점이 극대, 극소, 안장점임을 판별
