# 1. Probability review

1. Random Variables

   - 랜덤확률변수로 random한 event에 value based outcome을 가지게 된다.
     (예시로는 동전 던지기, 주사위 던지기 등등..)
   - 실제 나온 sample의 경우는 소문자로 작성하는 것이 일반적
   - i.i.d.(independent and identically distributed)한 성질을 가지고 있다.
   - Expected Value
     <p align="center">$\mathbb{E}[X] = \sum_{x} x \cdot P(X = x) $</p>
   - Variance
     <p align="center">$\text{Var}(X) = \sum_{x} (x - \mathbb{E}[X])^2 \cdot P(X = x)$</p>
     <p align="center">$= \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$</p>
   - Bayes'rule, Jenson's inequality와 같은 공식들을 사용가능

2. Random variables의 예시 => 동전던지기 <br/>
 $P(x=1)=0.75, P(x=0)=0.25 \quad \text{here, } X \sim \text{Bernoulli}(0.75)$  <br/>
 When flipping conin N times: $X_1,X_2, \cdots, X_N \sim \text{Bernoulli}(0.75)$<br/>
 expected value : 0.75<br/>
 variance : 0.1875

3. Information theory
   - entropy(H) : 확률이 낮을수록 정보량이 높고 엔트로피가 크다. 떄로는 "expected surprise"로 불리기도 한다.
     <p align="center">$H(X) = -\sum_{x} P(X=x) \log P(X=x) = \mathbb{E}[- \log P(X=x)]$</p>
   - Cross entropy : P와 Q의 분포를 비교, $P=Q$라면 $H(P,Q)=H(P)$
     <p align="center">$H(P,Q) = -\sum_{x} P(X=x) \log Q(X=x) = \mathbb{E}_p[- \log Q(X=x)]$</p>
   - KL divergence :
     <p align="center">$D_{KL} (P \parallel Q) = -\sum_{x} P(X=x) \log \frac{P(X=x)}{Q(X=x)} = H(P,Q) - H(P)$</p>
   - an aside - Monte Carlo estimation :
     <p align="center">$\mathbb{E}_p[f(X)] \cong \frac{1}{N} \sum_{i=1} ^ {N} f(x_i), \quad X_1,X_2, \cdots, X_N \overset{i.i.d.}{\sim} P$</p>
     (latex수식은 제대로 적었는데 시그마가 계속 위로 올라가네요)

   - 추가조사 : 마르코프 부등식, 체비셰프 부등식, 젠슨 부등식
      - 마르코프 부등식 : $\mathbb{E}[X] \geq t P(X \geq t), P(X \geq a) \leq \frac{\mathbb{E}[X]}{a}$
      - 체비셰프 부등식 : $P(|X - \mu| \geq k\sigma) \leq \frac{1}{k^2}$
      - 젠슨 부등식 : $\phi\left(\mathbb{E}[X]\right) \leq \mathbb{E}[\phi(X)]$ (단, 개구간 I에서 정의된 convex함수 일 떄)

# 3. Linear algebra review

1. Vector, matrices, and tensors
   - A Vector is a "1-dimensional" row or column of number
   - A matrix is a "2-dimensional" table
   - Sometimes, "higher dimensional" objects are called tensors
   - 사용될 때 곡선이 없는 굵은 글자로 표시(Vector는 소문자, 이외의 고차원들은 대문자로 작성)

2. Norm
   - 벡터 공간에서 벡터의 크기나 길이를 나타내는 함수로, 물리적인 "길이"에 해당하는 수학적 개념
   - $L_1 norm$은 벡터의 각 성분의 절대값의 합으로 정의
     <p align="center">$\|x\|_1 = \sum_{i=1}^{n} |x_i|$</p>
   - $L_p norm$은 벡터 성분의 p제곱합의 1/p제곱으로 정의됨
     <p align="center">$\|x\|_p = \left( \sum_{i=1}^{n} |x_i|^p \right)^{1/p}$</p>


3. Matrix
   - Square matrix : $n=m$
   - Symmetric matrix : $a_{ij} = a_{ji}, A = A^T$
   - Positive semidefinite matrix : a square, symmetric matrix A for which $X^{T}AX \geq 0, for all X$
   - trace of square matrix $tr(A) = \sum_i a{ii}$
   - Forbenius norm : $\|A\|_F = \sqrt{\sum_{i=1}^{m}\sum_{j=1}^{n} |a_{ij}|^2} = \sqrt{tr(A^{T}A)}$

   <p align="center"><img src="https://github.com/junofficial/mppi_RobotArm/assets/124868359/8aeb1cf3-e6ab-47d1-9b8c-3c4911e0dfe7" width="600" height="300"/></p>

   - Nearest neighbor classifier의 경우 Linear에 비해 학습 데이터를 정확하게 분류할 수 있다.
   - 1-nearest neighbor의 경우 가장 정확하게 분류할 수 있지만 overfitting의 전형적인 예시라고 볼 수 있다.
   - 15-nearest neighbor의 경우 약간 smoothing이 될 수 있는 것을 볼 수 있지만, 수식적으로 표현할 수 없고 non-Linear하다.
  

