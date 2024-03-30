# 1. Probability review

1. Random Variables

   - 랜덤확률변수로 random한 event에 value based outcome을 가지게 된다.
     (예시로는 동전 던지기, 주사위 던지기 등등..)
   - 실제 나온 sample의 경우는 소문자로 작성하는 것이 일반적
   - i.i.d.(independent and identically distributed)한 성질을 가지고 있다.
   - Expected Value
     <p align="center">$E[X] = \sum_{x} x \cdot P(X = x) $</p>
   - Variance
     <p align="center">$\text{Var}(X) = \sum_{x} (x - E[X])^2 \cdot P(X = x)$</p>
     <p align="center">$= E[(X - E[X])^2] = E[X^2] - (E[X])^2$</p>
   - Bayes'rule, Jenson's inequality와 같은 공식들을 사용가능

2. Random variables의 예시 => 동전던지기 <br/>
 $P(x=1)=0.75, P(x=0)=0.25 \quad \text{here, } X \sim \text{Bernoulli}(0.75)$  <br/>
 When flipping conin N times: $X_1,X_2, \cdots, X_N \sim \text{Bernoulli}(0.75)$<br/>
 expected value : 0.75<br/>
 variance : 0.1875

3. Information theory
   - entropy(H) : 확률이 낮을수록 정보량이 높고 엔트로피가 크다.
     <p align="center">$H(X) = -\sum_{x} P(X=x) \log P(X=x) = E[- \log P(X=x)]$</p>


그 예제로는 밑에 같은 것들이 존재한다.
1. Recognizing digits
2. Determining 이메일 스팸
3. 주식 예측 etc..
   
# 2. Types of machine learning problems

<p align="center"><img src="https://github.com/junofficial/CS189_note/assets/124868359/c2399e29-3207-4c91-8eff-b38d95781946" width="600" height="300"/></p>

# 3. A model for classification

- classification의 그림과 같이 O와 X를 나누는 classification을 진행한다고 하자 이떄의 가로 축을 x1, 세로 축을 x2라 할 때

<p align="center">$\theta_1 x_1 + \theta_2 x_2 + \theta_3 = 0$</p>

라고 parameter를 통한 식으로 표현할 수 있을 것이다. 이 때 $\theta_1 x_1 + \theta_2 x_2 + \theta_3 \le 0$ 이면 O에 해당하는 부분일 것이며 그 이외에는 X의 범위에 해당할 것이다.

# 4. A classification example

1. A linear classifier
  
   <p align="center"><img src="https://github.com/junofficial/mppi_RobotArm/assets/124868359/c69f6ed6-a2e2-4c81-9d23-bb8f6e9d1ad4" width="300" height="300"/></p>
   
   - Linear decision boundary를 통해 분류하게 되고 잘 구분하지 못하는 단점이 있다.

2. Nearest neighbor classifiers

   <p align="center"><img src="https://github.com/junofficial/mppi_RobotArm/assets/124868359/8aeb1cf3-e6ab-47d1-9b8c-3c4911e0dfe7" width="600" height="300"/></p>

   - Nearest neighbor classifier의 경우 Linear에 비해 학습 데이터를 정확하게 분류할 수 있다.
   - 1-nearest neighbor의 경우 가장 정확하게 분류할 수 있지만 overfitting의 전형적인 예시라고 볼 수 있다.
   - 15-nearest neighbor의 경우 약간 smoothing이 될 수 있는 것을 볼 수 있지만, 수식적으로 표현할 수 없고 non-Linear하다.
  
3. Bayes optimal classifier

   - 이미 데이터가 어떻게 생성되는지 알 때 사용하는 방법으로 분류가 제대로 이루어졌는지를 확인

# 5. Training set vs Test set vs Validation set

1. Training set
     
   - Training set의 error : classifier의 error를 minimize하기 위해 학습시킴

2. Test set

   - Test set의 error : 학습된 classifier가 얼마나 잘 작동하는지를 평가
   - Test set은 Training set과 완전히 분리되어 있어야 함
   - 다만 이 때의 error값이 Training set의 error 보다 훨씬 크게 된다면 overfitting이 일어난 것
  
3. Validation set

   - Training set의 일부를 evaluating하는 데에 사용하여 overfitting을 예방
   - 또한 학습의 정도를 중간중간 확인하여 학습을 멈추거나 하이퍼파라미터를 설정하는데에 사용
