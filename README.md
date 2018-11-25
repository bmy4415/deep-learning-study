##### machine learning
- 컴퓨터가 학습할 수 있도록 하는 알골리즘과 기술을 개발하는 분야
- data로부터 leanring하여 특정 task를 수행

##### representation learning
- 대상의 feature를 learning하는 것

##### train set, test set, validation set(dev set)
- train set: 학습을 위한 data set
- test set: 학습의 결과를 evaluate하기 위한 data set, 학습 중간에는 아에 사용하면 안됨
- validation set: 학습의 중간에 학습이 잘 되고있는지를 평가하기 위한 set(hyper parameter등의 결정에 사용)
- 보통 전체 data set중 일부(15% 내외)를 test set으로 아에 분리한 후 나머지 85%내외를 train set(약70%) + validation set(약15%)로 분리한다
- validation set은 학습 중간에 학습이 잘 이루어지는지를 평가할 때 사용하는데, hyperparameter등 의 결정에 사용한다

##### cross validation, k-fold cross validation
- 전체 data set이 너무 적어서 train set또는 validation set이 너무 적을 경우 명시적으로 validation set을 나누는 대신 test set을 제외한 모든 data set을 k개의 chunk로 나눈 후 각각의 chunk를 validation set으로 하는 k번의 train을 한 후에 각각의 결과의 평균을 validation 결과로 사용하는 방법

##### normarlization
- data의 scale을 맞추기 위해 처리해주는 일련의 행위들
- data의 scale을 맞추지 않으면 일부 data가 다른 모든 data를 overwhelming하여 잘못된 결과가 도출될 수 있음
- 통계적 의미의 standardization($x=\frac { x-\mu  }{ \sigma }$), normarlization($x=\frac { x-{ x }_{ min } }{ { x }_{ max }-{ x }_{ min } }$)을 모두 포함하는 것 같음

##### regularization
- ~~overfitting을 막기위해 사용하는 방법 중 하나로 hypathesis의 굴곡을 sharp -> flat하게 해주는 효과가 있음~~ **-->** overfitting을 막기위한 모든 방법을 통칭, L1 reg(cost에 |W| 추가), L2 reg(cost에 |W|^2 추가)
- cost function을 정의할 때 weight의 값을 더해주는 방식으로 구현 가능
- ex) cost = $cost=\sum { { (y-\bar { y } ) }^{ 2 } } +\lambda \sum { { \left| W \right|  }^{ 2 } }$

##### regression vs classification
- regression: output이 continuous value임, ex) probability estimation
	- linear regression: output의 범위가 제한적이지 않음
	- logistic regression: output이 0과 1 사이의 값임
- classification: output이 discrete value임(class label)
- logistic regression의 마지막에 softmax등을 이용해서 output value -> class label로 바꾸는 방식으로 classification을 만들 수 있음

##### linear regression
- 연속된 범위의 값을 추정
- parameter: w, b
- activation: identity(y = x)
- cost: MSE(mean square error) J = $\frac { 1 }{ m } \sum _{ i=1 }^{ m }{ { ({ y }_{ i } - { \bar { y }  }_{ i }) }^{ 2 } }$

##### logisctic regression
- probability estimation을 위한 linear model
- parameter: w, b
- activation: sigmoid
- cost(log loss): J = $\frac { 1 }{ m } \sum _{ i=1 }^{ m }{ L({ y }_{ i },{ \bar { y }  }_{ i }) } = \frac { 1 }{ m } \sum _{ i=1 }^{ m }{ { -y }_{ i }\log { { \bar { y }  }_{ i } } -(1-{ y }_{ i }) } \log { (1-{ \bar { y }  }_{ i }) }$

##### neuron
- neural net의 기본 구성 요소
- weight, bias, adder, activation으로 구성
- signal: weighted sum of input X ($\sum _{ i=1 }^{ w }{ { x }_{ i }{ w }_{ i } } + b$)
- output = activation h(signal)

##### deep feedforward network
- function approximator
- input X에 대해 hidden layer에서 계산을 진행해나가고 그 결과로 Y가 도출됨(feed forward), 즉 information propagation이 한 방향으로만 이루어짐
- feed back 없음 **-> RNN과 비교**


##### 기타 정보
- https://blog.lunit.io/2018/08/03/batch-size-in-deep-learning/ -> learning rate와 batch size의 적절한 조합을 잘 찾아야함 -> 최적 hyperparameter조합을 잘 찾는게 매우 중요함, batch size도 '잘' 정해야 하는 요소인데, 작은 경우 좋은 점이 있음(실험 결과적으로 안정적인 training 가능)
