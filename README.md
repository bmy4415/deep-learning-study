##### machine learning
- 컴퓨터가 학습할 수 있도록 하는 알골리즘과 기술을 개발하는 분야
- data로부터 leanring하여 특정 task를 수행

##### representation learning
- 대상의 feature를 learning하는 것

##### parameter vs hyper parameter
- parameter: train 중에 machine에 의해 바뀌는 값들, 즉 weight와 bias
- hyper parameter: 좋은 train을 위해 사람이 직접 바꿔주는 값들(learning rate, batch size 등)

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

##### one-hot encoding, one-hot vector
- 범주형 data를 1차원 binary vector로 변환하는 방법
- 필요성 -> 포도 = 0, 사과 = 1, 참외 = 2와 같이 single interger value로 값을 분류할 경우 avg(포도, 참외) = 1 = 사과, 즉 포도와 참외의 평균은 사과이다 등의 이상한 operation이 발생할 수 있음
- classifier의 경우 class를 single integer value가 아니라 한개의 값만 1이고 나머지는 0인 size n(# of classes) vector로 나타낼 수 있음
- ex) 0~9 digit classifier에서 [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] = 1
- 참고: https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f

##### assignment 0-3
- tf.placeholder : data를 받는 변수(input)
- tf.Variable : model의 parameter, 즉 tf에서 optimize하면서 값이 바뀌는 변수
- softmax: activation의 일종으로 n개의 입력을 받아서 각각의 입력이 0~1, 입력의 합은 1로 만들어 주는 함수, softmax의 결과값은 확률과 같은 역할 가능(각각 0과1사이의 양수이며 합이 1이므로)
- cross entrophy: 정의 및 의미, http://blog.naver.com/PostView.nhn?blogId=gyrbsdl18&logNo=221013188633, classifier의 경우 cost function에 cross entrophy 이용

##### max vs argmax
- max of f(x) : 모든 x에 대하여 f(x)가 취할수 있는 값 중 가장 큰 f(x)의 값
- argmax of f(x) : f(x)가 가장 커지는 x의 값

##### numpy broadcasting
- shape: 각 dimension의 크기를 element로 갖는 tuple
- dimension: 축(ex) x, y, z), array의 element를 select할 때 각각의 축을 건너야함
- rank: # of dimnesion
- ex) arr = np.array([[1,2], [3,4], [5,6]])의 경우 축(dimension)이 2개 ** -> 2번 select해야 특정 element에 접근 가능**, dimension이 2개이므로 **rank2인 array**, shape는 각 dimension의 크기를 element로 갖는 tuple이므로 **(3,2)**
- broadcasting: numpy에서는 일반적으로 shape가 다른 array 간의 연산이 불가능한데, 특정 조건이 만족하면 array를 자동으로 변환해서 연산을 진행시켜 주는 것, ex) np.array([1,2,3]) + 3의 경우 array와 값을 계산해야하므로 계산이 불가능해야 하지만 broadcasting에 의해 각각의 element에 3을 더한 [4,5,6]을 return한다
- 참고: http://sacko.tistory.com/16
- broadcasting된 결과 shape 계산하기 -> https://docs.scipy.org/doc/numpy-1.10.1/user/basics.broadcasting.html#general-broadcasting-rules

##### annealing learning rate
- fixed size learning rate를 쓰는것 보다 점차적으로 작아지는 learning rate를 쓰는 것이 더 잘 train할 수 있음, 현실을 예시로 들어보면 미국 뉴욕에서 한국의 경복궁을 찾아갈 때, **비행기(new york -> 인천공항) -> 버스(인천공항 -> 광화문) -> 도보(광화문 -> 경복궁)**순으로 찾아가는 것 처럼 big leanring rate -> small learning rate를 통해 더 세밀한 training가능
- how to annealing
	- step decay: 매 n스텝마다 learning rate를 줄임, ex) 10스텝 마다 절반으로 줄임
	- exponentially deacy: a = a0 * e^-kt, a0와 k는 hyper parameter, 즉 매 step마다 exponentially deacy함
	- 1/t decay: a = a0 / (1 + kt), a0와 k는 hyper parameter
	- k라는 hyper parameter에 대한 해석이 어렵기 때문에 step deacy를 사용하는 경우가 많음

##### SGD(stochastic gradient descent) vs minibatch vs full batch
- stochastic gradient descent: train set 중 하나만 뽑아서 그 gradient를 이용함
- minibatch gradient descent: train set중 minibatch(m개)를 뽑아서 그 평균 gradient를 이용함
- (full) batch gradient descent: 전체 train set을 모두 이용해서 gradient를 계산함
- 보통 m = 2^n인 minibatch를 이용함(2^n인 이유는 memory에 fit할 때 속도가 빠르기 때문)

##### 기타 정보
- https://blog.lunit.io/2018/08/03/batch-size-in-deep-learning/ -> learning rate와 batch size의 적절한 조합을 잘 찾아야함 -> 최적 hyperparameter조합을 잘 찾는게 매우 중요함, batch size도 '잘' 정해야 하는 요소인데, 작은 경우 좋은 점이 있음(실험 결과적으로 안정적인 training 가능)
- numpy는 매 실행마다 해당 operation에 대한 정보만 있지만 tensorflow는 computational graph 전체에 대한 정보가 있어서 일반적으로 더 빠름
