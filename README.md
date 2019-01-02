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

##### non-linearity
- activation function으로 non-linear function인 sigmoid, ReLU, tanh 등을 사용하며 특히 여러겹의 layer를 쌓아서 효과를 보려면 반드시 non-linear function을 사용해야한다. *y = c1x, y = c2x* 2개의 linear function으로 layer를 쌓는다고 가정하자. 그러면 2번째 layer의 activation은 c2(c1x) = c1c2x 이므로 이는 *y = c1c2x*인 한개의 layer로 표현이 가능하다. 즉 activation으로 linear function을 이용하여 layer를 쌓는 것은 의미가 없다.

##### CNN(Convolutional Neural Network)
- 일반적인 DNN(Deep Feed Forward Neural Network)에서는 현재 layer와 다음 layer사이에 fully connected 되어 있다. 특히 크기가 큰 image의 경우 input layer에서 1d vector화 할 때 vector의 size가 매우 커지고 이로 인해 hidden layer의 weights의 갯수가 매우 빠르게 늘어난다. 예를들어 color image이고 256 x 256 픽셀크기인 이미지는 1d vector화 했을 때 한개의 neuron에 무려 256x256x3(=196608) 개의 weights가 필요하다. Too many parameters는 overfitting의 위험이 있다. 그래서 **CNN에서는 입력을 이미지로 가정하고 이미지에 적합한 아키텍쳐를 이용한다.(convolution layer)**
- 한줄 정의: convolution operation을 사용하는 layer가 포함된 neural network
	- convolution operation은 kernel의 feature를 detect 한다
	- 내적과 유사한 operation으로 값이 크게나오면 비슷하다고 생각할 수 있음(두 벡터의 내적에서 값이 클수록 같은 방향을 의미하므로)
- fully connected layer
	- 이전 layer와 현재 layer 간의 모든 neron이 연결된 layer
- 특징
	- grid topology에 대하여 잘 작동함
		- time series(1d grid)
		- image(2d grid)
		- video(3d grid)
	- local connectivity
		- insight: 예를 들어 '입' 이라는 특징을 detect 할 때 근접한 pixel만 보면됨
	- parameter sharing(= 커널 = filter) => parameter 수가 적음
		- insight: 어떤 image에서 유용한 feature였다면, 같은 image의 다른 위치 또는 다른 이미지에서도 유용한 feature일 것이다
		- 똑같은 갯수의 neuron을 output으로 만들고자 할 때 fully connected와 비교해보면 훨씬 더 적은 parameter 사용
	- pooling/subsampling hidden units (pooling layer)

- pooling layer
	- 주로 down sampling의 목적(width, height를 줄여줌) => parameter의 갯수와 계산량을 줄여줌 => but 정보 손실의 영향으로 줄여가는 추세
	- max pooling: 해당 region에서 가장 큰 값만 가져옴

- 1x1 filter
	- spatial(widht and height)는 그대로 유지하고 depth를 adjust 하는 용도로 사용 ex) GoogLeNet의 inceoption module
	- activation 추가 가능
	- FC layer와 유사한 역할을 할 수 있음
		- input의 neuron의 갯수가 FC layer를 통과한 후 output neruon의 갯수로 바뀜
		- 1x1 conv에서도 input의 depth가 1x1 filter 갯수에 의해 output depth로 바뀜

- dilated convolution
	- filter를 그대로 input에 긁는게 아니라 중간에 dilation rate를 두고 긁음
	- ex) 3x3 filter를 input의 5x5 영역에 긁음 (dilation rate=2)
	- 똑같은 갯수의 parameter로 더 넓은 영역을 탐색할 수 있음
- tranposed convolution
	- upsampling(작은 image => 큰 image)에 사용 가능


##### Many CNN architectures
- AlexNet
	- conv -> pooling -> normal -> conv -> pooling -> normal -> conv -> conv -> conv -> FC -> FC -> FC
	- 즉 conv와 pooling과 normal과 FC를 순차적으로 배열
	- ReLU를 최초로 사용

- ZFNet
	- AlexNet과 구조는 같으나 hyperparameter(filter size, # of filters)를 바꿔서 더 좋은 결과 만듬

- VGG Net
	- ZFNet이나 AlexNet과 거의 같은데 **filter size를 줄이는 대신 더 deep하게 layer를 쌓음 => small/deeper network**
	- 7x7 1 layer == 3x3 3 layer(size의 측면에서), but less parameters(7x7=49, 3x3x3=27) and deeper(non-linearity를 더 잘 활용가능)

- GoogLeNet
	- 어떤 filter size가 좋을지 모르니 다양한 filter size를 모두 이용
	- 특히 filter size 별로는 parallel하게 계산할 수 있는 장점이 있음
	- 문제는 filter를 굉장히 많이 사용하므로 계산량이 너무 많음
	- 문제 해결을 위해 **inception module**을 도입
	- inception module은 **1x1 convolution layer를 이용한 dimension reduction**으로 계산량을 확 줄여줌
	- 그 결과 계산량 뿐만아니라 parameter의 수도 확 줄어듬

- ResNet
	- 여기서부터 layer가 매우 deep 해짐
	- 기본적으로 neural net이 깊어질수록(deeper) vanishing gradient 이슈로 train이 잘 안됨
	- input 'x'로 부터 output 'H(x)'를 학습하는 대신, f(x) = h(x) - x, 즉 원래 원하던 output 과의 차이(residual)을 학습함
	- 그 결과 망이 깊어져도 학습이 더 잘됨
	- TODO: ResNet 논문 읽어보기(https://arxiv.org/abs/1512.03385)
	- Wide ResNet
		- depth 보다 residual이 important factor 이다
		- depth 대신에 width를 늘려보자, 즉 filter의 갯수를 늘려보자
			- filter를 늘려도 계산은 parallel하게 할 수 있는 장점이 있음
	- TODO
		- ResNext => https://arxiv.org/abs/1611.05431
		- SENet
			- https://arxiv.org/abs/1709.01507
			- https://jayhey.github.io/deep%20learning/2018/07/18/SENet/
		- stochastic depth => https://arxiv.org/abs/1603.09382


##### weight and bias initialization
- gradient 기반 optimizer에서 매우 중요
- if 모든 weight을 0으로 initialize => 모든 neuron의 activation이 같음 => no asymmetry => train이 잘 안됨
- if weight을 너무 작은 값으로 initialize => 모든 activation이 0이되고 gradient도 0이됨 => train이 잘 안됨
- if weight을 너무 큰 값으로 initialize => 모든 activation이 saturate됨 => train이 잘 안됨
- 결국 **적절한** 값으로 weight를 잘 초기화 해야함
	- intuition: z = simga(i: 1~n)(WiXi), 즉 n이 커지면 Wi가 그에 따라 작아져야 비슷한 z가 나옴 => weight를 n과 관련지어서 초기화 해보자
	- xavier initialization: Var(W) = 1 / n^in
		- sigmoid 등의 activation과 잘 맞음
	- xavier variant: Var(W) = 2 / (n^in + n^out)
		- sigmoid 등의 activation과 잘 맞으며 back propagation에도 좋음
	- he initialization: xavier 계열은 종종 ReLU activation과 잘 안맞음 => Var(W) = 2 / n^in 이용
- bias는 0으로 set하는 것으로 충분하다(asymmetry를 만드는 것은 weight로 충분함)

##### batch normalization
- weight initialization은 초기 activation의 분포를 이쁘게 만들어주는 효과가 있었음




##### 기타 정보
- https://blog.lunit.io/2018/08/03/batch-size-in-deep-learning/ -> learning rate와 batch size의 적절한 조합을 잘 찾아야함 -> 최적 hyperparameter조합을 잘 찾는게 매우 중요함, batch size도 '잘' 정해야 하는 요소인데, 작은 경우 좋은 점이 있음(실험 결과적으로 안정적인 training 가능)
- numpy는 매 실행마다 해당 operation에 대한 정보만 있지만 tensorflow는 computational graph 전체에 대한 정보가 있어서 일반적으로 더 빠름
- tf.truncated_normal -> tf.initializers.he_normal()로 initializer를 바꿨더니 학습이 급격하게 잘됨
- dropout 추가했더니 학습이 급격하게 잘됨
