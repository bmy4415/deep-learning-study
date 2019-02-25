##### machine learning
- 컴퓨터가 학습할 수 있도록 하는 알고리즘과 기술을 개발하는 분야
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
- 통계적 의미의 standardization(![equation](https://latex.codecogs.com/gif.latex?x%27%3D%5Cfrac%20%7B%20x-%5Cmu%20%7D%7B%20%5Csigma%20%7D))과 normarlization(![equation](https://latex.codecogs.com/gif.latex?x%27%3D%5Cfrac%20%7B%20x-%7B%20x%20%7D_%7B%20min%20%7D%20%7D%7B%20%7B%20x%20%7D_%7B%20max%20%7D-%7B%20x%20%7D_%7B%20min%20%7D%20%7D)) 둘 다  포함하는 것 같음

##### regularization
- ~~overfitting을 막기위해 사용하는 방법 중 하나로 hypathesis의 굴곡을 sharp -> flat하게 해주는 효과가 있음~~ **-->** overfitting을 막기위한 모든 방법을 통칭, L1 reg(cost에 |W| 추가), L2 reg(cost에 |W|^2 추가) 등이 있음
- weight normalization(= weight regularization)에는 크게 L1 norm과 L2 norm 2가지를 사용함
- cost function을 정의할 때 norm의 값을 더해주는 방식으로 구현 가능
- ex) L2 weight normalization: ![equation](https://latex.codecogs.com/gif.latex?cost%3D%5Csum%20%7B%20%7B%20%28y-%5Cbar%20%7B%20y%20%7D%20%29%20%7D%5E%7B%202%20%7D%20%7D%20&plus;%5Clambda%20%5Csum%20%7B%20%7B%20%5Cleft%7C%20W%20%5Cright%7C%20%7D%5E%7B%202%20%7D%20%7D)

##### regression vs classification
- regression: output이 continuous value임, ex) probability estimation
	- linear regression: output의 범위가 제한적이지 않음
	- logistic regression: output이 0과 1 사이의 값임
- classification: output이 discrete value임(class label)
- logistic regression의 마지막에 softmax등을 이용해서 output value -> class label로 바꾸는 방식으로 classification을 만들 수 있음

##### linear regression
- 연속된 범위의 값을 추정하는 linear model
- parameter: w, b
- activation: identity(![euqation](https://latex.codecogs.com/gif.latex?y%3Dx)), 또는 별도의 activation이 없다고도 할 수 있음
- cost function: MSE(mean squared error) ![equation](https://latex.codecogs.com/gif.latex?cost%20%3D%20%5Cfrac%20%7B%201%20%7D%7B%20m%20%7D%20%5Csum%20_%7B%20i%3D1%20%7D%5E%7B%20m%20%7D%7B%20%7B%20%28%7B%20y%20%7D_%7B%20i%20%7D%20-%20%7B%20%5Cbar%20%7B%20y%20%7D%20%7D_%7B%20i%20%7D%29%20%7D%5E%7B%202%20%7D%20%7D)

##### logisctic regression
- probability estimation을 위한 linear model
- parameter: w, b
- activation: sigmoid
- cost function: log loss ![equation](https://latex.codecogs.com/gif.latex?cost%3D%5Cfrac%20%7B%201%20%7D%7B%20m%20%7D%20%5Csum%20_%7B%20i%3D1%20%7D%5E%7B%20m%20%7D%7B%20L%28%7B%20y%20%7D_%7B%20i%20%7D%2C%7B%20%5Cbar%20%7B%20y%20%7D%20%7D_%7B%20i%20%7D%29%20%7D%20%3D%20%5Cfrac%20%7B%201%20%7D%7B%20m%20%7D%20%5Csum%20_%7B%20i%3D1%20%7D%5E%7B%20m%20%7D%7B%20%7B%20-y%20%7D_%7B%20i%20%7D%5Clog%20%7B%20%7B%20%5Cbar%20%7B%20y%20%7D%20%7D_%7B%20i%20%7D%20%7D%20-%281-%7B%20y%20%7D_%7B%20i%20%7D%29%20%7D%20%5Clog%20%7B%20%281-%7B%20%5Cbar%20%7B%20y%20%7D%20%7D_%7B%20i%20%7D%29%20%7D)

##### neuron
- neural net의 기본 구성 요소
- weight, bias, adder, activation으로 구성
- signal: weighted sum of input X (![equation](https://latex.codecogs.com/gif.latex?%5Csum%20_%7B%20i%3D1%20%7D%5E%7B%20w%20%7D%7B%20%7B%20x%20%7D_%7B%20i%20%7D%7B%20w%20%7D_%7B%20i%20%7D%20%7D%20&plus;%20b))
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
- ex) arr = np.array([[1,2], [3,4], [5,6]])의 경우 축(dimension)이 2개 **-> 2번 select해야 특정 element에 접근 가능**, dimension이 2개이므로 **rank2인 array**, shape는 각 dimension의 크기를 element로 갖는 tuple이므로 **(3,2)**
- broadcasting: numpy에서는 일반적으로 shape가 다른 array 간의 연산이 불가능한데, 특정 조건이 만족하면 array를 자동으로 변환해서 연산을 진행시켜 주는 것, ex) np.array([1,2,3]) + 3의 경우 array와 값을 계산해야하므로 계산이 불가능해야 하지만 broadcasting에 의해 각각의 element에 3을 더한 [4,5,6]을 return한다
- 참고: http://sacko.tistory.com/16
- broadcasting된 결과 shape 계산하기 -> https://docs.scipy.org/doc/numpy-1.10.1/user/basics.broadcasting.html#general-broadcasting-rules

##### annealing learning rate
- fixed size learning rate를 쓰는것 보다 점차적으로 작아지는 learning rate를 쓰는 것이 더 잘 train할 수 있음, 현실을 예시로 들어보면 미국 뉴욕에서 한국의 경복궁을 찾아갈 때, **비행기(new york -> 인천공항) -> 버스(인천공항 -> 광화문) -> 도보(광화문 -> 경복궁)**순으로 찾아가는 것 처럼 big leanring rate -> small learning rate를 통해 더 세밀한 training가능
- how to annealing
	- step decay: 처음에는 고정 learning rate로 학습하다가 이후에 매 n스텝마다 learning rate를 줄임, ex) 10스텝 마다 절반으로 줄임
	- exponentially deacy: ![equation](https://latex.codecogs.com/gif.latex?a%3D%5Cfrac%20%7B%20%7B%20a%20%7D_%7B%200%20%7D%20%7D%7B%20%7B%20e%20%7D%5E%7B%20kt%20%7D%20%7D), ![equation](https://latex.codecogs.com/gif.latex?%7B%20a%20%7D_%7B%200%20%7D%2C%20k)는 hyper parameter, t는 step 수, 즉 매 step마다 exponentially deacy함
	- 1/t decay: ![equation](https://latex.codecogs.com/gif.latex?a%3D%5Cfrac%20%7B%20%7B%20a%20%7D_%7B%200%20%7D%20%7D%7B%201&plus;kt%20%7D), ![equation](https://latex.codecogs.com/gif.latex?%7B%20a%20%7D_%7B%200%20%7D%2C%20k)는 hyper parameter
	- k라는 hyper parameter에 대한 해석이 어렵기 때문에 step deacy를 사용하는 경우가 많음
    - resource가 충분하다면 learning rate를 낮게해서 오랜 시간동안 학습 시키는 것이 좋음

##### SGD(stochastic gradient descent) vs minibatch vs full batch
- stochastic gradient descent: train set 중 하나만 뽑아서 그 gradient를 이용함
- minibatch gradient descent: train set중 minibatch(m개)를 뽑아서 그 평균 gradient를 이용함
- (full) batch gradient descent: 전체 train set을 모두 이용해서 gradient를 계산함
- 보통 ![equation](https://latex.codecogs.com/gif.latex?m%3D2%5En)인 minibatch를 이용함(메모리의 크기에 맞추기 위해)

##### non-linearity
- activation function으로 non-linear function인 sigmoid, ReLU, tanh 등을 사용하며 특히 여러겹의 layer를 쌓아서 효과를 보려면 반드시 non-linear function을 사용해야한다.
- ![equation](https://latex.codecogs.com/gif.latex?y%3D%7B%20c%20%7D_%7B%201%20%7Dx%2C%20y%3D%7B%20c%20%7D_%7B%202%20%7Dx) 2개의 linear function으로 layer를 쌓는다고 가정하자. 그러면 2번째 layer에서 ![equation](https://latex.codecogs.com/gif.latex?activation%3D%7B%20c%20%7D_%7B%202%20%7D%28%7B%20c%20%7D_%7B%201%20%7Dx%29%3D%7B%20c%20%7D_%7B%201%20%7D%7B%20c%20%7D_%7B%202%20%7Dx) 이므로 한개의 layer로 표현이 가능하다. 즉 linear activation으로 여러개의 layer를 쌓는 것은 의미가 없다.

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
- filter
	- input image와 convolution operation을 수행하는 parameters
- padding
	- convolution, pooling을 적용할 때 input image의 상하좌우에 붙여주는 0
	- padding과 stride로 output의 width와 height를 조절할 수 있다
- stride
	- convolution, pooling을 적용할 때, filter를 몇칸씩 건너 뛰면서 이동할 것인지를 나타냄
	- padding과 함께 사용되는데 stride가 1일 때와 1보다 클 때가 헷갈릴 수 있다
		- https://www.tensorflow.org/api_guides/python/nn#Convolution 참고
		- same padding
			- ![equation](https://latex.codecogs.com/gif.latex?out%3D%5Cleft%5Clceil%20%5Cfrac%20%7B%20in%20%7D%7B%20stride%20%7D%20%5Cright%5Crceil)
		- valid padding
			- ![equation](https://latex.codecogs.com/gif.latex?out%3D%5Cleft%5Clceil%20%5Cfrac%20%7B%20in-filter&plus;1%20%7D%7B%20stride%20%7D%20%5Cright%5Crceil)

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
	- TODO: ResNet 논문 읽어보기(https://arxiv.org/abs/1512.03385), 학습이 잘되는 이유?(f(x)의 gradient가 1(큰 숫자)여서 인듯함)
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
	- intuition: ![equation](https://latex.codecogs.com/gif.latex?z%3D%5Csum%20_%7B%20i%20%7D%5E%7B%20%7D%7B%20%7B%20x%20%7D_%7B%20i%20%7D%7B%20w%20%7D_%7B%20i%20%7D%20%7D), 즉 n이 커지면 ![equation](https://latex.codecogs.com/gif.latex?%7B%20w%20%7D_%7B%20i%20%7D)가 그에 따라 작아져야 비슷한 크기의 z가 나옴 => weight를 n과 관련지어서 초기화 해보자
	- xavier initialization: ![equation](https://latex.codecogs.com/gif.latex?Variance%28W%29%3D%5Cfrac%20%7B%201%20%7D%7B%20%7B%20n%20%7D_%7B%20in%20%7D%20%7D)
		- sigmoid 등의 activation과 잘 맞음
	- xavier variant: ![equation](https://latex.codecogs.com/gif.latex?Variance%28W%29%3D%5Cfrac%20%7B%202%20%7D%7B%20%7B%20n%20%7D_%7B%20in%20%7D&plus;%7B%20n%20%7D_%7B%20out%20%7D%20%7D)
		- sigmoid 등의 activation과 잘 맞으며 back propagation에도 좋음
	- he initialization: xavier 계열은 종종 ReLU activation과 잘 안맞음 => ![equation](https://latex.codecogs.com/gif.latex?Variance%28W%29%3D%5Cfrac%20%7B%202%20%7D%7B%20%7B%20n%20%7D_%7B%20in%20%7D%20%7D) 의 경우 잘 된다
- bias는 0으로 set하는 것으로 충분하다(asymmetry를 만드는 것은 weight로 충분함)

##### batch normalization
- weight initialization은 초기 activation의 분포를 이쁘게 만들어주는 효과가 있었음
- normalization(![equation](https://latex.codecogs.com/gif.latex?x%27%3D%5Cfrac%20%7B%20x-%7B%20mean%20%7D_%7B%20batch%20%7D%20%7D%7B%20%7B%20std%20%7D_%7B%20batch%20%7D%20%7D)) + linear transform(![equation](https://latex.codecogs.com/gif.latex?y%3Dax&plus;b))로 이루어짐(a, b는 learn 해야할 parameter)
- higher learning rate, regularizer의 효과가 있음
- batch에 dependent함(각 minibatch 마다 mean과 std를 구함)
	- RNN, GAN, RL, online-learning 등에서 사용하기 힘듬
	- batch independent normalizer
		- weight norm: https://arxiv.org/pdf/1602.07868.pdf
		- layer norm(sequential model에 잘 작동): https://arxiv.org/pdf/1607.06450.pdf
		- instance norm(sequential model에 잘 작동)
		- group norm(video recognition에 잘 작동): https://arxiv.org/pdf/1803.08494.pdf
- 결론
	- dnn, cnn => batch norm 사용 시도
	- rnn, gan rl => batch independent normalizer 사용 고려
- TODO: 6. training art 32page 공부 더하기
	- 구글링
	- batch norm 논문

##### training tips
- hyper parameters
	- coarse-to-fine sampling
		- 처음에 넓은 범위에 작은 epochs로 train을 하다가 좁은 범위에 큰 epochs로 변경
		- grid search 대신 random sampling을 이용
		- sampling 할 때
			- binary/discrete => sample from bernoulli / multinoulli distribution
			- real valued => uniform / log-uniform(usually better)
- capacity는 높게하고 regularization을 철저히 할 것

##### paper study: Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps
- source: https://arxiv.org/pdf/1312.6034.pdf 참고
- Abstract
	- trained cnn model을 visualization 하는 방법을 다룸
	- 첫째로 class score를 maximize하는 image를 generate 해봄 => conv net이 capture하는 개념을 visualize 해줌
	- 둘째로 (given image, given class)에 대한 saliency map을 compute 해봄 => weakly supervised object localization에 사용 가능, image에서 class score에 기여하는 부분을 highlight 해줌
	- deconvolutional network와 gradient-based convnet visualization 사이의 연관성에 대해 언급
- Introduction
	- cnn이 large scale image recognition problem에서 성공한 architecture가 되면서 cnn이 capture하는 visual appearance에 대해 연구가 늘어남
    - 이전에 unsupervised DBN, auto-encoder에 대한 visualization에 대한 시도가 있었음
	- cnn에 대해서는 deconvolutional network로 visualize 하려는 시도가 있었음
	- 이 논문에서는 pre-trained cnn model을 이용하여 (이전 연구들과 비교하여) 다음의 3가지를 성취
		- (human level에서) understandable한 visualization에 성공
			- Erhan의 연구를 최초로 cnn에 적용
		- given image, given class에 대하여 single back propagation을 통해 pixel wise 기여도를 구하는 방법을 제시 => saliency map
			- 이는 추후에 weakly supervised object localization에 이용 가능
		- gradient-based visualization 방법이 deconvolution network의 일반화된 방법임에 대하여 설명(conv layer 뿐만아니라 다양한 layer에 사용 가능)
- Class Model Visualization
	- input: trained model S, class C
	- output: image
	- pre-train된 model을 이용하여 주어진 class의 class score을 maximize하는 image를 generate 해내는 방법
	- Sc(I)를 image I의 class c에 대한 score라고 할 때, ![equation](https://latex.codecogs.com/gif.latex?objective%3D%7B%20S%20%7D_%7B%20c%20%7D%28I%29-%5Clambda%20%7B%20%28%7B%20%5Cleft%5C%7C%20I%20%5Cright%5C%7C%20%7D_%7B%202%20%7D%29%20%7D%5E%7B%202%20%7D) 을 maximize하는 I를 찾아내면 됨
		- lambda: regularization parameter
		- cnn train과 유사하게 back propagation을 이용해서 local optima를 찾을 수 있음
		- objective를 maximize하므로 gradient descent가 아니라 gradient aescent 이용
		- model의 weight를 바꾸는 것이 아니라 input image I를 바꿈
		- image generate 방법
			1. I를 random noise로 초기화(필요하다면 normalize 추가)
			2. forward propagation을 통해 objective 값 계산
			3. backward propagation을 통해 objecitve에 대한 I의 gradient를 구함
			4. ![equation](https://latex.codecogs.com/gif.latex?I%5Cleftarrow%20I&plus;%5Clambda%20*gradient)
			5. 2~4의 과정을 반복
		- softmax의 결과인 probability distribution이 아니라, 그 이전 값인 unnormalized score를 이용하는 이유
			- intuition: softmax의 결과는 target class의 score를 증가시켜도 되지만, 나머지 class의 score를 감소시켜도 되므로 원래의 목표인 class score의 maximization과 살짝 다름
			- 실험해 봤더니 probability를 쓴 경우 결과 image가 understandable 하지 않음
- Image-Specific Class Saliency Visualization
	- input: trained model S, image I, class C
	- output: saliency map
	- image I의 class C에 대한 각 pixel의 기여도를 나타냄
	- 그러므로 input image의 spatial shape(H,W) == saliency map의 shape(H,W)
	- ![equation](https://latex.codecogs.com/gif.latex?%7B%20S%20%7D_%7B%20c%20%7D%28I%29%3D%5Cfrac%20%7B%20%5CDelta%20%7B%20S%20%7D_%7B%20c%20%7D%28I%29%20%7D%7B%20%5CDelta%20I%20%7D%20*I&plus;b) => 1차 테일러 근사에서 출발
		- ![equation](https://latex.codecogs.com/gif.latex?grad%3D%5Cfrac%20%7B%20%5CDelta%20%7B%20S%20%7D_%7B%20c%20%7D%28I%29%20%7D%7B%20%5CDelta%20I%20%7D) 라고 하면, grad 자체가 I의 score에 대한 기여로 해석 가능
			- grad의 특정 위치값이 크다 => 이 위치의 값이 score의 변화에 큰 영향을 준다 => 이 위치가 중요한 pixel이다
		- saliency map 구하는 방법
			1. input image I를 forward pass를 거쳐 class score를 구함 (softmax 이전의 값)
			2. 1의 결과 중 target class에 해당하는 값만을 extract
			3. 2의 결과의 input image I에 대한 gradient를 구함(![equation](https://latex.codecogs.com/gif.latex?%5Cfrac%20%7B%20%5CDelta%20%28result%5Cquad%20of%5Cquad%202.%29%20%7D%7B%20%5CDelta%20I%20%7D))
			4. 3의 결과에 절대값(absolute)을 취함 -> 영향력의 크기를 보기 위해, 음수의 경우도 영향력이 크다고 할 수 있음
			5. input image의 scale에 따라
				- grey scale이면 4의 결과의 shape이 (H,W)이고 이 자체가 saliency map임
				- multi-channel이면 4의 결과의 shape이 (H,W,C)이고 이것을 channel-wise max를 취하면 shape이 (H,W)가 되고 이것이 saliency map임
	- weakly supervised object localization에 관련된 부분은 graph cut image segmentation 등 다른 부분에 대한 지식이 필요하여 skip(TODO)
- Relation to Deconvolutional Networks
	- deconvnet-based reconstruction과 gradient based visualization의 비교
	- 사실의 이 둘이 거의 비슷함
		- 둘다 backward propagation을 통해 image를 generate함
		- conv layer
			- ![equation](https://latex.codecogs.com/gif.latex?%7B%20X%20%7D_%7B%20n&plus;1%20%7D%3D%7B%20X%20%7D_%7B%20n%20%7D%5Cquad%20conv%5Cquad%20%7B%20K%20%7D_%7B%20n%20%7D)
			- gradient based: ![equation](https://latex.codecogs.com/gif.latex?grad%28%7B%20X%20%7D_%7B%20n%20%7D%29%3Dgrad%28%7B%20X%20%7D_%7B%20n&plus;1%20%7D%29%5Cquad%20conv%5Cquad%20flipped%28%7B%20K%20%7D_%7B%20n%20%7D%29)
			- deconvnet based: ![equation](https://latex.codecogs.com/gif.latex?R_%7B%20n%20%7D%3D%7B%20R%20%7D_%7B%20n&plus;1%20%7D%5Cquad%20conv%5Cquad%20flipped%28%7B%20K%20%7D_%7B%20n%20%7D%29)
		- relu layer
			- ![equation](https://latex.codecogs.com/gif.latex?X_%7B%20n&plus;1%20%7D%3Dmax%28%7B%20X%20%7D_%7B%20n%20%7D%2C%200%29)
			- gradient based: ![equation](https://latex.codecogs.com/gif.latex?grad%28%7B%20X%20%7D_%7B%20n%20%7D%29%3Dgrad%28%7B%20X%20%7D_%7B%20n&plus;1%20%7D%29*1%5Cquad%20if%5Cquad%20%7B%20X%20%7D_%7B%20n%20%7D%3E0%5Cquad%20else%5Cquad%200)
			- deconvnet based: ![equation](https://latex.codecogs.com/gif.latex?%7B%20R%20%7D_%7B%20n%20%7D%3D%7B%20R%20%7D_%7B%20n&plus;1%20%7D*1%5Cquad%20if%5Cquad%20%7B%20R%20%7D_%7B%20n&plus;1%20%7D%3E0%5Cquad%20else%5Cquad%200)
			- gradient based에서는 n번째 layer를 보고 grad(Xn)값을 정하지만, deconvnet based에서는 (n+1)번째 layer를 보고 Rn 값을 정하는 차이가 있음
		- max pooling layer
			- ![equation](https://latex.codecogs.com/gif.latex?%7B%20X%20%7D_%7B%20n&plus;1%20%7D%28p%29%3D%5Cmax%20_%7B%20q%5Cquad%20in%5Cquad%20region%5Cquad%20p%20%7D%7B%20%28q%29%20%7D)
			- gradient based: ![equation](https://latex.codecogs.com/gif.latex?grad%28%7B%20X%20%7D_%7B%20n%20%7D%29%28q%29%3Dgrad%28%7B%20X%20%7D_%7B%20n&plus;1%20%7D%29%28p%29*1%5Cquad%20if%5Cquad%20q%5Cquad%20is%5Cquad%20max%5Cquad%20in%5Cquad%20region%5Cquad%20p%5Cquad%20else%5Cquad%200)
			- deconvnet based: 'switch'? in deconvnet(deconvnet 논문을 읽어야 알 수 있을듯함)
		- 위와 같은 이유로 deconvnet과 gradient based back propagation visualization이 equivalent하다고 할 수 있음
		- 또한 gradient based는 conv layer 뿐만아니라 어떤 종류의 layer에도 사용 가능
- Conclusion
	- 이 paper에서 2가지 visualization technique를 제시함
		- generate artificial image that maximizes class score
		- image-specific class saliency map
	- saliency map은 graph-cut image segmentation에도 활용 가능(segmentation이나 detection model에 대한 train없이도 가능)
	- gradient based visualization과 deconvolutional network 사이의 연결성에 대한 언급
	- 후속 연구로 saliency map을 이용하여 learning하는 방법에 대한 연구를 하면 좋을듯 함

##### paper study: Understanding Neural Networks Through Deep Visualization
- source: http://yosinski.com/media/papers/Yosinski__2015__ICML_DL__Understanding_Neural_Networks_Through_Deep_Visualization__.pdf
- Abstract
	- DNN, CNN 등에서 좋은 성과(performance)를 내는 등 발전이 많았음
	- 그러나 중간 과정(hidden layer)에서 어떤 일을 하는지에 대해서는 black box인 상태
	- 중간 과정을 visualize, interpret 할 수 있다면 더 나은 intuition을 통해 발전 가능
	- 이 논문에서 visualize, interpret과 관련하여 2가지 tool을 제시
		- image, video input에 대하여 trained convnet의 각 layer마다의 activation을 visualize해주는 software를 개발 - input의 변화에 따라 activation의 변화를 바로바로 확인 가능
		- regularized optimization을 통해 각 layer가 capture하는 **recognizable**한 image를 생성하는 방법 제시, 이전의 방법들과 비교하여 더 다양한 regularizer를 사용하였고 더 understandable한 image를 생성 가능
	- 위의 2가지 tool은 모두 pre-trained convnet에 대하여 사용 가능
- Introduction
	- faster computing(GPU 등), training technique(dropout 등), activation(relu 등) 등의 발전으로 high performance를 내는 architecture가 많이 발전해왔음
	- 그러나 hidden layer에서 어떤 일을 하는지에 대한 understading은 여전히 부족
		- layer가 깊어지고 parameter가 많아지고 각 neuron이 복잡하게 연결되어 있어서 분석이 힘듬
	- hidden layer가 하는 일에 대해서 분석할 수 있다면 더 나은 architecture를 만드는데에 intuition을 줄 것임 -> ex) deconvnet에서 smaller filter size에 대한 영감을 얻어서 ZFnet이 ImageNet 2013 우승함
	- visualization을 통해 newcomer들에게 network에 대하여 쉽게 설명할 수 있음
	- 첫번째 tool(software)은 static image나 live video를 input으로 받아서 input에 대한 각 layer의 activation을 visualize 해줌
	- 두번째 tool에서 trained CNN의 각 layer가 어떤 feature를 train했는지 visualize 할 수 있는 방법을 제시(regularize technique 설명)
	- 그 이전 연구들에 대하여 설명
- Visualizing Live Convnet Activations
	- 개발한 software에 대해서 설명
	- FC layer만으로 이루어진 network의 경우 neuron의 순서가 의미가 없어서 vector을 visualize 하는 것이 큰 의미가 없음(단 CNN에서 마지막 FC layer는 학습된 image의 조합을 통해 classification을 하므로 의미가 있음)
	- CNN의 경우 conv layer에서의 activation들은 local connectivity가 있으므로 activation vector를 plot 하는 것이 spatial한 의미가 있음
	- 이 software를 이용해서 얻은 intuition
		- 어떤 feature를 detect 할 때, all layer에 걸쳐서 하나의 feature를 detect 하지 않고 하나의 특정 layer에서 complex feature를 detect하는 layer들이 존재 했음
		- Google이나 Flickr에서 들어오는 image에 대해서는 전반적으로 classification 정확도가 높았지만 webcam(video input)에 대해서는 그렇지 않았음. 그 이유 중 하나는 training 때 보지 못한 class의 object가 input으로 들어왔기 때문이라고 추론할 수 있음
- Visualizing via Regularized Optimization
	- 이 paper에서 적용한 regularization에 대하여 설명
	- ![equation](https://latex.codecogs.com/gif.latex?obj%3D%7B%20a%20%7D_%7B%20i%20%7D%28x%29-%7B%20R%20%7D_%7B%20%5Ctheta%20%7D%28x%29)
	- R-theta는 L2 norm을 의미, 그 후에 x를 update 할 때 다음과 같이 update함
	- ![equation](https://latex.codecogs.com/gif.latex?x%5Cleftarrow%20%7B%20r%20%7D_%7B%20%5Ctheta%20%7D%28x&plus;%5Ceta%20%5Cfrac%20%7B%20%5CDelta%20obj%20%7D%7B%20%5CDelta%20x%20%7D%20%29)
	- r-theta는 다음의 4가지 regularization을 모두 적용함
		- L2 decay
			- L2 decay는 small number of extreme values가 결과에 dominant 하게 되는 것을 방지해줌
			- ![euqation](https://latex.codecogs.com/gif.latex?%7B%20r%20%7D_%7B%20%5Ctheta%20%7D%3D%281-%7B%20%5Ctheta%20%7D_%7B%20decay%20%7D%29%5Cbullet%20x)
		- Gaussian blur
			- gradient aescent로 image를 generate하면 high-frequency image가 나오는데, 이는 activation을 크게 해주지만 인간의 눈으로 보기에 real하지 않음
			- 그래서 Gaussian blur를 통해 frequency를 낮춰주었음 -> parameter: bandwidth
			- 이 blur operation은 매우 expensive하기 때문에 매 iteration마다 할 수는 없고, every n step마다 수행하였음 -> parameter: every n step
			- blur operation을 small bandwidth로 여러번 할 경우 큰 band width로 한번 수행한 것과 equivalent한 결과를 나타내며, 특히 여러 번의 blur 동안 input image가 조금씩 변해도 전체 결과가 (큰 bandwidth로 한번 수행한 결과와)similiar하게 나오는 효과가 있음
			- parameter: ![euqation](https://latex.codecogs.com/gif.latex?%7B%20%5Ctheta%20%7D_%7B%20b%5C_%20width%20%7D%5Cquad%20/%5Cquad%20%7B%20%5Ctheta%20%7D_%7B%20b%5C_%20every%20%7D)
		- Clipping pixels with small norm
			- 위의 2가지 regularizer를 적용한 후에 남은 image x에는 많은 non-zero value가 있는데, 이것들 하나하나가 activation에 기여하지만 이를 전부 나타내지 말고 main object만 나타내고 싶고(특징이 두드러 질 수 있도록), 그래서 각 pixel의 norm을 계산하여 norm이 threshold인 ![equation](https://latex.codecogs.com/gif.latex?%7B%20%5Ctheta%20%7D_%7B%20n%5C_%20pct%20%7D) 보다 작은 pixel은 전부 0으로 만들어줌
		- Clipping pixels with small contribution
			- norm이 적은 pixel을 지워내는 방법 외에 activation에 대한 contribution이 작은 pixel을 지워낼수 있음(pixel 값을 0으로 만들 수 있음)
			- activation에 대한 pixel의 기여도를 확인하는 방법에는 다음의 2가지가 있음
				1. 
					- 직접 각 pixel들을 0으로 만든 후에 activation의 변화량을 측정함, 다음의 수식에서 '_j'는 j번째 pixel을 0으로 만든 image를 의미함
					- ![equation](https://latex.codecogs.com/gif.latex?%5Cleft%7C%20%7B%20a%20%7D_%7B%20i%20%7D%28x%29-%7B%20a%20%7D_%7B%20i%20%7D%28%7B%20x%20%7D_%7B%20%5C_%20j%20%7D%29%20%5Cright%7C)
					- 모든 pixel에 대하여 직접 0으로 만든 후 difference를 측정하기 때문에 cost가 매우 큼
				2. 
					- 위의 방법 대신 activation을 linear approximate 할 수 있음
						- ![equation](https://latex.codecogs.com/gif.latex?y%3Dax&plus;b%2Ca%3D%5Cfrac%20%7B%20%5CDelta%20y%20%7D%7B%20%5CDelta%20x%20%7D)
					- activation에 대한 contribution은 ![equation](https://latex.codecogs.com/gif.latex?%5Cleft%7C%20%5Csum%20_%7B%20c%20%7D%5E%7B%20%7D%7B%20%28x%5Ccirc%20%5Cfrac%20%7B%20%5CDelta%20%7B%20a%20%7D_%7B%20i%20%7D%28x%29%20%7D%7B%20%5CDelta%20x%20%7D%20%29%20%7D%20%5Cright%7C)로 측정할 수 있음
					- 절대값을 이용하는 이유는 어떤 방향이든(activation이 커지는/작아지는) 영향력의 크기를 확인하기 위함임
					- 이 때, 특정 pixel을 0으로 만들어서 activation을 크게하는 operation을 수행할 수 있지만, 이와 관련된 것은 gradient aescent에서 이미 수행하는 것이므로 이 단계에서는 수행하지 않음.
					- 영향력의 크기가 threshold인 ![equation](https://latex.codecogs.com/gif.latex?%7B%20%5Ctheta%20%7D_%7B%20c%5C_%20pct%20%7D) 보다 작은 pixel은 0으로 만들어줌
		- 위의 4가지 regularizer는 개별적으로(individually) 수행되었을 때 효과가 있지만 특히, 같이(combined) 적용할 경우 효과가 더 커짐. 이 점이 이전의 연구와 우리의 연구의 차이점임.
	- combined hyperparamter를 고르기 위해 random search를 이용
- Discussion and Conclusion
	- live visualization software를 통해 상위 layer에서 하나의 layer 자체가 complex feature를 detect 함을 확인할 수 있었음(여러 layer에 distributed되지 않고)(단, 모든 layer가 그렇다는 것은 아님) -> 마지막 쪽의 FC layer만 조정해서 transfer learning이 가능함
	- combination of regularizer를 통해
		- 개별적으로 효과가 있는 것들을 combine해서 더 좋은 결과가 나올 수 있음
		- 기존에 discriminative network는 input을 generate하기에 적절하지 않다는 연구가 많았지만, 이는 prior(modeling)이 weak 해서 그랬던 것 같음(우리의 경우, discriminative network에서 충분히 recognizable한 image generate에 성공하였음)
		- 즉 discriminative에서 non-discriminative information을 무시하는 것이 아니라 여전히 hidden layer 어딘가에 해당 정보를 contain하고 있음
- supplementary
	- http://yosinski.com/deepvis 참고
		- 'natural image prior' == 'regularization' : activation을 maximize하는 동시에 natural image와 비슷하게 보이려는 노력을 통칭함
		- CNN을 크게 conv layer과 FC layer로 나누어서 생각하면, conv layer에서 local connectivity를 통해 simple feature를 detect하고 top layer쪽의 FC layer에서 feature를 combine하여 complex and abstract representation을 learn한다고 생각할 수 있음
		- image classification 과정에서 의도하지 않았던(명시적 학습 목표가 아닌) feature를 학습함(ex)bowtie classification을 위해 face를 학습함, bowtie는 주로 face와 함께 등장하기 때문에)
		- 특정 neuron 1개가 특정 feature(ex) face, ear 등)을 학습할 수 있음(이는 feature가 여러 neuron에 distribute 되어 있어서 single neuron으로는 아무것도 확인할 수 없다는 주장과 상반됨)
		- 지금까지는 discriminative DNN이 non-discriminative한 feature를 무시한다고 생각해왔었는데, 이 실험결과에 따르면 그렇지 않다(hidden layer에서 classification과 직접 관련되지 않은 feature도 capture한다). 그러면 personal robot 등도 train 과정에서 그럴 수 있고, train 과정에서 input으로 사용 되는 image의 어떤 feature가 의도치 않게 hidden layer에서 capture 될 수 있다는 것을 의미한다. 즉 robot 청소기가 청소를 위해 input image로 train을 할 때, 의도치 않게 내 모습을 train할 수 있고, 이는 privacy problem과 연결될 수 있다.

##### norm
- vecotr의 길이나 크기(magnitude)를 측정하는 방법 -> **loss를 vectorize한다면 loss에 사용할 수 있고, weight도 vector로 생각할 수 있기 때문에 weight normalization에도 사용 가능**
- ![equation](https://latex.codecogs.com/gif.latex?%7B%20L%20%7D_%7B%20p%20%7D%3D%7B%20%28%5Csum%20_%7B%20i%20%7D%5E%7B%20n%20%7D%7B%20%7B%20%5Cleft%7C%20%7B%20x%20%7D_%7B%20i%20%7D%20%5Cright%7C%20%7D%5E%7B%20p%20%7D%20%7D%20%29%20%7D%5E%7B%20%5Cfrac%20%7B%201%20%7D%7B%20p%20%7D%20%7D)
- p를 차수라고 하고 L1, L2 norm을 주로 사용
- ![equation](https://latex.codecogs.com/gif.latex?%7B%20L%20%7D_%7B%201%20%7D%3D%5Cleft%7C%20x_%7B%201%20%7D%20%5Cright%7C%20&plus;%5Cleft%7C%20%7B%20x%20%7D_%7B%202%20%7D%20%5Cright%7C%20&plus;...%5Cleft%7C%20%7B%20x%20%7D_%7B%20n%20%7D%20%5Cright%7C)
- ![equation](https://latex.codecogs.com/gif.latex?%7B%20L%20%7D_%7B%202%20%7D%3D%5Csqrt%20%7B%20%7B%20%7B%20x%20%7D_%7B%201%20%7D%20%7D%5E%7B%202%20%7D&plus;%7B%20%7B%20x%20%7D_%7B%202%20%7D%20%7D%5E%7B%202%20%7D&plus;...%7B%20%7B%20x%20%7D_%7B%20n%20%7D%20%7D%5E%7B%202%20%7D%20%7D)
- 참고 http://taewan.kim/post/norm/

##### bucketing / binning
- continuous value를 bucket 단위의 discrete 단위로 변환하는 행위
- ex) 0~10도 사이의 온도는 0.1도의 민감도(sensitivity)에서 각각 (0~0.1), (0.1~0.2) .. (0.9~10)의 100가지 bucket으로 변환할 수 있다

##### confusion matrix
- classification의 예측 성공률을 요약한 표
- class label이 N개일 경우 NxN 행렬로 나타낼 수 있음(axis1=label, axis2=prediction)

##### embedding
- continuous value로 표현된 categorical data
- 일반적으로 고차원 vector를 저차원 vector로 mapping 시키는 것을 의미함

##### deep leanring vs machine learning
- machine learning
	- logic based algorithm이 아니라 data based algorithm 즉 data로 부터 learning하는 방법
- deep learning
	- machine learning의 한 분야로서 deep neural network를 이용하는 방법
- **주요 차이**
	- machine learning에서는 data의 feature를 hard-coding(hand-written)함, 즉 직접 알아내야함
	- deep learning에서는 data의 feature를 hidden layer를 통해 automatically 알아냄

##### About correlation
- Only numeric features can have correaltion(Alpahbet, String cannot have correaltion normarlly)
- If increase in feature A leads to increase in feature B, then they are positively correlated.
- Value 1 means perfect positive correaltion
- If increase in feature A leads to decrease in feature B, then they are negatively correlated.
- Value -1 means perfect negative correlation
- If correlation between A and B are about 1, then increase in A leads to increase in B. This means that A and B are very similar feature

##### 기타 정보
- https://blog.lunit.io/2018/08/03/batch-size-in-deep-learning/ -> learning rate와 batch size의 적절한 조합을 잘 찾아야함 -> 최적 hyperparameter조합을 잘 찾는게 매우 중요함, batch size도 '잘' 정해야 하는 요소인데, 작은 경우 좋은 점이 있음(실험 결과적으로 안정적인 training 가능)
- numpy는 매 실행마다 해당 operation에 대한 정보만 있지만 tensorflow는 computational graph 전체에 대한 정보가 있어서 일반적으로 더 빠름
- tf.truncated_normal -> tf.initializers.he_normal()로 initializer를 바꿨더니 학습이 급격하게 잘됨
- dropout 추가했더니 학습이 급격하게 잘됨
