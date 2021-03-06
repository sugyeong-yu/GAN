# Chapter 4. GAN
## 1. 기본 GAN
----------------------------
### 생성자(genarater)와 판별자(discriminator)
 >  - 생성자 : 랜덥한 잡음을 원본 데이터셋에서 샘플링한 것처럼 보이는 샘플로 변환
 >  - 판별자 : 원본 데이터셋에서 추출한 샘플인지 생성자가 만든 가짜인지를 구별
 >  - 생성자가 판별자를 속이는데 더 능숙해지고 판별자가 가짜샘플을 정확하게 구별하는 능력을 유지하도록 학습됨.
### DB-구글의 Quick, Draw! data set
 > - 28*28픽셀크기의 흑백낙서이미지로 주제별로 레이블되어있음.
 > - 온라인 게임으로부터 수집된 데이터셋.
 > - 어떤 물체나 개념을 그림으로 그리고 신경망이 이 낙서의 제목을 추측한다.
 > - data dir에 있는 camel numpy파일을 다운로드.
### Model
 > Models dir 안에 정의해둔 GAN.py에 정의되어 있음.
 > ```
 > from models.GAN import GAN
 > ```
### 판별자
 > - 판별자의 목표는 이미지가 진짜인지 가짜인지 예측하는 것이다.
 > - 이는 지도학습에서 이미지 분류 문제이다. 따라서 합성곱 층을 쌓고 완전 연걸 층을 출력층으로 놓은 네트워크 구조를 사용할 수 있다.
 > - DCGAN\ 
 > : GAN의 원본논문에서는 합성곱 층 대신 fc 층을 사용했다. 하지만 합성곱층을 사용했을때 성능이 더 우수한것으로 밝혀졌다. 이에 conv layer를 사용한 GAN을 DCGAN이라 하였지만 현재는 대부분 conv layer를 사용하므로 GAN은 DCGAN이랑 동일한 의미를 가진다.
 > - 예제에서는 단순히 만들기위해 배치정규화를 수행하지 않았다.
 > - gan의 판별자(discriminator)의 구조와 코드\
 > ![image](https://user-images.githubusercontent.com/70633080/105457101-9d2bb000-5cc9-11eb-80e2-5421b0c239a1.png)
 > ```
 > discriminator_input = Input(shape=self.input_dim, name='discriminator_input')
 >
 >       x = discriminator_input

 >       for i in range(self.n_layers_discriminator):
 >
 >           x = Conv2D(
 >               filters = self.discriminator_conv_filters[i]
 >               , kernel_size = self.discriminator_conv_kernel_size[i]
 >               , strides = self.discriminator_conv_strides[i]
 >               , padding = 'same'
 >               , name = 'discriminator_conv_' + str(i)
 >               , kernel_initializer = self.weight_init
 >               )(x)
 >
 >           if self.discriminator_batch_norm_momentum and i > 0:
 >               x = BatchNormalization(momentum = self.discriminator_batch_norm_momentum)>>(x)
 >
 >           x = self.get_activation(self.discriminator_activation)(x)
 >
 >           if self.discriminator_dropout_rate:
 >               x = Dropout(rate = self.discriminator_dropout_rate)(x)
 >
 >       x = Flatten()(x)
 >       
 >       discriminator_output = Dense(1, activation='sigmoid', kernel_initializer = self.weight_init)(x)
 >
 >       self.discriminator = Model(discriminator_input, discriminator_output) # 케라스모델로 판별자를 정의 , 이미지를 입력받아 0과 1사이의 숫자하나를 출력
 > ```
 
 > - 일부 합성곱 층에 stride=2를 사용해 tensor size를 줄였지만 channel의 수는 증가시켰다.
 > - 마지막 층의 sigmoid 활성화 함수는 출력을 0~1사이값으로 만듬. 이는 이미지에 대해 진짜일 예측확률
### 생성자
 > - 입력: 다변수 표준 정규 분포에서 추출한 벡터
 > - 출력: 원본 훈련 데이터의 이미지와 동일한 크기의 이미지
 > - GAN의 생성자는 VAE의 디코더와 동일한 목적을 가지고 수행한다. 따라서 유사하다.
 > - 잠재공간의 벡터를 이미지로 변환하는 것.
 >   - 잠재공간의 벡터를 조작하여 **원본차원에 있는 이미지의 고수준 특성을 바꿀 수 있다.**  
 > #### upsampling
 > > - VAE에서 사용한 Conv2DTranspose 클래스는 합성곱 연산을 수행하기 전에 픽셀사이에 0을 추가한다.
 > > - GAN에서는 케라스의 UPSampling2D 층을 사용해 텐서의 너비와 높이를 두배로 늘린다.
 > > - 단순히 입력의 각 행과 열을 반복하여 크기를 두배로 만든다.
 > > - 그다음 stride=1인 합성곱 층을 사용해 합성곱 연산을 수행한다.
 > > - 이는 전치합성곱과 비슷하지만 0이아닌 기존 픽셀값을 사용해 upsampling한다.
 > > - 두 방식 (upsampling+conv / conv2dtranspose) 모두 원본이미지차원으로 돌릴때 사용할 수 있는 변환방법이다.
 > >   - upsampling2d와 conv2dtranspose 방법을 모두 test해서 어떤것이 더 나은 결과를 만드는지 확인해야한다.
 > > - Conv2dTransepose는 출력 이미지 경계에서 계단모양이나 체크무늬 패턴을 만들수 있어 출력품질을 떨어 뜨릴 수도 있다. 하지만 여전히 많이 사용되고 있는 방법이다.
 > > 1. nearest : 가장 가까운 픽셀값으로 채우는 방식.
 > > 2. bilinear : 코너값을 기준으로 간격에 따라서 채우는 방식 
 > > 3. bilinear, align_corners=True : True로 설정되면 Input점 의 edge(corner)가 Output의 edge(corner)와 정렬을 맞춘 상태에서 interpolation을 합니다. 반면 False 인 상태이면 algin을 맞추지 않은 상태로 inpterpolation을 하게됩니다. 간단하게 말하면 align_corners = True인 상태에 값들이 더 넓게 펼쳐져 있습니다.
 > > <img src="https://user-images.githubusercontent.com/70633080/105463154-fe0bb600-5cd2-11eb-906a-d13b261566a4.png" width=50% height=50%>
 > - 아래그림은 생성자의 구조이다.\
 > ![image](https://user-images.githubusercontent.com/70633080/105461497-a704e180-5cd0-11eb-89c1-6e5c559958e8.png)
 > ```
 > generator_input = Input(shape=(self.z_dim,), name='generator_input')
 >
 >       x = generator_input
 >
 >       x = Dense(np.prod(self.generator_initial_dense_layer_size), kernel_initializer =  self.weight_init)(x)
 >
 >       if self.generator_batch_norm_momentum:
 >           x = BatchNormalization(momentum = self.generator_batch_norm_momentum)(x)
 >
 >       x = self.get_activation(self.generator_activation)(x)
 >
 >       x = Reshape(self.generator_initial_dense_layer_size)(x)
 >
 >       if self.generator_dropout_rate:
 >           x = Dropout(rate = self.generator_dropout_rate)(x)
 >
 >       for i in range(self.n_layers_generator):
 >
 >           if self.generator_upsample[i] == 2:
 >               x = UpSampling2D()(x)
 >               x = Conv2D(
 >                   filters = self.generator_conv_filters[i]
 >                   , kernel_size = self.generator_conv_kernel_size[i]
 >                   , padding = 'same'
 >                   , name = 'generator_conv_' + str(i)
 >                   , kernel_initializer = self.weight_init
 >               )(x)
 >           else:
 >
 >               x = Conv2DTranspose(
 >                   filters = self.generator_conv_filters[i]
 >                   , kernel_size = self.generator_conv_kernel_size[i]
 >                   , padding = 'same'
 >                   , strides = self.generator_conv_strides[i]
 >                   , name = 'generator_conv_' + str(i)
 >                   , kernel_initializer = self.weight_init
 >                   )(x)
 >
 >           if i < self.n_layers_generator - 1:
 >
 >               if self.generator_batch_norm_momentum:
 >                   x = BatchNormalization(momentum = self.generator_batch_norm_momentum)(x)
 >
 >               x = self.get_activation(self.generator_activation)(x)
 >                   
 >               
 >           else:
 >
 >               x = Activation('tanh')(x)
 >
 >
 >       generator_output = x
 >
 >       self.generator = Model(generator_input, generator_output)
 > ```
 > 1. 길이가 100인 벡터로 생성자의 입력을 정의한다.
 > 2. 3,136개의 유닛을 가진 Dense층을 놓는다.
 > 3. 배치정규화와 RELU활성화 함수를 적용시켜 7*7*64 텐서로 바꾼다.
 > 4. 네개의 CONV2D층을 통과시킨다. 처음 2개는 Upsampling2D 층 뒤에 놓인다. 마지막 층을 제외하고 나머지 층에는 배치정규화와 RELU함수를 사용한다. 
 > 5. tang활성화함수를 이용해 출력을 원본이미지와 같은 [-1,1]범위로 변환한다.
 > 6. 케라스 모델로 생성자를 정의한다. 이 모델은 길이 100의 벡터를 받아 [28,28,1]크기의 텐서를 출력한다.
 
### 훈련
 > - 진짜 이미지 = 1, 생성된 이미지 = 0
 > - 판별자는 진짜이미지에 가까울 수록 1에 가까운 숫자를 출력하게 됨.
 > #### 판별자 훈련 과정
 > - 입력 : 훈련세트에서 랜덤하게 선택한 진짜 샘플 + 생성자의 출력
 >   - 진짜 이미지 : 1
 >   - 생성된 이미지 : 0
 > - 출력 : 진짜이미지에 대해서는 1에 가까운 값, 가짜이미지에 대해서는 0에 가까운 값\
 > 원본이미지와 생성된 이미지의 차이점을 구분할 수 있도록 판별자를 훈련시킬수있다.
 >
 > #### 생성자 훈련 과정
 > 진짜 이미지가 잠재공간의 어떤포인트에 매핑되는지 알려주는 훈련세트가 없다.  > 대신 판별자를 속이는 이미지를 생성한다. 
 > - 판별자와 연결한 케라스 모델,즉 생성자 모델을 만들어야 한다. 
 > - 입력 : 랜덤하게 생성한 100차원 잠재공간벡터
 > - 출력 : 판별자의 출력, (판별자가 진짜라고 생각할수있는 이미지를 생성자가 만드는 것이 목적)
 > - 손실함수 : 이진크로스엔트로피 , 판별자의 출력과 타깃(1) 사이의 손실
 > - 전체 모델을 훈련할 때 생성자의 가중치만 업데이트 되도록 **판별자의 가중치를 동결하는게 중요하다.**
 >   - 판별자의 가중치를 동결하지 않으면 생성된 이미지를 진짜라고 여기도록 조정되기 때문.
 >
 > - 아래 그림은 GAN의 훈련과정을 나타낸 그림이다.\
 > ![image](https://user-images.githubusercontent.com/70633080/105466222-462cd780-5cd7-11eb-89d2-0f9d4b14f2b9.png)
 > - GAN훈련하기
 > - 판별자 모델을 컴파일하고 생성자를 훈련할 모델을 컴파일하는 코드
 > ```
 > ### def _build_adversarial(self):
 >      COMPILE DISCRIMINATOR
 >
 >       self.discriminator.compile(
 >       optimizer=self.get_opti(self.discriminator_learning_rate)  
 >       , loss = 'binary_crossentropy'
 >       ,  metrics = ['accuracy']
 >       ) # 이진분류이므로 bce를 사용한다.
 >       
 >       ### COMPILE THE FULL GAN
 >
 >       self.set_trainable(self.discriminator, False) # 판별자의 가중치 동결
 >
 >       model_input = Input(shape=(self.z_dim,), name='model_input')
 >       model_output = self.discriminator(self.generator(model_input)) # 생성자의 출력->판별자의 입력
 >       self.model = Model(model_input, model_output) # 생성자 모델 정의 
 >       
 >       #  일반적인 생성자보다 판별자가 더 강해야하므로 학습률이 판별자보다 느리다.
 >       self.model.compile(optimizer=self.get_opti(self.generator_learning_rate) ,  loss='binary_crossentropy', metrics=['accuracy']) # 전체 모델 컴파일
 > ```
 > 
 > - 판별자 훈련코드
 > ```
 > def train_discriminator(self, x_train, batch_size, using_generator):
 >       # 판별자 훈련하기
 >       valid = np.ones((batch_size,1)) # 진짜이미지 레이블=1
 >       fake = np.zeros((batch_size,1)) # 가짜이미지 레이블 =0
 >
 >       if using_generator:
 >           true_imgs = next(x_train)[0]
 >           if true_imgs.shape[0] != batch_size:
 >               true_imgs = next(x_train)[0]
 >       else:
 >           # 진짜 이미지로 훈련
 >           idx = np.random.randint(0, x_train.shape[0], batch_size) # 0, 8000 중 batch_size만큼만 추출
 >           true_imgs = x_train[idx]
 >       #생성된 이미지로 훈련
 >       noise = np.random.normal(0, 1, (batch_size, self.z_dim)) # 정규분포로 평균0 과 표준편차1로 (batch_size,100) size로 생성
 >       gen_imgs = self.generator.predict(noise)
 >       # 훈련하여 loss랑 acc 구하기
 >       d_loss_real, d_acc_real =   self.discriminator.train_on_batch(true_imgs, valid)
 >       d_loss_fake, d_acc_fake =   self.discriminator.train_on_batch(gen_imgs, fake)
 >       d_loss =  0.5 * (d_loss_real + d_loss_fake)
 >       d_acc = 0.5 * (d_acc_real + d_acc_fake)
 >
 >       return [d_loss, d_loss_real, d_loss_fake, d_acc, d_acc_real, d_acc_fake]
 > ```
 > - 생성자 훈련코드
 > ```
 >    def train_generator(self, batch_size):
 >       valid = np.ones((batch_size,1))
 >       noise = np.random.normal(0, 1, (batch_size, self.z_dim))
 >       return self.model.train_on_batch(noise, valid) # 생성자 훈련에서 판별자는 동결되어 가중치가 변하지 않음
 > ```
 > - 실제로는 판별자와 생성자를 번갈아가면서 훈련한다.
 > ```
 > epochs=2000
 > batch_size=64
 > for epoch in range(epochs):
 >   train_discriminator(x_train,batch_size)
 >   train_generator(batch_size)
 > ```
- 04_01_gan_camel_train.ipynb 에서 결과확인
### GAN에서의 문제점
 > #### 진동하는 손실
 > 판별자와 생성자의 손실이 장기간 안정된 모습을 보여주지 못하고 큰 폭으로 진동하기 시작할 수 있다. 그런데 장기간으로 심하게 출렁거리는 것이 아닌 정상적으로는 손실이 안정되거나 점진적으로 증가하거나 감소하는 형태를 보여야한다.
 
 > #### 모드붕괴
 > 이는 생성자가 판별자를 속이는 **적은 수의 샘플을 찾을 때 일어난다.** 즉, 판별자를 잘 속이지 못할 때 발생한다. 따라서 생성자는 한정된 샘플 이외에는 다른 샘플을 생성하지 못한다.
 > - 모드 = **판별자를 항상 속이는 하나의 샘플**을 뜻함.
 > - 모드붕괴란?
 > 판별자의 가중치를 업데이트 하지않고 몇번의 배치 동안 생성자를 훈련할 경우 생성자는 모드를 찾으려는 경향이 있고 잠재공간의 모든 포인트를 이 샘플에 매핑할 수 있다. 이는 즉 '손실함수의 gradient가 0에 가까운 값으로 무너진다' 는 뜻이다.
 > - 잘 맞는 샘플하나만 찾으려고 한다는 것. 
 > - 하나의 포인트에 속지 못하도록 판별자를 훈련시켜도 생성자는 판별자를 속이는 모드를 쉽게 찾을 것이다. 생성자가 이미 입력에 무감각해져 다양한 출력을 만들필요가 없기 때문.\
 > ![image](https://user-images.githubusercontent.com/70633080/105479505-e12dad80-5ce7-11eb-84e0-7cf9503fe394.png)
 > #### 유용하지 않은 손실
 > 생성자의 손실이 작을수록 생성이미지의 품질이 더 좋을 것이다? \
 > 그렇지 않다. 생성자는 현재 판별자에 의해서만 평가되는데 판별자는 점점 향상되므로 훈련과정의 다른지점에서 평가된 손실을 비교할 수 없다.\
 > 따라서 학습이 진행될 수록 이미지품질은 향상되어도 생성자의 손실함수는 증가한다. 생성자의 손실과 이미지품질사이의 연관성부족은 훈련과정 모니터링을 어렵게 만든다.
 >
 > #### GAN의 한계점 해결
 > - WGAN과 WGAN-GP를 살펴본다.
 > - 이 둘은 앞에서의 구조를 조금 바꾼것으로 모델의 전체적인 안전성을 크게 향상시키고 모드붕괴와 같은 문제점의 발생가능성을 줄였다.
 --------------------------------
## 2. WGAN(와서 스테인 GAN)
-----------------------------------
WGAN은 안정적인 GAN훈련을 위한 첫번째 발전 중 하나이다. 
- **생성자가 수렴하는 것 과 샘플의 품질을 연관 짓는 의미있는 손실 측정방법**
  - 새로운 손실함수를 소개한다. 이진크로스엔트로피 대신 이 손실함수를 사용할 경우 더 안정적으로 수렴할 수 있다는 것이다.
- 최적화 과정의 안정성 향상
- WGAN의 구조 그림\
![image](https://user-images.githubusercontent.com/70633080/105580183-2fb57780-5dce-11eb-9ef2-d3809aad340f.png)
### 와서스테인 손실
 > #### 이진크로스엔트로피
 > ![이진크로스엔트로피 손실함수](https://user-images.githubusercontent.com/70633080/100074879-9d9efa80-2e82-11eb-91ba-a27e855e4970.PNG)
 > - 진짜이미지에 대한 예측과 타깃 y=1를 비교하고, 생성된 이미지에 대한 예측과 타깃 y=0을 비교해 손실을 계산한다.
 > #### Gan 판별자, 생성자 손실 최소 최대화
 > - GAN판별자 손실 최소화\
 > : 생성자 G를 훈련하기 위해 **생성된 이미지에 대한 예측 p=D(G(Z))와 타깃 y=1을 비교해 손실을 계산**한다. 
 > ![GAN판별자 손실 최소화 함수](https://user-images.githubusercontent.com/70633080/100078201-9c6fcc80-2e86-11eb-922e-a6eafb14c361.jpg)
 > - GAN생성자 손실 최소화
 > ![GAN생성자 손실 최소화 함수](https://user-images.githubusercontent.com/70633080/100078206-9f6abd00-2e86-11eb-9534-27d8e6428031.jpg)
 >
 > #### 와서스테인 손실 함수
 > - 와서스테인 손실은 1과 0대신 y=1, y=-1을 사용한다. 
 > - 또한 판별자의 마지막 층에서 sigmoid 활성화함수를 제거하여 예측 p가 [0,1]범위에 국한되지 않고 [-무한대, +무한대] 범위의 어떤 숫자도 될 수 있도록 만든다. 
 > - 따라서 WGAN의 판별자는 보통 비평자(critic)이라 부른다.
 > - WGAN의 비평자는 진짜이미지에 대한 점수를 높임으로써 **진짜이미지와 생성된 이미지에 대한 예측사이의 차이를 최대화한다.**
 > - WGAN은 이진 크로스엔트로피 대신 더 작은 학습률을 사용하는 경우가 많다.
 > #### 와서스테인 손실 최소화
 > - WGAN 비평자 손실 최소화
 >   - WGAN의 비평자 D를 훈련하기 위해 **진짜 이미지에 대한 예측 p(i)=D(xi)와 타깃 y=1**을 비교하고
 >   - **생성된 이미지에 대한 예측 p(i)=D(G(Zi))와 타깃 y=-1**을 비교해 손실을 계산한다.\
 > ![WGAN 비평자 손실 최소화함수](https://user-images.githubusercontent.com/70633080/100312307-538a5600-2ff5-11eb-824f-16eda08f00df.png)
 > - WGAN 생성자 손실 최소화
 >   - WGAN의 생성자를 훈련하려면 **생성된 이미지에 대한 예측 p(i)=D(G(zi))와 타깃 y=1**을 비교하여 손실을 계산한다. 
 > ![WGAN 생성자 손실 최소화함수](https://user-images.githubusercontent.com/70633080/100312420-9ea46900-2ff5-11eb-9ee1-bcdafbac7933.png)
 > ```
 > model.compile(optimizer=RMSprop(lr=0.00005),
 >                loss=wasserstein
 >                )
 > ```
 > #### 정리
 > - WGAN은 기존의 GAN과 아래와 같은 차이점을 둔다.
 > 1. discriminator 대신 새로 정의한 critic을 사용한다.
 >   1. discriminator : 가짜/진짜를 판별하기 위해 sigmoid를 사용하고 output은 가짜/진짜에 대한 확률 값이다.
 >   2. critic : EM(Earth Mover) distance로 얻은 scalar값을 이용한다.
 > 2. EM distance : 확률 분포 간의 거리를 측정하는 척도 중 하나이다. 
 > - 아래와 같은 장점을 가진다.
 > 1. gradient를 잘 전달
 > 2. critic과 generator를 최적점까지 학습 할 수 있다.
 > 3. traing시 discriminator과 generator간의 balance를 주의깊게 살피지 않아도 된다.
 > 4. GAN에서 일반적으로 생성되는 모드붕괴를 해결 가능하다.
 >
 > #### EM Distance(Earth Mover's Distance)를 사용한 이유
 > ![image](https://user-images.githubusercontent.com/70633080/105579805-17445d80-5dcc-11eb-8530-87ec53707d1d.png)
 > - 두 확률분포간의 거리를 나타내는 지표
 > 1. Total Variation(TV)\
 > ![image](https://user-images.githubusercontent.com/70633080/105579824-3b07a380-5dcc-11eb-9d63-b615473d0c48.png)
 > 2. Kullback-Leibler(KL) Divergence\
 > ![image](https://user-images.githubusercontent.com/70633080/105579836-4bb81980-5dcc-11eb-9dfc-a99375606297.png)
 > 3. Jensen-Shannon(JS) Divergence\
 > ![image](https://user-images.githubusercontent.com/70633080/105579852-5b376280-5dcc-11eb-9c13-458982d3839c.png)
 > 4. Earth Mover's Distance\
 > ![image](https://user-images.githubusercontent.com/70633080/105579859-68545180-5dcc-11eb-9926-7b2a9b1fc758.png)
 > - 1,2,3 번의 거리지표는 추정하는 모수에 따라 거리가 달라질 뿐만 아니라 그 값이 상수 또는 무한대의 값을 가지게 된다.
 > - TV,KL/JS Divergence는 두 분포가 서로 겹치는 경우에 0, 겹치지 않는 경우에는 무한대 또는 상수로 극단적인 값을 가지게된다.
 > - 결론적으로 TV,KL/JS Divergence을 loss로 사용한다면 gradient가 제대로 전달되지 않아 학습이 어려워진다.
 > - 최종적으로 사용하는 Wasserstein loss\
 > ![image](https://user-images.githubusercontent.com/70633080/105579980-047e5880-5dcd-11eb-8c0d-466ff2a65565.png)
 
### 립시츠제약
 > 시그모이드를 사용하지 않아 [-무한대, +무한대] 범위의 값을 출력하기 때문에 손실값이 커잘 수있다. 이를 방지하기 위해 손실함수에 추가적인 제약이 필요하다.
 > - 비평자 C(판별자 D)는 1-립시츠 연속함수여야한다.
 > - 다음 식이 부등식을 만족할 때 이 함수를 1-립시츠 연속함수 라고 부른다.\
 > ![image](https://user-images.githubusercontent.com/70633080/100313726-91d54480-2ff8-11eb-9896-0f9b015c4ecc.png)\
 > 립시츠 함수는 임의의 두 지점의 기울기가 어떤 상숫값(1) 이상 증가하지 않는 함수이다. 따라서 기울기의 절댓값의 최대는 1이다.
 > - |D(x1)-D(x2)|는 비평자 예측간의 절댓값 차이.\
 >   - 이것은 비평자의 예측이 변화 할 수 있는 비율을 제한한것이다. 
 >   - 즉, 기울기의 절댓값이 어디서나 최대 1이어야한다는 뜻. 
 >   - 다른말로 하면 이직선은 어느 지점에서나 상승하거나 하강하는 비율이 한정되어 있다는 것이다.
 > - 립시트 연속함수\
 > : 그래프 어디에 원뿔을 놓더라도 이 함수는 항상 원뿔 바깥쪽에 놓인다. = 아래직선은 어디서 상승하거나 하강하는 비율이 정해져있다.\
 > ![image](https://user-images.githubusercontent.com/70633080/105480739-77160800-5ce9-11eb-9544-ad2dc38b8801.png)
 > - 가중치 클리핑을 통해 립시츠 제약을 부과할 수 있다.
 > - 이런 제약이 부과될때만 와서스테인 손실이 작동하는 이유에 대해 잘 설명해놓은 http://bit.ly/2MwS8rc 를 참고.
 
 
### 가중치 클리핑
 > - WGAN 논문 저자는 훈련 배치가 끝난 후 가중치 클리핑을 통해 립시츠 제약을 부과하는 방법을 보였다. 
 > - 판별자의 가중치를 [-0.01,0.01] 안에 놓이도록 조정한것.
 > - 이는 다양한 weight를 좁은 범위로 클리핑하게되면 다양성을 잃어버리고 학습이 오래걸릴 수 있다.
 > - models dir 안 WGAN.py
 > ```
 > def train_critic(self, x_train, batch_size, clip_threshold, using_generator):
 >
 >       valid = np.ones((batch_size,1))
 >       fake = -np.ones((batch_size,1))
 >       # 진짜 이미지 
 >       if using_generator:
 >           true_imgs = next(x_train)[0]
 >           if true_imgs.shape[0] != batch_size:
 >               true_imgs = next(x_train)[0]
 >       else:
 >           idx = np.random.randint(0, x_train.shape[0], batch_size)
 >           true_imgs = x_train[idx]
 >       
 >       # 생성된 이미지
 >       noise = np.random.normal(0, 1, (batch_size, self.z_dim))
 >       gen_imgs = self.generator.predict(noise)
 >       # 진짜와 생성된 이미지로의 훈련
 >       d_loss_real =   self.critic.train_on_batch(true_imgs, valid)
 >       d_loss_fake =   self.critic.train_on_batch(gen_imgs, fake)
 >       d_loss = 0.5 * (d_loss_real + d_loss_fake)
 >
 >       for l in self.critic.layers:
 >           weights = l.get_weights()
 >           weights = [np.clip(w, -clip_threshold, clip_threshold) for w in weights]
 >           l.set_weights(weights)
 > ```
### WGAN훈련
 > - 기본 GAN은 gradient vanishing을 피하기 위해 **판별자가 너무  강해지지 않도록 하는 것이 중요**하다.\
 > - WGAN은 GAN의 훈련 어려움 중 하나를 제거할 수 있다. 
 > - 바로 판별자와 생성자의 훈련균형을 맞추는 것이다. 
 >   - WGAN은 생성자를 업데이트하는 중간에 판별자를 여러번 훈련하여 수렴에 가깝게 할 수 있다. 일반적으로 **생성자를 1번 업데이트할때 판별자를 5번 업데이트** 할 수 있다.
 > ```
 > for each in range(epochs):
 >    for _ in range(5) :
 >        train_critic(x_train,batch_size=128., clip_threshold=0.01)
 >    train_generator(batch_size)
 > ```
 > - 기본 GAN과 WGAN의 차이점 요약
 > 1. wgan은 와서스테인손실을 사용.
 > 2. wgan은 진짜는 레이블 1, 가짜는 레이블 -1을 사용
 > 3. wgan 비평자의 마지막 층에는 시그모이드가 필요하지 않다.
 > 4. 매 업데이트 후 판별자의 가중치를 클리핑한다.
 > 5. 생성자를 업데이트할 때마다 판별자를 여러번 훈련한다.
- WGAN분석으로 04_02_wgan_cifar_train.ipynb 참고.
-----------------------
## 3. WGAN-GP
----------------------
 ### wasserstein GAN-Gradient penalty
 > - 이는 비평자의 가중치 클리핑이 undesired behavior을 유발할 수 있는 주장에서 제안된 방법이다.
 > - gradient penalty를 통해 같은문제로 부터 시달리지 않을 것이다 라는것.
 > - 다양한 GAN구조에 대해 안정적인 학습, weight clipping에 대한 성능향상, 고품질이미지생성, 개별샘플링 없이 문자 수준의 GAN언어모델을 선보인다.
 > - WGAN-GP 생성자는 WGAN 생성자와 정확히 같은 방식으로 정의하고 컴파일 한다. 바꾸어야 할 것은 **비평자의 정의와 컴파일 단계**이다.
 > - WGAN 비평자 -> WGAN-GP 비평자
 > > 1. 비평자 손실 함수에 그레디언트 패널티 항을 포함한다.
 > > 2. 비평자의 가중치를 클리핑 하지 않는다.
 > > 3. 비평자에 배치정규화 층을 사용하지 않는다.
### 그레디언트 패널티 손실
 > 기존의 판별자 훈련과 비교했을때 진짜이미지와 가짜이미지에 대한 와서스테인손실과 더불어 그레디언트 패널티 손실이 추가되어 전체손실함수를 구성하고 있는 것을 알 수 있다.
 > ![wgan-gp 비평자 훈련과정](https://user-images.githubusercontent.com/70633080/100320575-b6371e00-3004-11eb-8920-656c8a737b81.png)
 > 그레디언트 패널티 손실은 입력 이미지에 대한 예측의 gradient norm과 1사이의 차이를 제곱한 것이다.\
 > 따라서 그레디언트 패널티 항을 최소화하는 가중치를 찾으려고 할 것이며 모델이 립시츠제약을 따르도록 한다.
 > - 일부 지점에서만 그레디언트를 계산한다. 모든 곳에서 계산하기는 힘들기 때문이다.
 > - 한쪽에 치우치지 않기 위해 진짜 이미지와 가짜 이미지를 연결한 직선을 따라 무작위로 포인트를 선택해 보간한 이미지를 사용한다.
 > - models dir 안 WGANGP.py 파일 RandomWeightedAverage층 
 > > merge_function 층에서 진짜 이미지와 가짜 이미지의 쌍을 연결하는 직선위에 놓인 픽셀 기준의 보간된 이미지를 반환한다.\
 > > gradient_panalty_loss함수는 보간된 포인트에서 계산된 그레디언트와 1사이의 차이를 제곱하여 반환한다. ( 기울기가 1에 가까우면서 클리핑을 방지하는 역할)
 > ```
 > def _build_adversarial(self):           
 >       #-------------------------------
 >       # Construct Computational Graph
 >       #       for the Critic
 >       #-------------------------------
 >
 >       # Freeze generator's layers while training critic
 >       self.set_trainable(self.generator, False) # 생성자의 가중치를 동결함. 보간된 이미지가 손실함수에 관여하므로 생성자는 비평자 훈련을 위한 일부
 >
 >       # Image input (real sample)
 >       real_img = Input(shape=self.input_dim)
 >
 >       # Fake image
 >       z_disc = Input(shape=(self.z_dim,))
 >       fake_img = self.generator(z_disc)
 >
 >       # critic determines validity of the real and fake images
 >       # 와서스테인 손실을 계산하기 위해 진짜이미지와 가짜이미지를 비평자에 통과시킴
 >       fake = self.critic(fake_img)
 >       valid = self.critic(real_img)
 >
 >       # Construct weighted average between real and fake images
 >       interpolated_img = RandomWeightedAverage(self.batch_size)([real_img, fake_img]) # 보간된 이미지를 만들고 다시 비평자에 통과시킴
 >       # Determine validity of weighted sample
 >       validity_interpolated = self.critic(interpolated_img)
 >
 >       # Use Python partial to provide loss function with additional
 >       # 'interpolated_samples' argument
 >       partial_gp_loss = partial(self.gradient_penalty_loss,
 >                         interpolated_samples=interpolated_img) # 예측과 진짜레이블 두개의 입력만 기대한다.따라서 partial함수를 이용해 보간된 이미지를 gradient_penaly_loss함수적용
 >       partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names 
 >
 >       self.critic_model = Model(inputs=[real_img, z_disc],
 >                           outputs=[valid, fake, validity_interpolated]) # 비평자를 훈련하기 위해 모델에 두개의 입력이 정의됨. 출력이 3개 -1(가짜), 1(진짜), 0(더미)
 >
 >       self.critic_model.compile(
 >           loss=[self.wasserstein,self.wasserstein, partial_gp_loss]
 >           ,optimizer=self.get_opti(self.critic_learning_rate)
 >           ,loss_weights=[1, 1, self.grad_weight]
 >           ) # 총3개의 손실함수로 비평자를 컴파일한다. 전체손실은 세 손실의 합. 또한 원본논문의 권유에 따라 그레디언트 손실에 10배 가중치를 부여한다. 
 > ```     
### 배치정규화
 > WGAN-GP의 비평자에서는 배치정규화를 사용하면 안된다. 배치 정규화는 같은 배치 안의 이미지 사이에 상관관계를 만들기 때문에 그레디언트 패널티의 효과가 떨어진다.
- 04_03_wgangp_faces_train.ipynb 로 연습해보기.
