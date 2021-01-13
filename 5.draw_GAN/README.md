# Chapter 5.draw_GAN
- style transfer 
> style image가 주어졌을때, 같은부류에 속한것 같은 느낌을 주도록 base image를 변환하는 모델을 훈련하는 것.\
> style image에 내재된 분포 를 모델링하는 것이 아닌 image에서 style을 결정하는 요소만 추출하여 base image에 주입하는 것.\
> :exclamation: style image와 base image를 보간방법으로 합쳐서는 안된다. \
> style image의 내용이 비쳐보이거나 색이 탁하고 흐릿해지기때문이다.\
> 또한, 하나의 이미지를 사용하는 것이아니라 style image set 전체에서 아티스트의 style을 찾아낸다. \
> 아티스트가 base image를 이용해 원복작품을 유지하면서 다른작품의 스타일기교로 완성하는 듯한 효과를 원하는것이다.\
![style_transfer](https://user-images.githubusercontent.com/70633080/102320857-34f9f800-3fc0-11eb-95cc-a42375292550.png)
------------------------------------------
## 1. cycleGAN
------------------------------------------
> cycleGAN은 생성모델링, 특히 style transfer분야에서 핵심적 발전 중 하나인 모델이다.\
> 샘플 쌍으로 구성된 훈련세트 없이도 참조 이미지세트의 스타일을 다른 이미지로 복사하는 모델을 훈련할 수 있는 방법을 보였다.
> - pix2pix와 같은 이전의 style transfer은 training set의 각이미지가 source와 target 도메인에 모두 존재해야했다. 이는 불가능한경우도 있었다. 예를 들어 피카소가 그린 "엠파이어스테이트빌딩"을 구할 수 없거나 동일한 위치에서 서있는 말과 얼룩말의 사진을 찍으려면 엄청난 노력이 필요하다.
> - cycleGAN은 source와 target 도메인에 이미지 쌍없이 모델이 어떻게 훈련하는지 보여준다.
> ![pix2pix2_cyclegan](https://user-images.githubusercontent.com/70633080/102322553-94590780-3fc2-11eb-8018-d2a62203c933.png)
> - 훈련에 사용할 데이터 다운로드
> ```
> bash ./scripts/download_cyclegan_data.sh apple2orange
> ```
### 개요
 > - cycleGAN은 4개의 모델로 구성된다. 2개의 생성자 + 2개의 판별자\
 > -첫번째 생성자 g_AB\
 >  : 도메인 A의 이미지를 도메인 B로 바꿈\
 > -두번째 생성자 g_BA\
 >  : 도메인 B의 이미지를 도메인 A로 바꿈\
 > 생성자는 훈련을 위한 이미지 쌍이 없기때문에 생성자가 만든 이미지가 설득력이 있는지 결정하는 두개의 판별자를 훈련한다.\
 > -첫번째 판별자 d_A\
 >  : 도메인 A의 진짜이미지와 생성자 g_BA가 만든 가짜 이미지를 구별할 수 있도록 훈련\
 > -두번째 판별자 d_B\
 >  : 도메인 B의 진짜이미지와 생성자 g_AB가 만든 가짜 이미지를 구별할 수 있도록 훈련
 > - 05_01_cyclegan_train.ipynb 에서  cyclegan훈련실행가능
 > < 모델 생성 >
 > ```
 > gan = CycleGAN(
 >   input_dim = (IMAGE_SIZE,IMAGE_SIZE,3)
 >   ,learning_rate = 0.0002
 >   , buffer_max_length = 50
 >   , lambda_validation = 1
 >   , lambda_reconstr = 10
 >   , lambda_id = 2
 >   , generator_type = 'unet'
 >   , gen_n_filters = 32
 >   , disc_n_filters = 32
 >   )
 > ```
 > - 보통 cycleGAN의 생성자로는 U-Net 또는 ResNet 둘 중 하나를 선택한다.
### 생성자(U-Net)
 > - U-Net의 구조\
 > ![U-NET](https://user-images.githubusercontent.com/70633080/102707505-ecec1580-42de-11eb-9eaf-68cc7107dc7b.png)
 > - VAE와 비슷한 방식으로 다운샘플링과 업샘플링으로 구성된다. \
 > -다운샘플링은 입력이미지를 공간방향(사이즈)으로 압축, 채널방향으로는 확장\
 > -업샘플링은 공간방향으로 표현을 확장, 채널의 수는 감소\
 > - 하지만 VAE와 달리 U-Net network의 다운샘플링과 업샘플링에는 크기가 동일한 층끼리 연결된 skip connection이 있다.\
 > -VAE는 선형적으로 데이터가 입력부터 출력까지 층을 차례로 거쳐 네트워크를 통과한다.\
 > -U-Net은 skip connection이 있기 때문에 지름길을 통해 뒷 쪽 layer에 정보를 전달한다.\
 > - skip conncection\
 > network의 다운샘플링에 각 층 모델은 이미지가 무엇인지는 감지하지만 어디에있는지 위치정보를 잃는다.\
 > 업샘플링의 각 층에서 다운샘플링동안 잃었던 공간정보를 되돌린다.\
 > 이것이 skip connection이 필요한이유!!
 > - 다운샘플링에서의 고수준추상정보(image의 style)을 network의 앞쪽 층으로 부터 전달된 구체적 공간정보(image contents)와 섞는다.
 > #### concatenate 층
 > > 특정 축을 따라서 여러 층을 합친다.\
 > > 예를들어, 케라스에서 두개의 층 x와 y를 합칠 수 있다.
 > > ```
 > > concatenate()([x,y])
 > > ```
 > > - U-Net에선 concatenate 층을 사용해 업샘플링 층과 동일한 크기의 출력을 내는 다운샘플링 쪽의 층을 연결한다.\
 > > 이 층들은 채널 차원을 따라 합쳐지므로 채널 수가 k개에서 2k개로 늘어나며 공간방향차원은 동일하게 유지된다.\ 
 > > - concatenate층은 층을 접합하는 역할만 하므로 학습되는 가중치는 없다.
 > #### InstanceNormalizaion 층
 > > - CycleGan의 생성자는 BatchNormalization 층 대신 InstanceNormalization층을 사용한다. > style transfer문제에서 더 좋은 결과를 나타낸다.\
 > > -InstanceNormalization 층은 배치단위가 아닌 개별 샘플을 각각 정규화 한다.\
 > > -BatchNormalization층과 달리 이동평균을 위해 훈련과정에서 계산하는 mu와 sigma 파라미터가 필요하지 않다. \
 > > -각층을 정규화하기위해 사용되는 평균과 표준편차는 채널별로 나누어 샘플별로 계산된다.\
 > > -또한 스케일이나 이동(beta)파라미터를 사용하지 않기 때문에 학습되는 가중치가 없다.\
 > - U-net 생성자
 > - models/cycleGAN.py
 > ```
 > def build_generator_unet(self):
 >
 >       def downsample(layer_input, filters, f_size=4):
 >           d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
 >           d = InstanceNormalization(axis = -1, center = False, scale = False)(d)
 >           d = Activation('relu')(d)
 >           
 >           return d
 >
 >       def upsample(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
 >           u = UpSampling2D(size=2)(layer_input)
 >           u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(u)
 >           u = InstanceNormalization(axis = -1, center = False, scale = False)(u)
 >           u = Activation('relu')(u)
 >           if dropout_rate:
 >               u = Dropout(dropout_rate)(u)
 >
 >           u = Concatenate()([u, skip_input])
 >           return u
 >
 >       # Image input
 >       img = Input(shape=self.img_shape)
 >
 >       # Downsampling
 >       d1 = downsample(img, self.gen_n_filters) 
 >       d2 = downsample(d1, self.gen_n_filters*2)
 >       d3 = downsample(d2, self.gen_n_filters*4)
 >       d4 = downsample(d3, self.gen_n_filters*8)
 >
 >       # Upsampling
 >       u1 = upsample(d4, d3, self.gen_n_filters*4)
 >       u2 = upsample(u1, d2, self.gen_n_filters*2)
 >       u3 = upsample(u2, d1, self.gen_n_filters) 
 >
 >       u4 = UpSampling2D(size=2)(u3)
 >       output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)
 >
 >       return Model(img, output_img)
 > ```
### 판별자 (U-net)
 > - cycleGAN의 판별자는 숫자하나(진짜일확률)이 아니라 16*16 크기의 채널 하나를 가진 tensor를 출력한다.\
 > cycleGAN이 patchGAN으로 불리는 모델의 판별자 구조를 승계하기 때문이다.\
 > -patchGAN\
 > 이미지 전체에 대한 예측이 아닌 중첩된 patch로 나누어 각 patch가 진짜인지 추측한다.\
 > 따라서 판별자 출력은 하나의 숫자가 아닌 각 patch에 대한 예측확률을 담은 tensor가 된다.\
 > - 판별자의 합성곱 구조로 인해 자동으로 이미지가 patch로 나뉜다.\
 > - 장점\
 > 내용이 아닌 스타일을 기반으로 판별자가 얼마나 잘 구별하는지 손실함수가 측정할 수 있다.\
 > 따라서 판별자는 내용이 아닌 스타일로 두 이미지가 다른지 구분한다.
 > - models/cycleGAN.py
 > ```
 > def build_discriminator(self):
 >
 >       def conv4(layer_input,filters, stride = 2, norm=True):
 >           y = Conv2D(filters, kernel_size=(4,4), strides=stride, padding='same', kernel_initializer = self.weight_init)(layer_input)
 >          
 >           if norm:
 >               y = InstanceNormalization(axis = -1, center = False, scale = False)(y)
 >
 >           y = LeakyReLU(0.2)(y)
 >          
 >           return y
 >
 >       img = Input(shape=self.img_shape)
 >
 >       y = conv4(img, self.disc_n_filters, stride = 2, norm = False)
 >       y = conv4(y, self.disc_n_filters*2, stride = 2)
 >       y = conv4(y, self.disc_n_filters*4, stride = 2)
 >       y = conv4(y, self.disc_n_filters*8, stride = 1)
 >
 >       output = Conv2D(1, kernel_size=4, strides=1, padding='same',kernel_initializer = self.weight_init)(y)
 >
 >       return Model(img, output)
 > ```
 > - cycleGAN의 판별자는 연속된 합성곱 신경망이다. (첫번째 층 제외하고 모두 sampling 정규화 즉, instanceNormalization을 사용)
 > - 마지막 합성곱 층은 하나의 필터를 사용하고 활성화함수는 적용하지 않는다.
### cycleGAN 컴파일
 > - 입력과 출력을 가지고 있으므로 두 컴파일을 컴파일 할 수 있다.
 > - models/cycleGAN.py의 compile_models 함수
 > ```
 > def compile_models(self):
 >
 >       # Build and compile the discriminators
 >       self.d_A = self.build_discriminator()
 >       self.d_B = self.build_discriminator()
 >       
 >       self.d_A.compile(loss='mse',
 >           optimizer=Adam(self.learning_rate, 0.5),
 >           metrics=['accuracy'])
 >       self.d_B.compile(loss='mse',
 >           optimizer=Adam(self.learning_rate, 0.5),
 >           metrics=['accuracy'])
 > ```
 > - 하지만 생성자는 쌍을 이루는 이미지가 데이터셋에 없기 때문에 바로 컴파일 할 수 없다.\
 > 따라서 세가지 조건으로 생성자를 동시에 평가한다. 
 > 1. 유효성 \
 > 각 생성자에서 만든 이미지가 대응되는 판별자를 속이는가? ( g_BA의 출력이 d_A를 속이고, g_AB의 출력이 d_B를 속이는가?)
 > 2. 재구성 \
 > 두 생성자를 교대로 적용하면 원본이미지를 얻는가? (CYCLEGAN은 순환재구성의 조건으로부터 이름을 따왔다.)
 > 3. 동일성\
 > 각 생성자를 자신의 타깃도메인에 있는 이미지에 적용했을때, 이미지가 바뀌지 않고 그대로 남아있는가?
 > ```
 > # Build the generators
 >       if self.generator_type == 'unet':
 >           self.g_AB = self.build_generator_unet()
 >           self.g_BA = self.build_generator_unet()
 >       else:
 >           self.g_AB = self.build_generator_resnet()
 >           self.g_BA = self.build_generator_resnet()
 >
 >       # For the combined model we will only train the generators
 >       self.d_A.trainable = False
 >       self.d_B.trainable = False
 >
 >       # Input images from both domains
 >       img_A = Input(shape=self.img_shape)
 >       img_B = Input(shape=self.img_shape)
 >
 >       # Translate images to the other domain
 >       fake_B = self.g_AB(img_A)
 >       fake_A = self.g_BA(img_B)
 >       # Translate images back to original domain
 >       reconstr_A = self.g_BA(fake_B)
 >       reconstr_B = self.g_AB(fake_A)
 >       # Identity mapping of images
 >       img_A_id = self.g_BA(img_A)
 >       img_B_id = self.g_AB(img_B) 
 >
 >       # Discriminators determines validity of translated images
 >       valid_A = self.d_A(fake_A)
 >       valid_B = self.d_B(fake_B)
 >
 >       # Combined model trains generators to fool discriminators
 >       self.combined = Model(inputs=[img_A, img_B],
 >                             outputs=[ valid_A, valid_B,
 >                                       reconstr_A, reconstr_B,
 >                                       img_A_id, img_B_id ])
 >       self.combined.compile(loss=['mse', 'mse',  # MSE는 평균제곱오차, MAE는 평균절대오차
 >                                   'mae', 'mae',
 >                                   'mae', 'mae'],
 >                           loss_weights=[  self.lambda_validation,                       self.lambda_validation,
 >                                           self.lambda_reconstr, self.lambda_reconstr,
 >                                           self.lambda_id, self.lambda_id ],
 >                           optimizer=Adam(0.0002, 0.5))
 >
 >       self.d_A.trainable = True
 >       self.d_B.trainable = True
 > ```
 > - 결합된 모델은 각 도메인의 이미지 배치를 입력으로 받고 각 도메인에 대해 (3개의 조건에 맞게) 3개의 출력을 제공한다.
 > - 총 6개의 출력이 만들어 진다.
 > - 앞선 gan과 동일하게 판별자의 가중치를 동결한다.\
 > 판별자가 모델에 관여하지만 생성자의 가중치만 훈련한다.
 > - 전체 손실은 각 조건에 대한 손실의 가중치 합이다. \
 > 평균제곱오차 : 유효성조건에 사용(진짜 1 과 가짜 0 타깃에 대해 판별자의 출력을 확인한다.)\
 > 평균 절댓값오차 : 이미지 대 이미지 조건에 사용한다. (재구성과 동일성 조건)\
### cycleGAN 훈련
 > - 판별자와 생성자를 교대로 훈련하는 GAN의 기본 훈련 방식을 따ㅡㄹㄴ다.
 > - 판별자가 사용하는 타깃이 진짜일 경우 16*16 패치가 모두 1, 가짜일 경우 모두 0
 > - models/cycleGAN.py의 train_discriminators, train_generators, train method을 요약, 정리
 > ```
 > batch_size=1
 > patch= int(self.img_rows/2**4)
 > self.disc_patch= (patch,patch,1)
 > 
 > valid= np.ones((batch_size,)+self.disc_patch) # 진짜는 1, 가짜는 0 (패치마다 하나의 타깃을 설정)
 > fake= np.zeros((batch_size,)+self.disc_patch)
 > 
 > for epoch in range(self.epoch, epochs):
 >   for batch_i, imgs_A, imgs_B) in enumerate(data_loader.load_batch(batch_size)):
 > 
 >      fake_B= self.g_AB.predict(imgs_A)
 >      fake_A= self.g_BA.predict(imgs_B)
 >
 >      dA_loss_real= self.d_A.train_on_batch(imgs_A,valid)
 >      dA_loss_fake= self.d_A.train_on_batch(fake_A, fake)
 >      dA_loss= 0.5*np.add(dA_loss_real,dA_loss_fake)
 >     
 >      dB_loss_real= self.d_B.train_on_batch(imgs_B,valid)
 >      dB_loss_fake= self.d_B.train_on_batch(fake_B, fake)
 >      dB_loss= 0.5*np.add(dB_loss_real,dB_loss_fake)
 >
 >      d_loss= 0.5*np.add(dA_loss, dB_loss)
 >      g_loss= self.combined.train_on_batch([imgs_A, imgs_B],
 >                                           [valid,valid,
 >                                           imgs_A,imgs_B,
 >                                           imgs_A,imgs_B])
 > ```
 > - 판별자를 훈련시키려면 먼저, 생성자를 이용해 가짜 이미지의 배치를 만든다. 그 후 가짜와 진짜 배치로 각 판별자를 훈련한다. \
 > 일반적으로 cycleGAN의 배치크기는 1(하나의 이미지) 이다.
 > - 생성자는 앞서 컴파일된 결합모델을 통해 동시에 훈련된다. 6개의 출력은 컴파일 단계에서 정의한 6개의 손실함수에 대응한다.
### cycleGAN 분석
 > - 동일성손실의 중요성을 알아보기 위해 손실함수에서 동일성손실을 위한 가중치 파라미터를 0으로 설정했을때
 > ![동일성가중치=0일때](https://user-images.githubusercontent.com/70633080/102707486-c1692b00-42de-11eb-9244-3326e740ab99.png)
 > - 오렌지를 사과로 바꿀 수 있지만 선반의 색이 흰색으로 반전됨.\
 > 배경색의 변환을 막아주는 동일성 손실항이 없기 때문.
 > - 동일성 항은 이미지에서 변환에 필요한 부분 이외에는 바꾸지 않도록 생성자에게 제한을 가하는 것이다.
 > - 동일성 손실이 너무 작으면 색이 바뀌는 문제가 생긴다.
 > - 동일성 손실이 너무 크면 cycleGAN이 입력을 다른 도메인의 이미지처럼 보이도록 바꾸지 못한다.
 > - 따라서 세개의 손실함수 가중치의 균형을 잘 잡는것이 중요하다.
## 2. CycleGAN으로 모네그림 그리기
> - cycleGAN의 개념을 사용한 애플리케이션에 대해 살펴보자\
> - CycleGAN이기 때문에 아티스트의 그림을 실제 사진으로도 바꿀 수 있다.
> - Data set
> ```
> bash ./scripts/download_cyclegan_data.sh monet2photo
> ```
> - 모델 생성
> ```
> gan = CycleGAN(
>   input_dim = (256,256,3)
>   ,learning_rate = 0.0002
>   , lambda_validation = 1
>   , lambda_reconstr = 10
>   , lambda_id = 5
>   , generator_type = 'resnet'
>   , gen_n_filters = 32
>   , disc_n_filters = 64
>   )
> ```
### 생성자 (ResNet)
 > #### ResNet
 > > - 이전 층의 정보를 네트워크의 앞쪽에 있는 한개 이상의 층으로 스킵 한다는 점에서 U-Net과 유사하다.
 > > - ResNet은 잔차블록(residual block)을 차례대로 쌓아 구성한다.
 > > ![Resnet](https://user-images.githubusercontent.com/70633080/102707883-2b370400-42e2-11eb-908e-bd7ce11c2b56.png)
 > > - Resnet은 residual block안에 가중치를 가진층 -> relu -> 가중치를 가진층 으로 구성된다.
 > > 가중치를 가진 층은 cycleGAN의 샘플정규화를 사용한 합성곱 층이다.
 > > - residual block을 만드는 케라스 코드
 > > ```
 > > form keras.layers.merge import add
 > > def residual(layer_input,filter):
 > >  shortcut=layer_input
 > >  y= Conv2D(filters,kernel_size=(3,3),strides=1,padding='same')(layer_input)
 > >  y=InstanceNormalization(axis=-1,center=False,scale=False)(y)
 > >  y= Activation('relu')(y)
 > >  y= Conv2D(filters,kernel_size(3,3),strides=1,padding='same')(y)
 > >  y= InstanceNormalization(axis=-1,center=False,scale=False)(y)
 > >
 > >  return add([shortcut,y])
 > > ```
 > - Resnet 생성자는 잔차블록 양쪽에 다운샘플링과 업샘플링 층이 있다. 
 > - 전체 ResNet 구조
 > ![Resnet_generator](https://user-images.githubusercontent.com/70633080/102708027-4a826100-42e3-11eb-828b-43906622af47.png)
 > - Resnet은 gradient vanishing 문제가 없다.\
 > 오차 그레디언트가 잔차블록의 스킵연결을 통해 네트워크에 그대로 역전파되기 때문.
### CycleGAN 분석
 > - 아티스트-사진 스타일 트랜스퍼에 대해 최상의 결과를 얻기 위해 200번의 epoch동안 모델을 훈련.
 > - 훈련과정 초기단계에서 생성자의 출력
 > ![훈련단계별](https://user-images.githubusercontent.com/70633080/102708168-3f7c0080-42e4-11eb-8d12-3cd6b2e6064f.png)
 > - 모델이 모네그림을 사진으로 변환하는 것과 그 반대로 변환하는 것을 배우는 과정을 보여준다.
 > - 첫번째 행은 점차 모네가 사용한 특유의 색깔과 붓질을 사진에서 볼 수 있다. 색은 자연스럽고 경계선은 부드럽게.
 > - 두번째 행에서는 반대현상이 일어난다. ( 모네가 직접 그린것 같은 그림으로 사진을 변환)
 > - 200 epoch 훈련한 모델이 만든 결과
 > ![image](https://user-images.githubusercontent.com/70633080/102708245-eeb8d780-42e4-11eb-9932-839255117c12.png)
## 3. 뉴럴 스타일 트랜스퍼
 > - 앞서 본것과 다른 종류의 스타일 트랜스퍼 애플리케이션
 > - 뉴럴스타일트랜스퍼\
 > 훈련세트를 사용하지 않고 이미지의 스타일을 다른 이미지로 전달\
 > ![image](https://user-images.githubusercontent.com/70633080/102867967-e2678280-447c-11eb-8ce3-ea93a7e2bd2b.png)\
 > 이는 세 부분으로 구성된 손실함수의 가중치 합을 기반으로 작동한다.
 > 1. 콘텐츠 손실(content loss)\
 > : 합성된 이미지는 베이스 이미지의 콘텐츠를 동일하게 포함해야한다.
 > 2. 스타일 손실 (style loss)\
 > : 합성된 이미지는 스타일 이미지와 동일한 일반적인 스타일을 가져야한다.
 > 3. 총 변위 손실 (total variation loss)\
 > : 합성된 이미지는 픽셀처럼 격자문의가 나타나지 않고 부드러워야 한다.
 > - 경사하강법으로 이 손실을 최소화한다.\
 > -즉, 많은 반복을 통해 손실함수의 음의 gradient양에 비례해 픽셀값을 업데이트한다.\
 > -반복이 진행됨에 따라 손실은 점차 줄어들어 베이스이미지의 콘텐츠와 스타일이미지를 합친 합성이미지를 얻게된다.
 > - 베이스 이미지와 스타일이미지 두개만 가지고 있기 때문에 이전에서 적용한 훈련세트에서 학습한 정보를 기반으로 새로운 이미지를 생성하는 것이 불가능하다.
 > - 따라서 사전 훈련된 심층신경망을 사용해 손실함수에 필요한 이미지에 대한 중요한 정보를 얻을 수 있다.
 > - 3개의 손실함수를 정의한다.
 >   1. 콘텐츠 손실
 >   2. 스타일 손실
 >   3. 총 변위 손실
### 3.1 콘텐츠 손실
 > - 콘텐츠의 내용과 전반적인 사물의 배치측면에서 두이미지가 얼마나 다른지를 측정하는 손실함수이다.\
 > - 콘텐츠 손실은 개별픽셀값과 무관하다. ( 픽셀값 비교하는 방법은 적절하지 않다.)\
 > : 두 이미지의 장면이 같더라도 개별 픽셀값이 비슷할 것이라 기대하기는 어렵다.
 > - 건물, 하늘, 강과 같은 **고차원특성**의 존재와 **대략적인 위치** 를 기반으로 이미지를 점수화 해야한다.
 > - 신경망은 초기 layer에서 단순한 특징들을 결합해 더 깊은층에서 자연스럽게 더 높은 수준의 특성을 학습한다.
 > - 베이스 이미지와 합성된 이미지에 대해 이 출력을 계산해 그 사이의 **평균제곱오차(MSE)**를 측정하면 콘텐츠 손실함수가 된다.
 > - 예제에서 사용할 사전훈련된 네트워크는 VGG-19이다.
 >   - ImageNet 데이터셋의 100만개이상의 이미지를 1000개 이상의 범주로 분류하도록 훈련된 19개의 층을 가진 합성곱 신경망\
 > ![image](https://user-images.githubusercontent.com/70633080/104405952-85557d00-55a1-11eb-9c80-0e568aa2673a.png)\
 > ![image](https://user-images.githubusercontent.com/70633080/104411747-5396e300-55ae-11eb-9a9a-a7d1e5eae68a.png)
 > ```
 > from keras.applications import vgg19 # 사전훈련된 vgg19모델을 임포트
 > from keras import backend as K
 >
 > base_image_path='/path_to_images/base_image.jpg'
 > style_reference_image_path='/path_to_images/styled_image.jpg'
 >
 > content_weight=0.01
 > base_image=K.variable(preprocess_image(base_image_path))# 베이스이미지와 스타일 이미지를 위한 두개의 케라스 변수와 생성된 합성이미지를 담을 플레이스홀더를 정의
 > style_reference_image=K.variable(preprocess_image(style_reference_image_path))
 > combination_image=K.placeholder((1,img_nrows,img_ncols,3))
 >
 > input_tensor=K.concatenate([base_image,style_reference_image,combination_image],axis=0)# 세이미지를 연결하여 vgg19모델의 입력텐서를 만든다. , axis=0 -> 채널축으로 합친다
 > model=vgg19.VGG19(input_tensor=input_tensor, weights='imagenet',include_top=False) # include_top=False는 이미지 분류를 위한 네트워크의 마지막 fx층의 가중치를 사용하지 않는다는 뜻이다. 우리는 입력이미지의 고수준특징을 감지하는 합성곱층에만 관심이 있기 때문.
 >
 > outputs_dict=dict([(layer.name,layer.output) for layer in model.layers])
 > layer_Features=outputs_dict['block5_conv2']# 다섯번째 블록의 두번째 합성곱층을 콘텐츠손실계산을 위해 사용. 
 > # 더 낮거나 깊은 층을 선택하면 손실함수가 정의하는 콘텐츠에 영향을 미친다. 따라서 생성된 합성 이미지의 성질이 바뀌게 될 수 있다.
 > base_image_features=layer_Features[0,:,:,:]
 > combination_features=layer_Features[2,:,:,:]  # vgg19에 주입된 입력텐서에서 베이스이미지특성과 합성이미지특성을 추출.
 >
 > def content_loss(content,gen):
 >   return K.sum(K.square(gen-content))
 >
 > content_loss=content_weight * content_loss(base_image_features,combination_features) # 두이미지에 대한 층의 출력간에 거리제곱합을 계산하고 가중치 파라미터를 곱해 콘텐츠손실을 얻는다.
 > ```
### 3.2 스타일 손실
 > - 스타일이 비슷한 이미지는 특정 층의 **특성맵 사이의 동일한 상관관계패턴**을 가진다는 아이디어를 기반으로 한다.
 > - vgg19의 어떤 층에서 한 채널은 녹색부분을, 다른채널은 뾰족함을 감지하고 또다른 채널은 갈색부분을 감지한다고 했을때, 두 이미지에 대한 이 특성맵들이 얼마나 동시에 활성화되는지를 측정.
 > - 특성맵이 얼마나 동시에 활성화되는지 수치적으로 측정하려면?
 >   - 특성맵을 펼치고 스칼라곱(또는 dot product)를 계산한다.
 >   - 값이 크면 상관관계가 크고, 작으면 상관관계가 없음을 뜻한다.
 > - 층에 있는 모든 특성사이의 스칼라곱을 담은 행렬을 정의할 수 있다. 이를 **gram matrix**라고한다.
 > - gram matrix의 size는 채널수 * 채널수 즉, 필터 수 * 필터 수 이다.
 > - 스타일이 비슷한 이미지끼리 비슷한 그람행렬을 가진다.
 > - 따라서 베이스이미지와 합성된 이미지에 대해 네트워크의 여러층에서 그람행렬을 계산해야한다.
 > - 그 후 두 그람행렬의 제곱오차합을 사용해 유사도를 비교한다.\
 > ![image](https://user-images.githubusercontent.com/70633080/104411796-690c0d00-55ae-11eb-8d12-a9da5edd7b24.png)\
 > ![image](https://user-images.githubusercontent.com/70633080/104411823-7628fc00-55ae-11eb-9f41-bc8414ee3491.png)
