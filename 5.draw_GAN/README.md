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
 > !!!!!!그림 !!!!!!!!!!!!1\
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
 > 
