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
 > > -각층을 정규화하기위해 사용되는 평균과 표준편차는 채널별로 나누어 샘플별로 계산된다.
 > > -또한 스케일이나 이동(beta)파라미터를 사용하지 않기 때문에 학습되는 가중치가 없다.
