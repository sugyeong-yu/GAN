# Chapter 7.Muse_GAN
- 어려운 점
  - 컴퓨터로 듣기좋은 음악을 작곡하기 위해서 모델이 순차적인 음악의 구조를 학습하고 재생성할 수 있어야한다. 
  - 또한, 이산적인 확률집합을 사용해 연속적인 악보를 만들 수 있어야한다. 
  - 그러나 음악창작에서는 텍스트생성과 달리 pitch와 rhythm이 있어 더 어렵다. 이에 음악은 화음의 스트림을 동시에 처리해야한다. 
- 예제에서는 하나의 선율을 가진 음악생성에 초점을 맞추어 문제를 단순화해본다.
- RNN에서 **어텐션 매커니즘**을 사용한다. 
  - 이를 사용하면 RNN이 다음에 올 음표를 예측하기 위해 이전의 어느 음표에 초점을 맞출지 선택할 수 있다.
- 마지막으로 여러개의 선율을 가진 음악을 생성하는 작업을 다루고 GAN기반의 구조를 사용해 여러 성부(소프라노, 알토, 테너, 베이스)를 가진 음악을 생성해본다.

## 1. 음악을 생성모델 훈련데이터로 변환하기
- [07_01_notation_compose](https://github.com/sugyeong-yu/GAN/blob/main/7.muse_GAN/07_01_notation_compose.ipynb)참고
### 1-1. 음악기호
- python의 music21 lib사용
```
from music21 import converter
# dataload
dataset_name = 'cello'
filename = 'cs1-2all'
file = "./data/{}/{}.mid".format(dataset_name, filename)

original_score = converter.parse(file).chordify() # 1
original_score.show()
```
- original_score.show() 결과
<img src="https://user-images.githubusercontent.com/70633080/107189900-6f669b00-6a2d-11eb-99c5-4543b4fa5032.png" width=50% height=50%> 
```
original_score.show('text')
```
- original_score.show('text') 결과

![image](https://user-images.githubusercontent.com/70633080/121471377-88428580-c9fa-11eb-88fd-8c7810921ce6.png)


- 1: chordify method를 사용해 여러파트로 나누어진 음표를 하나의 파트에서 동시에 연주되는 화음으로 압축하여 show.
    - 이 음악은 하나의 악기로 연주하므로 파트를 나눌필요가 없지만 여러악기를 사용하는 음악의 경우 파트를 나누는 것이 좋다.
  
```
from music21 import chord, note
notes = []
durations = []

for element in original_score.flat:
    
    if isinstance(element, chord.Chord):
        notes.append('.'.join(n.nameWithOctave for n in element.pitches))
        durations.append(element.duration.quarterLength)

    if isinstance(element, note.Note):
        if element.isRest:
            notes.append(str(element.name))
            durations.append(element.duration.quarterLength)
        else:
            notes.append(str(element.nameWithOctave))
            durations.append(element.duration.quarterLength)
```
- 이는 악보를 순회하며 각음표와 쉼표의 피치와 박자를 두개의 list로 추출한다.
- 코드 전체는 하나의 문자열로, 코드의 개별음표는 점으로 구분한다. 
- 각 음표의 이름뒤에 있는 숫자는 음표가 속한 옥타브를 지칭한다.\
![image](https://user-images.githubusercontent.com/70633080/107190696-780ba100-6a2e-11eb-8eb9-9d22118df898.png)
- 이에 피치의 시퀀스가 주어지면 다음 피치를 예측하는 모델을 만들어야한다.
- keras를 사용해 피치와 박자를 동시에 처리할 수 있는 모델을 만들 수 있다.

## 2. 첫번째 음악생성 RNN
- 모델훈련을 위한 dataset을 만들기 위해 **피치와 박자를 정수값으로 변환**한다.\
![image](https://user-images.githubusercontent.com/70633080/107191281-46470a00-6a2f-11eb-99fa-d430a7f76013.png)
![image](https://user-images.githubusercontent.com/70633080/107191301-4f37db80-6a2f-11eb-9a1e-24df96fdac46.png)
- 이는 텍스트데이터의 전처리와 동일하다.
- 1. 임베딩 층을 사용해 정수를 벡터로 변환한다.
- 2. 데이터를 32개의 음표씩 나누어 훈련세트를 만든다. ( target은 시퀀스에 있는 one-hot encoding된 다음 피치와 박자이다.)
- 데이터셋의 샘플\
![image](https://user-images.githubusercontent.com/70633080/107191465-8dcd9600-6a2f-11eb-841d-dfab1c759ef5.png)
- 본 예제에서 어텐션매커니즘을 사용한 LSTM network를 사용한다.
- 어텐션 매커니즘은 순환층이나 합성곱층이 필요하지 않고 완전히 어텐션으로만 구성된 **transformer model**을 탄생시켰다.
  - transformer 구조는 chapter 9에서.
- 따라서, 어텐션과 LSTM을 연결, 이전음표의 시퀀스가 주어지면 다음 음표를 예측하는데 초점을 맞춘다.

### 2-1 어텐션
: 어텐션매커니즘은 원래 텍스트번역문제, 특히 영어문장을 프랑스어로 번역하는 문제에 주로 적용된 모델이다.
- 이전 인코더-디코터 네트워크방식에서의 문제는 **문맥벡터가 병목될 수 있다는 것**이다.
  - 특히 긴 문장의 시작 부분의 정보는 문맥벡터에 도달할 때 희석될 수 있다.
  - 따라서 이런 종료의 인코더-디코더 네트워크는 디코더가 원본 문장을 올바르게 번역하기 위해 필요한 모든 정보를 유지하지 못한다.
- 음표나 음표의 시퀀스를 파악하기 위해서는 가장 최근정보가 아닌 시퀀스를 한참 거슬러 올라간 이전정보를 사용하는 것이 중요하다.
#### 어텐션매커니즘
- 모델이 인코더RNN의 마지막 은닉상태만 문맥벡터로 사용하지 않는다.
- 인코더 RNN의 이전 time step에 있는 은닉상태의 가중치 합으로 문맥벡터를 만든다.
- 인코더의 이전 은닉상태와 디코더의 현재 은닉상태를 문맥벡터 생성을 위한 덧셈 가중치로 변환하는 일련의 층이다.
### 2-2 어텐션매커니즘 생성
- 아래그림은 순환 층의 은닉상태에 어텐션 매커니즘을 적용한 네트워크이다.\
![image](https://user-images.githubusercontent.com/70633080/107197824-c1142300-6a37-11eb-8f34-8e58c03368c9.png)\
![image](https://user-images.githubusercontent.com/70633080/107312429-beffa200-6ad3-11eb-8f53-53d3906b2588.png)
1. 각 은닉상태 h_j가 정렬함수(alignment function)을 통과하여 스칼라값e_j를 출력한다.
  - 이 예에서 정렬함수는 하나의 출력유닛과 tanh 활성화 함수를 가진 단순한 fully connected layer이다.
2. 다음 벡터 e_1 ~ e_n에 softmax함수가 적용되어 가중치벡터 a_1 ~ a_n을 만든다.
3. 마지막으로 각 은닉상태벡터 h_j와 해당하는 가중치 a_j를 곱해 더한 후, 문맥벡터 c를 만든다.
  - 문맥벡터 c는 은닉상태벡터와 길이가 동일하다.
- 문맥벡터는 softmax 활성화함수를 가진 Dense층을 통과하여 다음 음표에 대한 확률분포를 출력한다.
- 아래 코드는 models의 [RNNAttention.py](https://github.com/sugyeong-yu/GAN/blob/main/7.muse_GAN/models/RNNAttention.py) 의 create_network()에서 볼 수 있다.
```
notes_in = Input(shape=(None,)) # 1 (음표)
durations_in = Input(shape=(None,)) # 2 (박자)

x1 = Embedding(n_notes,embed_size)(notes_in)
x2 = Embedding(n_durations,embed_size)(durations_in)

x = Concatenate()([x1,x2]) # 3
x = LSTM(rnn_units, return_sequences=True)(x) # 4
x = LSTM(rnn_units, return_sequences=True)(x)

e = Dense(1,activation='tanh')(x) # 5
e = Reshape([-1])(e)

alpha = Activation('softmax')(e) # 6

c = Permute([2,1])(RepeatVector(rnn_units)(alpha)) # 7
c = Multiply()([x,c])
c = Lambda(lambda xin : K.sum(xin,axis=1),output_shape=(rnn_units,))(c)

notes_out = Dense(n_notes,activation='softmax', name='pitch')(c) # 8
durations_out = Dense(n_durations,activation='softmax',name='duration')(c)

model = Model([notes_in,durations_in],[notes_out,duration_out]) # 9

att_model =  Model([notes_int,durations_int],alpha) # 10
opti = RMSprop(lr=0,001)
model.compile(loss=['categorical_crossentropy','categorical_crossentropy'], optimizer=opti) # 11
```
- 1: Network의 입력은 2개이다. (이전 음표이름, 박자에 대한 시퀀스) 
  - 어텐션 매커니즘에서는 고정된 입력의 길이를 필요로 하지않는다. 따라서 **시퀀스의 길이를 지정하지 않는다.**
- 2: Embedding 층은 음표이름과 박자에 대한 정수값을 vector로 반환한다.
- 3: 하나의 긴 벡터로 연결되어 순환층의 입력으로 사용된다.
- 4: 두개의 LSTM층을 사용한다. 마지막 은닉상태가 아니라 **전체 은닉상태의 시퀀스를 다음층에 전달하기 위해 return_sequence를 True로** 지정한다.
- 5: 정렬함수는 하나의 출력유닛과 tanh 활성화 함수를 가진 단순한 Dense층이다.
  - Reshape을 통해 출력을 하나의 vector로 펼친다. ( 이는 입력시퀀스의 길이와 동일하다)
- 6: 정렬된 값에 softmax함수를 적용해 가중치를 계산한다.
- 7: 은닉상태의 가중치 합을 얻기위해 RepeatVector 층으로 이 가중치를 rnn_units번 복사해 [rnn_units,seq_len]크기의 행렬을 얻는다.
  - 그 후 이행렬과 마지막 LSTM층의 은닉상태와 원소별곱셈을 수행.
  - 이 결과는 [seq_len,rnn_units] 크기가 된다.
  - 마지막으로 lambda layer를 사용해 seq_len축을 따라 더해 rnn_units길이의 문맥벡터를 만든다.
- 8: 네트워크의 출력은 2개이다. (다음 음표이름, 다음 음표의 길이)
- 9: 최종 모델은 이전음표이름과 박자를 입력으로 받고 다음 음표이름과 박자에 대한 분포를 출력한다.
- 10: Network가 순환층의 은닉상태에 어떻게 가중치를 부여하는지 보기위해 alpha 벡터를 출력하는 모델을 만든다.
- 11: 음표이름과 박자 출력은 모두 다중분류문제이다. 따라서 categorical_crossentropy를 사용해 컴파일한다.\
![image](https://user-images.githubusercontent.com/70633080/107330388-d438f880-6af4-11eb-8bd1-1c0431cfc3a3.png)
- 어텐션을 이용한 LSTM 훈련은 [07_02_lstm_compose_train](https://github.com/sugyeong-yu/GAN/blob/main/7.muse_GAN/07_02_lstm_compose_train.ipynb)에서 가능하다.

### 2-3 어텐션을 사용한 RNN분석
- [07_03_lstm_compose_predict.ipynb](https://github.com/sugyeong-yu/GAN/blob/main/7.muse_GAN/07_03_lstm_compose_predict.ipynb)에서 실행할 수 있다.
- <START> 토큰시퀀스만 network에 넣어 처음부터 음악을 생성해보자.
1. (음표이름과 박자에 대한) 현재시퀀스가 주어지면 모델은 다음 음표이름과 박자에 대한 2개의 확률분포를 예측한다.
2. 확률분포에서 샘플링할때 **tenperature 매개변수를 사용**해 샘플링과정에 얼마나 많은 변동성을 부여할지 제어한다.
3. 선택된 음표를 저장하고 음표이름과 박자를 각각의 시퀀스 뒤에 추가한다.
4. 시퀀스 길이가 모델이 훈련한 시퀀스의 길이보다 크면 시퀀스 처음부분의 원소를 버린다.
5. 생성하고 싶은만큼 이 새로운 시퀀스로 과정을 반복한다.

- 훈련이 진행됨에 따라 음악이 점점 복잡해짐을 알 수 있다.
- 예측한 확률분포를 히트맵으로 그려 시간에 따른 피치분포를 분석할 수 있다.
<img src="https://user-images.githubusercontent.com/70633080/107333736-398ee880-6af9-11eb-8838-eed25fffb4d6.png" width=50% height=50%>
- 가로축 : 음표번호, 세로축 : 피치값(마디번호)
- 모델이 특정키에 속한 음표를 명확히 학습했다. 키에 속하지 않은 음표는 분포에서 비어있다.\
<img src="https://user-images.githubusercontent.com/70633080/107334381-fbde8f80-6af9-11eb-94ca-b104afcd0e28.png" width=50% height=50%>
- 이는 생성된 시퀀스의 각 지점에서 네트워크가 계산한 a벡터 원소의 값을 보여주는 어텐선 룩업테이블이다.
- 가로축 : 생성된 음표의 시퀀스 , 세로축 : 수평축을 따라 음표를 예측할때 네트워크의 어텐션이 주목하는 곳(a벡터)
- 붉은색 상자일수록 시퀀스에서 이 지점에 해당하는 은닉상태에 부여된 어텐션이 큼을 의미한다.
- 네트워크는 이전 음표에 거의 동등한 어텐션을 부여한다. 
  - 그러나 6개이전의 음표에는 거의 가중치를 부여하지 않는다. 
- 또한, 가까이에 있는 음표를 무시하는 경우가 있다. 
  - 악절이해에 추가정보를 제공하지 못하기 때문이다. 
  
### 2-4 인코더-디코더 네트워크의 어텐션
- 어텐션 매커니즘은 연속된 시퀀스 예측을 위해 이전상태중 어떤 것이 중요한지 결정하는데 도움을 주는 도구이다.
- 인코더-디코더 네트워크에 어텐션을 구성할수도 있다. 
- RNN decoder를 이용해 한음표씩 시퀀스를 만드는 것이 아닌 음표의 시퀀스를 예측한다.\
![image](https://user-images.githubusercontent.com/70633080/107337114-49103080-6afd-11eb-8144-754c676c0693.png)
- 어텐션이 추가된 인코더-디코더 구조이다.
- **디코더의 은닉상태도 어텐션매커니즘으로 들어간다.**
- 모델이 인코더의 이전 은닉상태 뿐만 아니라 디코더의 현재 은닉상태를 통해 주의를 집중할 곳을 결정한다.
### 2-5 다중선율음악 생성
- 지금까지의 RNN은 하나의 선율에 화음이 저장된 데이터를 사용했기 때문에 화음별 관계를 학습하지 못한다.
- 이는 이미지생성에서 RGB 3개의 채널을 연결하여 학습하는 것과 유사할 수 있다.
- 따라서 음악생성은 이미지생성문제로 다룰 수 있어 GAN을 음악에도 적용가능하다.
