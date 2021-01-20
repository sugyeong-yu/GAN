# Chapter 6. write_GAN
6장에서는 텍스트 데이터를 생성하는 모델을 만든다.
- 텍스트와 이미지의 차이점
1. 텍스트는 개별적인 데이터 조각(문자,단어)로 구성된다 반면 이미지픽셀은 연속적인 색상스펙트럼위의 한점이다.
    - 이산적인 텍스트 데이터에는 일반적 방식으로 역전파를 적용할 수 없다. 이문제를 해결할 방법을 찾아야한다. (cat -> dog 변환이 힘듬)
2. 텍스트는 시간 차원이 있지만 공간차원은 없다. 이미지는 두개의 공간차원이 있고 시간차원은 없다.
    - 텍스트에서 단어의 순서는 매우 중요하다. 반면 이미지는 뒤집어도 영향을 미치지 않는다. 
    - 단어사이에는 모델이 감지해야할 순서에 대한 의존성이 있는 경우가 많다. 반면 이미지는 모든 픽셀이 동시에 처리될 수 있다.
3. 텍스트는 개별단위(단어,문자)의 작은변화에 매우 민감하다. 이미지는 일반적으로 덜 민감하다.
    - 모든 단어가 문장의 전체의미를 나타내는에 중요하기 때문에 논리적인 텍스트를 생성하는 모델을 훈련하는 것은 매우 어렵다.
4. 텍스트는 규칙기반을 가진 문법구조를 가진다. 이미지는 픽셀값을 할당하기 위해 사전에 정의된 규칙이 없다.

## 1. LSTM 이란
 > LSTM은 순환신경망 RNN의 한 종류이다.
 > - RNN과의 다른점
 >      - RNN의 순환층은 매우 간단하고 tanh함수 하나로 구성되어 있다.
 >      - 이 함수는 타임스텝 사이에 전달된 정보를 -1과 1사이로 스케일을 맞춰준다. 그러나 vanising gradient문제가 발생하여 긴 시퀀스데이터에는 맞지 않았다.
 >      - 이에 긴 시퀀스에서 훈련할 수 있는 LSTM이 제시되었다.
 
## 2. 첫번째 LSTM 네트워크
 > - 데이터 : <http://www.gutenberg.net/>에서 이솝우화모음집 <http://bit.ly/2QQEf5T>를 다운.
 > 1. 데이터 가공 단계
 > - 06_01_lstm_text_train.ipynb 참고
 > ### 2.1 토큰화
 > 첫번째 단계로 텍스트를 정제하고 토큰화 한다.
 > #### 토큰화란?
 > > 토큰화란 텍스트를 단어나 문자와 같은 개별단위로 나누는 것을 의미한다. 텍스트 생성모델로 만드려는 종류에 따라 텍스트 토큰화 방법이 달라진다. 또한, 단어와 문자토큰은 각기 장단점이 있어 어떤 선택을 하느냐에 따라 텍스트정제방법과 모델의 출력에 영향을 미친다.
 > > - 단어토큰
 > >    - 모든 텍스트를 소문자로 변환한다. 그러나 사람이름이나 지명같은 고유명사는 대문자 그대로 남겨두어 별도로 토큰화하는 것이 더 좋을 수 있다.
 > >    - 어휘사전(단어사전)이 매우 클 수 있다. 희소한 단어는 별도의 토큰으로 포함하기보다 알려지지않은 단어 unknown word에 해당하는 토큰으로 바꾸어 신경망이 학습해야할 가중치 수를 줄이는게 좋다.
 > >    - 단어에서 어간(stem)을 추출할 수 있다. ex) browse,browing,browses,browsed => brows
 > >    - 구두점(마침표,쉼표)를 토큰화 하거나 모두 제거해야한다.
 > >    - 단어 토큰화를 사용하면 훈련어휘사전에 없는 단어는 모델이 예측할 수 없다.
 > >
 > > - 문자토큰
 > >    - 모델이 문자의 시퀀스를 생성해 훈련어휘사전에 없는 새로운 단어를 만들 수 있다.
 > >    - 대문자는 소문자로 바꾸거나 별도의 토큰으로 남겨둘 수 있다.
 > >    - 어휘사전이 비교적 매우 작다. 마지막출력층에 학습할 가중치 수가 적기 때문에 모델의 훈련 속도에 유리하다.
 >
 > - 예제에서는 어간 추출 없이 소문자 단어로 토큰화 한다.
 > - 모델이 문장의 끝이나 인용부호의 시작과 끝을 예측하기 위해 구두점도 토큰화 한다.
 > - 이야기사이에 줄바꿈 문자는 새로운 이야기를 위한 구분자 (||||||||) 로 바꾼다.
 > ```
 > seq_length = 20
 >
 > filename = "./data/aesop/data.txt"
 >
 > with open(filename, encoding='utf-8-sig') as f:
 >      text = f.read()
 > start_story = '| ' * seq_length
 >   
 > text = start_story + text
 > text = text.lower()
 > text = text.replace('\n\n\n\n\n', start_story)
 > text = text.replace('\n', ' ')
 > text = re.sub('  +', '. ', text).strip()
 > text = text.replace('..', '.')
 >
 > text = re.sub('([!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ ])', r' \1 ', text)
 > text = re.sub('\s{2,}', ' ', text)
 > if token_type == 'word':
 >   tokenizer = Tokenizer(char_level = False, filters = '')
 > else:
 >   tokenizer = Tokenizer(char_level = True, filters = '', lower = False)
 >   
 >   
 > tokenizer.fit_on_texts([text])
 >
 > total_words = len(tokenizer.word_index) + 1
 >
 > token_list = tokenizer.texts_to_sequences([text])[0]
 > ```
 > - 정제를 마친 원본 텍스트는 아래 그림과 같다.\
 > ![image](https://user-images.githubusercontent.com/70633080/104424354-9c589700-55c2-11eb-92bd-3c584f5fe93a.png)
 > - 인덱스에 매핑된 토큰 딕셔너리, 토큰화된 텍스트는 아래그림과 같다.\
 > ![image](https://user-images.githubusercontent.com/70633080/104424503-d033bc80-55c2-11eb-9b02-38916d8a05e5.png)
 > ### 2.2 데이터셋 구축
 > - LSTM은 단어의 시퀀스가 주어지면 이 시퀀스의 다음 단어를 예측하도록 훈련된다. 
 > - 따라서 시퀀스의 길이는 훈련 하이퍼파라미터이다.\
 > ex) 길이가 20인 시퀀스 사용, 텍스트를 20개의 단어 길이로 나눈다. 총 50,416개의 시퀀스가 만들어지므로 훈련데이터 셋 x는 [50416,20]크기의 배열이 된다.
 > - 각 시퀀스의 타깃은 다음 단어이다. 이 단어는 길이가 4,169인 벡터로 원핫인코딩 된다.
 > - 따라서 타깃 y는 [50416,4169]크기의 0또는 1을 가진 이진배열이 된다.
 > ```
 > def generate_sequences(token_list, step):
 >   
 >   X = []
 >   y = []
 >
 >   for i in range(0, len(token_list) - seq_length, step): # 인덱스 0 부터 마지막으로부터 20개 전까지
 >       X.append(token_list[i: i + seq_length])# 이전단어 , 인덱스 0 부터 마지막 -1 까지 봄
 >       y.append(token_list[i + seq_length]) # 다음단어(정답), 인덱스 1 부터 마지막까지 봄
 >   
 >
 >   y = np_utils.to_categorical(y, num_classes = total_words) # np.utils.to_categorical은 원핫인코딩해주는 함수이다.
 >   
 >   num_seq = len(X)
 >   print('Number of sequences:', num_seq, "\n")
 >   
 >   return X, y, num_seq
 >
 > step = 1
 > seq_length = 20
 >
 > X, y, num_seq = generate_sequences(token_list, step)
 >
 > X = np.array(X)
 > y = np.array(y)
 > ```
 >
 > ### 2.3 LSTM모델 구조
 > - 전체 모델의 구조는 아래 그림과 같다.
 > - 모델의 입력 : 정수 토큰의 시퀀스
 > - 출력 : 시퀀스 다음에 어휘사전에서 등장 할 수 있는 단어의 확률\
 > ![image](https://user-images.githubusercontent.com/70633080/104426511-610b9780-55c5-11eb-9ee7-d97aee084be5.png)
 >
 > ### 2.4 임베딩 층
 > - 임베딩 층은 각 토큰을 embedding_size길이의 벡터로 변환하는 **룩업테이블**이다.
 > - 따라서 이층에서 학습되는 가중치의 수는 어휘사전의 크기 * embedding_size 이다.
 > - Input 층 : [batch_size, seq_length]크기의 정수 시퀀스 텐서를 embedding층으로 전달. 
 > - output : [batch_size,seq_length,embedding_size]크기의 텐서를 출력
 > - 이 출력 텐서는 LSTM층으로 전달된다.
 >
 > ### 2.5 LSTM 층
 > #### 일반적인 순환층
 > - 순환층은 cell로 구성된다.
 > - hidden state h1은 한번에 한 타임스텝씩 시퀀스 x_t의 각 원소를 cell로 전달해 업데이트 한다.
 > - hidden state는 cell안에 있는 유닛의 개수와 길이가 동일한 벡터이다. 
 > - 이를 시퀀스에 대한 현재 cell의 지식으로 생각 할 수 있다.
 > - 타임스텝 t에서 cell은 이전 hiden state h_t-1와 현재 타임스텝 x_t의 데이터를 이용해 업데이트된 hidden state h_t를 만든다.
 > - 이런 순환과정은 시퀀스가 끝날때까지 계속된다.
 > - 시퀀스가 끝나면 cell의 최종 hidden state h_n을 출력하고 네트워크의 다음층으로 전달한다.
 > - 아래그림은 하나의 시퀀스가 순환 층을 통과하는 과정이다.\
 > ![image](https://user-images.githubusercontent.com/70633080/104428415-cfe9f000-55c7-11eb-8fb8-493f1c1540f1.png)
 > - 이 그림에서의 모든 셀은 동일한 가중치를 공유한다. (실제로는 모두 동일한 셀이다. 그림에서만 나눠그린것)
 >
 > ### 2.6 LSTM cell
 > #### 일반적 순환층이 아닌 LSTM cell의 내부
 > - LSTM cell은 이전 hidden statd h_t-1과 현재 단어 임베딩 x_t가 주어졌을때, 새로운 hidden state h_t를 출력한다.
 > - h_t의 길이는 LSTM에있는 유닛의 개수와 동일하다. **(이는 층을 정의할때 정해야하는 하이퍼파라미터이다.)**
 > - LSTM층에는 하나의 cell이 있고 이 cell은 여러개의 유닛을 가진다.
 > - 하나의 LSTM cell은 하나의 cell상태 C_t를 관리한다.
 > - cell상태를 현재 시퀀스의 상태에 대한 cell내부의 생각으로 볼 수 있다.
 > - 마지막 타임스텝 후 cell에서 출력되는 은닉상태 h_t와는 구분된다. 
 > - cell상태는 은닉상태(hidden_state)와 동일한 길이를 가진다. (cell에 있는 유닛개수)\
 > ![image](https://user-images.githubusercontent.com/70633080/105182733-c1c04480-5b70-11eb-93fc-fc3b679c6a54.png)
 > - f_t (forget gate)\
 > : ‘과거 정보를 잊기’위한 게이트다. 시그모이드 함수의 출력 범위는 0 ~ 1 이기 때문에 그 값이 0이라면 이전 상태의 정보는 잊고, 1이라면 이전 상태의 정보를 온전히 기억하게 된다. 즉, 얼마나 이전정보(C_t-1)을 유지할 것인가
 > - i_t(input gate)\
 > : 현재 정보를 기억하기’위한 게이트다. 이 값은 시그모이드 이므로 0 ~ 1 이지만 C_t~는 tanh 함수를 거치므로 -1 ~ 1 이 된다. 따라서 결과는 음수가 될 수도 있다.
 > - o_t(output gate)\
 > : 최종 결과 h_t를 위한 게이트이며, 업데이트 된 (tanh를 거친 ) c_t를 얼마나 다음 state로 보낼지 결정한다.
 > - h_t
 > : tanh를 지난 c_t와 o_t를 원소별 곱셈을 하여 새로운 은닉상태 h_t를 생성한다.
 > - LSTM 네트워크 만드는 코드
 > ```
 > from keras.layers import Dense, LSTM, Input, Embedding, Dropout
 > from keras.models import Model
 > from keras.opmizers import RMSprop
 > 
 > n_units=256 # cell의 유닛 개수
 > embedding_size=100 # token을 임베딩할 개수
 > text_in=Input(shape=(None,))
 > x=Embedding(total_words, embedding_size)(text_in) # (in_num, out_num)(data)
 > x=LSTM(n_units)(x)
 > x=Dropout(0.2)(x)
 > text_out=Dense(total_words, activation='softmax')(x)
 > 
 > model=Model(text_in,text_out)
 > 
 > opti=RMSprop(lr=0.001)
 > model.compile(loss='categorial_crossentropy',optimizer=opti)
 >
 > epoch=100
 > batch_size=32
 > model.fit(X,y,epochs=epoch,batch_size=batch_size,shuffle=True)
 > ```
 
 
