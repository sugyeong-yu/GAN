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
 
 ## 3. 새로운 텍스트 생성
 > LSTM net을 컴파일하고 훈련해보자. 이 과정을 통해 이 네트워크로 긴 텍스트 문자열을 생성할 수 있다.
 > 1. 기존단어의 시퀀스를 네트워크에 넣고 다음단어를 예측한다.
 > 2. 이 단어를 기존 시퀀스에 추가하고 과정을 반복한다.\
 > 이 네트워크는 샘플링 할 수 있는 각 단어의 확률을 출력한다. 즉, 결정적이지 않고 확률적으로 텍스트를 생성할 수 있다. 
 > - temperature 매개변수\
 > : 샘플링 과정을 얼마나 결정적으로 만들지 지정할 수 있다.
 > - LSTM으로 텍스트 생성하기 코드
 > ```
 > def sample_with_temp(preds, temperature=1.0):
 >   #확률배열에서 인덱스 하나를 샘플링하는 헬퍼 함수
 >   # softmax를 적용하기 전에 temperature스케일 매개변수로 logit에 가중치를 부여한다.
 >   # temperature이 0에 가까울수록 샘플링을 더 결정적으로 만든다. ( 가장 높은 확률을 가진 단어가 선택될 가능성이 높다는 것) # **무슨소리??**
 >   preds = np.asarray(preds).astype('float64') 
 >   preds = np.log(preds) / temperature
 >   exp_preds = np.exp(preds)
 >   preds = exp_preds / np.sum(exp_preds)
 >   probas = np.random.multinomial(1, preds, 1)
 >   return np.argmax(probas)
 >
 > def generate_text(seed_text, next_words, model, max_sequence_len, temp):
 >   output_text = seed_text
 >   
 >   seed_text = start_story + seed_text # seed_text는 생성과정을 시작하기 위해 모델에 전달할 단어 시퀀스이다. (공백도 가능) 이는 이야기의 시작을 나타내는 문자블럭(||||||||)뒤에 붙여진다.
 >   
 >   for _ in range(next_words): # 시퀀ㅅ스길이만큼 반복 
 >       token_list = tokenizer.texts_to_sequences([seed_text])[0] # 단어를 토큰의 리스트로 변환한다. 
 >       token_list = token_list[-max_sequence_len:]# 마지막 max_sequence_len 개의 토큰만 유지한다. LSTM층이 어떤길이의 시퀀스도 입력으로 받을 수 있다. 하지만 시퀀스가 길수록 다음단어생성에 시간이 걸리므로 시퀀스의 길이ㅡ를 제한한다.  
 >       token_list = np.reshape(token_list, (1, max_sequence_len))
 >       
 >       probs = model.predict(token_list, verbose=0)[0] # 모델은 다음단어에 대한 확률을 출력한다.
 >       y_class = sample_with_temp(probs, temperature = temp) # 다음단어 출력을 위해 샘플링함수에 확률과 temperature매개변수를 전달한다.
 >       
 >       if y_class == 0:
 >           output_word = ''
 >       else:
 >           output_word = tokenizer.index_word[y_class]
 >           
 >       if output_word == "|": # 출력단어가 스토리의 시작이면 이야기가 끝나도 새로운 이야기를 시작해야할 때이므로 단어생성을 멈춘다.
 >           break
 >           
 >       if token_type == 'word':
 >           output_text += output_word + ' '
 >           seed_text += output_word + ' '
 >       else:
 >           output_text += output_word + ' ' # 그렇지 않으면 새로운 단어를 seed_text에 덧붙이고 다음 생성과정을 반복할 준비를 한다. 
 >           seed_text += output_word + ' '
 >           
 >           
 >   return output_text
 > ```
 > 
 ## 4. RNN확장
 > ### 1. 적층 순환 네트워크
 > 지금까지 하나의 LSTM 층이 포함된 네트워크를 보았다. LSTM층을 쌓은 네트워크도 훈련할 수 있다. 이로 인해 텍스트에서 더 깊은 특성을 학습할 수 있다.
 > - 첫번째 LSTM층의 return_sequences 매개변수를 True로 지정한다. 
 > : 순환층이 마지막 타임스텝의 은닉상태만 출력하지 않고 모든 타임스텝의 은닉상태를 출력한다.
 > - 두번째 LSTM층은 첫번째 층의 은닉상태를 입력데이터로 사용한다.
 > - 아래 그림은 다층 RNN의 종류 및 흐름을 나타낸다.\
 > ![image](https://user-images.githubusercontent.com/70633080/105208417-03aab400-5b8c-11eb-9f1b-ddf62380ea15.png)
 > - 적층 LSTM Network 만들기
 > ```
 > text_in=Input(shape=(None,))
 > embedding=Embedding(total_words,embedding_size)
 > x=enbedding(text_in)
 > x=LSTM(n_units,return_sequences=True)(x)
 > x=LSTM(n_units)(x)
 > text_out=Dense(total_words,activation='softmax')(x)
 >
 > model=Model(text_in,text_out)
 > ```
 >
 > ### 2. GRU층 (gated recurrent unit)
 > 널리 사용하는 또 다른 RNN층은 GRU이다.
 > - LSTM cell과의 주요 차이점
 >      1. 삭제게이트와 입력게이트가 리셋게이트와 업데이트 게이트로 바뀐다.
 >      2. 셀상태와 출력게이트가 없다. 셀은 은닉상태만 출력한다.
 > - 아래 그림은 GRU cell의 그림이다.\
 > ![image](https://user-images.githubusercontent.com/70633080/105210057-e840a880-5b8d-11eb-9f97-594cf31a1d71.png)
 > 1. reset gate\
 > : 이전 타임 스텝의 은닉상태 h_t-1과 현재 단어 임베딩 x_t가 연결되어 reset gate를 만든다. 
 >      - 이는 **가중치행렬 W_r, 절편 b_r과 시그모이드 활성화함수를 가진 완전연결층이다.**
 >      - 결과벡터 r_t는 셀의 유닛개수와 길이가 동일하고 **0과1사이의 값을 저장한다.** 
 >      - 이값은 셀의 새로운 생각을 계산하는데 **이전 은닉상태 h_t-1을 얼마나 제공할지 결정한다.**
 > 2. h_t~\
 > : h_t~는 cell의 새로운 생각을 저장하는 벡터이다. 
 >      - reset gate와 은닉상태 h_t-1과 원소별곱셈이 된 후 현재 단어 임베딩 x_t와 연결된다.
 >      - 이 벡터는 가중치행렬 W, 절편 b와 tanh활성화함수를 가진 완전연결층에 주입되어 생성된다.
 >      - 이 벡터는 cell의 유닛개수와 길이가 동일하고 -1~1 사이의 값을 저장한다.
 > 3. update gate\
 > : 이전 타임스텝의 은닉상태 h_t-1과 현재 단어 임베딩 x_t의 연결은 update gate를 만들때도 사용된다.
 >      - **가중치행렬 W_z,절편 b_z와 시그모이드 활성화 함수를 가진 완전 연결층이다.**
 >      - 결과벡터 z_t는 셀의 유닛개수와 길이가 동일하고 **0~1사이의 값을 저장한다.**
 >      - 이 값은 **새로운 생각 h_t~가 이전타임스텝의 은닉상태 h_t-1과 얼마나 섞일지 결정한다.**
 > 4. 업데이트 된 은닉상태 h_t\
 > : 셀의 새로운 생각 h_t~와 이전타임스텝의 은닉상태 h_t-1은 업데이트 게이트 z_t가 결정하는 비율로 섞여서 업데이트 된 은닉상태 h_t를 만든다.
 >      - 이것은 cell의 출력이 된다.
 >
 > ### 3. 양방향 셀
 > inference 시 ( 훈련된 모델을 사용해 예측을 만드는 것) 전체 텍스트를 모델에 제공할 수 있는 예측 문제에서는 시퀀스를 전진방향만으로 처리할 이유가 없다. 후진방향으로도 처리할 수 있다.
 > - 양방향 층 (bidirectional layer)\
 >      - 두개의 은닉상태를 사용한다.
 >      1. 하나는 일반적인 전진방향으로 처리되는 시퀀스의 결과를 저장.
 >      2. 다른 하나는 시퀀스가 후진방향으로 처리될 때 만들어진다.
 > - 케라스에서는 다음과 같이 순환층의 wrapper로 구현할 수 있다.
 > ```
 > layer=Bidirectional(GRU(100))
 > ```
 > - 만들어진 층의 은닉상태는 셀 유닛개수의 두배길이를 가진 벡터이다. (전진 & 후진)
 > - 따라서 본 예제에서의 층의 은닉상태는 길이가 200인 벡터이다.
 ## 5. 인코더-디코더 모델
 > - 앞에서 LSTM 층이 데이터를 순차적으로 처리하여 은닉상태를 업데이트하는 방법을 알아보았다. 이 은닉상태는 층이 가진 현재 시퀀스에 대한 지식이다. 마지막 은닉상태를 fc layer에 연결하면 네트워크가 다음 단어에 대한 확률 분포를 출력한다.
 > - 이 작업에서는 시퀀스의 다음 단어 하나를 예측하는 것이 목적이 아니다. 대신 입력 시퀀스에 관련된 완전한 다른 단어의 시퀀스를 예측해야한다. 
 > - 언어번역\
 > : 네트워크에 source 언어로 된 텍스트를 주입하고 타깃 언어로 번역된 텍스트를 출력하는 것이 목적이다.
 > - 질문생성\
 > : 네트워크에 텍스트 문장을 주입하고 텍스트에 관해 가능한 질문을 생성하는 것이 목적이다.
 > - 텍스트 요약\
 > : 네트워크에 긴 텍스트 문장을 주입하고 이 문장의 짧은요약을 생성하는 것이 목적이다.
 > - 이런종류의 문제에 인코더-디코더라 불리는 네트워크 구조를 사용할 수 있다.
 > - 순차데이터에서의 인코더-디코더 동작과정
 > 1. 원본입력시퀀스는 인코더 RNN에 의해 하나의 벡터로 요청된다.
 > 2. 이 벡터는 디코더 RNN의 초기값으로 사용된다.
 > 3. 각 타임스텝에서 디코더 RNN의 은닉상태는 fc층에 연결되어 단어 어휘사전에 대한 확률 분포를 출력합니다. 이런식으로 인코더가 생성한 입력데이터로 디코더를 초기화 한다음 새로운 텍스트 시퀀스를 생성할 수 있다.\
 > ![image](https://user-images.githubusercontent.com/70633080/105316212-eb767b80-5c03-11eb-8620-6e5354bea1e7.png)
 > - 인코더의 마지막 은닉상태 = 전체 입력문서에 대한 하나의 표현
 > - 디코더는 이 하나의 표현을 순차적인 출력으로 변환 ( 훈련과정에서 다음단어를 생성할 필요는 없다. 이전 출력 분포에서 단어를 샘플링하는대신 그 다음 타임스텝에 진짜 다음단어가 셀에 주입되기 때문)
 > - 이러한 훈련방식을 **티처보상**이라고한다.
 ## 6. 질문-대답 생성기
 > 지금까지 배운 것을 모아서 텍스트에 대한 질문과 대답을 생성하는 모델을 만들어본다.
 > 모델은 두개의 부분으로 구성된다.
 >      1.  RNN 하나가 텍스트로부터 대답 후보를 고른다.
 >      2. 인코더-디코더 네트워크가 RNN이 선택한 대답후보 중 하나가 주어졌을때 적절한 질문을 생성한다.
 >  
 > ### 1. 질문-대답 데이터셋
 > - 말루바 NewsQA데이터셋을 사용한다.
 > - <http://bit.ly/2Ky7uJq>에 적힌 안내를 따라 다운로드 가능.
 > 1. story_id \
 > : story에 대한 고유 식별자
 > 2. story_next\
 > :  story의 text (문장)
 > 3. question\
 > : story text에 대한 질문
 > 4. answer_token_ranges\
 > : answer. 스토리 텍스트에서 대답의 토큰위치. ex) 24:27 , 이는 모델의 입력으로 사용하기 위해 원본데이터를 전처리하고 토큰화 한다.
 > - 변환한 후 훈련세트에 있는 각 샘플은 다음과 같은 다섯개의 특성으로 구성된다.
 > 1. document_tokens\
 > : 토큰화된 story text( 예를들어 [1,4633,7,66,11,...]).max_document_length길이로 클리핑되거나 제로패딩한다.
 > 2. question_input_tokens\
 > : 토큰화된 질문 (예를 들어 [2,39,1,52,...]).max_question_length( 또다른 파라미터)의 길이로 제로패딩된다.
 > 3. question_output_tokens\
 > : 한 타임스텝 밀린 토큰화된 질문 (예를들어 [39,1,52,1866,...]).max_question_length길이로 제로패딩된다.
 > 4. answer_masks\
 > : [max_answer_length, max_document_lenth] 크기의 이진 마스크 행렬, 질문에 대한 대답의 i번째 단어가 문서의 j번재 단어에 있다면 이 행렬의 [i,j]의 값이 1이고 그렇지않으면 0이다.
 > 5. answer_labels\
 > : max_document_length 길이의 이진벡터(예를들어, [0,1,1,0,...]). 문서의 i번째 단어가 대답의 일부분이면 이 벡터의 i번째원소가 1이고 그렇지않으면 0.
 > ### 2. 모델구조
 > - 아래 그림은 전체모델구조이다.
 > 
 > - 06_02_qa_train.ipynb 참고
 > 코드
 > - Glove 단어벡터 \
 >      - glove project 웹사이트에서 glove.6B.100d.txt 파일을 다운로드해야한다.
 >      - 그 다음 이책의 저장소에서 훈련데이터에 포함된 단어만 골라낸다.
 > ```
 > python ./utils/write.py
 > ```
 > - 모델의 두번째 부분은 대답을 받고 이에 맞는 질문을 구성하는 인코더-디코더 네트워크이다.
 > 코드
 >
 > ### 3. 예측
 > 이전에 본적없는 입력문서시퀀스에서 모델을 테스트하기위해 다음과정을 수행한다.
 > 1. 대답생성기에 문서텍스트를 주입해 대답의 샘플위치를 생성한다.
 > 2. 이 대답 중 하나르 선택해 인코더-디코더 질문 생성기로 전달한다.(적절한 대답 마스크를 만든다.)
 > 3. 문서와 대답마스크를 인코더에 주입하여 디코더의 초기상태를 생성한다.
 > 4. 이 초기상태로 디코더를 초기화하고 <start>토큰을 주입해 질문의 첫번째단어를 생성한다. 모델이 < end > 토큰을 예측할 때가지 하나씩 생성된 단어를 주입하여 이 과정을 반복한다.
 > - 훈련하는 동안 이 모델은 다음단어를 예측하는 대신 티처포싱을사용해 정답 단어를 디코더 셀로 주입한다. 
 > - 하지만 예측에서는 모델이 질문을 스스로 생성해야한다. 따라서 은닉상태를 유지하면서 예측된 단어를 다시 디코더셀로 주입할 수 있어야한다.
 > - 이를 처리하는 한가지 방법은 또다른 케라스 모델을 하나 정의하는 것이다.
 > - 이 모델은 현재 단어토큰과 현재 디코더의 은닉상태를 입력으로 받고 예측된 다음 단어의 분포를 출력하고 디코더의 은닉상태를 업데이트한다.
 > ```
 > 코드
 > ```
 > ### 4. 결과
 > - 06_03_qa_analysis.ipynb 를 참고.
 
 
