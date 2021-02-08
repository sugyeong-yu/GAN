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
- [07_01_notation_compose][0701]참고
[0701]: 주소
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
- [07_02_lstm_compose_train.ipynb][07_02] 참고
[07_02]:
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
