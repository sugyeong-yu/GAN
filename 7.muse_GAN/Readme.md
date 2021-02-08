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
- 1 : chordify method를 사용해 여러파트로 나누어진 음표를 하나의 파트에서 동시에 연주되는 화음으로 압축하여 show
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
