# Chapter 5.write_GAN
- style transfer 
> style image가 주어졌을때, 같은부류에 속한것 같은 느낌을 주도록 base image를 변환하는 모델을 훈련하는 것.\
> style image에 내재된 분포 를 모델링하는 것이 아닌 image에서 style을 결정하는 요소만 추출하여 base image에 주입하는 것.\
> :exclamation: style image와 base image를 보간방법으로 합쳐서는 안된다. \
> style image의 내용이 비쳐보이거나 색이 탁하고 흐릿해지기때문이다.\
> 또한, 하나의 이미지를 사용하는 것이아니라 style image set 전체에서 아티스트의 style을 찾아낸다. \
> 아티스트가 base image를 이용해 원복작품을 유지하면서 다른작품의 스타일기교로 완성하는 듯한 효과를 원하는것이다.\

