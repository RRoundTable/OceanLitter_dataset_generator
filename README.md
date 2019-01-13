# OceanLitter_dataset_generator
해양쓰레기 데이터를 생성해주는 Cycle_GAN입니다.


## overview

해양오염 이슈가 국내뿐만 아니라, 해외에서도 이슈로 떠오르고 있습니다.
그 중, 해양쓰레기를 검출하는 이슈는 'object detection'기술로 충분히 개선할 수 있는 상황입니다.

하지만, 해양쓰레기에 대한 이미지 데이터셋은 많이 부족한 상황입니다. 

이러한 상황을 개선하기 위해서 CycleGAN을 이용하여 데이터 셋을 생성하는 프로젝트입니다.


## model

![CycleGAN](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/cyclegan-vc/images/training_procedure.png)

- 모델설명

- A set과 B set 사이의 특징을 학습시켜, A를 B처럼 바꿀 수 있습니다.
  배경 혹은 스타일 등등을 바꿀 수 있습니다.
  
- paired data가 필요없는 것이 큰 특징입니다.



![CycleGAN](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQsZKBaOB1ivYwK7vi_GpllECgvPOC2WFbf-0rxKn6-IA4TB0pn)
- 위의 사진과 같이, 

## dataset



## result


## reference
