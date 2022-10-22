# -*- coding: utf-8 -*-
"""

@author: Bigvalue_Bigdata Lab

작성자 : 구름
목적 : KERAS를 이용한 다층퍼셉트론 구현 예시
"""
import numpy as np
from matplotlib import pyplot as plt

# ds=[[0,0,0],[0,1,0],[1,0,0],[1,1,1]] #AND
#ds=[[0,0,0],[0,1,1],[1,0,1],[1,1,1]] #OR
ds=[[0,0,0],[0,1,1],[1,0,1],[1,1,0]] #XOR

"""데이터 재구성"""
nds = np.array(ds) #넘파이로 변환

x_train = nds[:,:2]
y_train = nds[:,2:]

#%%
from keras.models import Sequential #개별 레이어른 선형적으로 적제하기 위한 모델
from keras.layers import Dense #일반적인 형태의 뉴럴네트워크 계층 / 앞선 학습에 사용한 은닉/출력층에 해당
from keras import optimizers

model = Sequential() #모델을 선언한다
model.add(Dense(units=5, activation='relu', input_shape=(2,))) #은닉층 2개 추가, 활성함수 시그모이드
# model.add(Dense(units=2, activation='relu', input_shape=(2,))) #은닉층 2개 추가, 활성함수 시그모이드
model.add(Dense(units=1, activation='sigmoid')) # 출력층 1개 추가, 활성함수 시그모이드

model.summary() #모델 요약

"""
Model: "sequential"                                               << 선형 모델입니다.
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 2)                 6          << 은닉층 2개에 파라미터가 6개 (W1 4개, B1 2개)
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 3          << 출력층 1개에 파라미터가 3개 (W2 2개, B2 1개)
=================================================================
Total params: 9                                                   << 전체 파라미터는 9개 입니다.
Trainable params: 9                                               << 학습 파라미터는 9개 입니다.
Non-trainable params: 0
_________________________________________________________________
"""
#loss함수와 학습률결정모델(optimizer) 선택하여 모델 컴파일
model.compile(loss='mse', optimizer='adam') #learning_rate=0.1

#학습을 시작한다. 3천번 돌아 주세요.
history  = model.fit(x_train, y_train, epochs=3000)


# 학습 정확성 값과 검증 정확성 값을 플롯팅 합니다.
plt.plot(history.history['loss'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Loss'], loc='upper left')
plt.show()

#%%
#예측 결과를 표시
model.predict(x_train)

x_test = np.array([[1,2]])
model.predict(x_test)

model.weights[0]
