# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 17:22:38 2021

@author: Kloud

GBM

"""
import pandas as pd
import numpy as np
import geopandas as gpd
import os, re
from glob import glob
from tqdm import tqdm
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

mpl.rc('font', family='NanumGothic')  # 한글 폰트 적용시
os.chdir('03 Dtree Ensemble/')

# %%
"""지니인덱스 계산 함수"""


def SplitIndex(y):
    mean = y.mean()
    ret = np.power(y - mean, 2) / len(y)
    return ret.sum()


"""분할 기준 찾기"""


def split_loop(x, y):
    criteria = x
    criteria = np.sort(np.unique(criteria))
    total = len(y)
    I = np.array([])
    for f, l in zip(criteria[:-1], criteria[1:]):
        split = np.mean([f, l])

        s1 = y[np.where(x < split, True, False)]
        s2 = y[np.where(x > split, True, False)]

        Gini = len(s1) / total * SplitIndex(s1) + len(s2) / total * SplitIndex(s2)

        I = np.append(I, np.array([f, l, split, Gini]))

    I = I.reshape(int(I.shape[0] / 4), 4)
    return I


def display_chart(x, y):
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (10, 10)
    plt.rcParams['font.size'] = 7
    plt.rcParams['font.family'] = 'NanumGothic'

    fig, ax = plt.subplots()

    ax.scatter(x=x, y=y, marker='o', color='blue')
    plt.show()


def display_chart_2(x, y, y_pred, split):
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (10, 10)
    plt.rcParams['font.size'] = 7
    plt.rcParams['font.family'] = 'NanumGothic'

    fig, ax = plt.subplots()

    ax.scatter(x=x, y=y, s=2, marker='o', color='blue')
    ax.scatter(x=x, y=y_pred, s=2, marker='o', color='black')
    ax.axvline(x=split, color='r', linewidth=1)
    plt.show()


# %%
"""학습용 데이터셋을 불러옴"""

sdot_data_total = pd.read_csv('data/sdot학습데이터.csv', sep='|', encoding='cp949')
sdot_data_total = sdot_data_total[~sdot_data_total['공'].isnull()]
"""전체 Sdot 평균기온과의 온도차 평균이 높으면 1, 낮으면 -1으로 종속변수 생성"""
# sdot_data_total['종속'] = sdot_data_total['온도차이'].apply(lambda x : -1 if x < 0 else 1)

tmp = sdot_data_total[['도', '온도차이']].fillna(0)

"""학습을 위하 pandas를 numpy로 변환하여 x와 y 배열 생성"""
x = np.array(tmp[['도']].astype('float').values)
y = np.array(tmp[['온도차이']].astype('float').values)

display_chart(x, y)

# %%
"""GBM"""

n_estimator = 100  # 100개 모델
cnt = x.shape[0]  # 전체 데이터 수를 샌다

split = np.zeros((n_estimator, 1), float)
pred_small = np.zeros((n_estimator, 1), float)
pred_big = np.zeros((n_estimator, 1), float)

bPrint = True

x_1 = x.copy()
y_1 = y.copy()

for m in tqdm(range(n_estimator)):

    # Gini Index 루프로 분할 기준 정하기
    I = split_loop(x_1[:, 0], y_1)
    # plt.figure(figsize=(10, 10))
    # plt.scatter(x=I1[:,2], y=I1[:,3],  marker='o', color='blue')
    # plt.legend(fontsize=20)
    # plt.show()

    split[m, 0] = I[np.where(I[:, 3] == I[:, 3].min(), True, False)][0, 2]

    y_small = y_1[np.where(x_1[:, 0] < split[m, 0], True, False)].mean()
    y_big = y_1[np.where(x_1[:, 0] > split[m, 0], True, False)].mean()

    pred_small[m, 0] = y_small
    pred_big[m, 0] = y_big

    # 샘플데이터의 예측 값 계산
    y_1_pred = np.where(x_1[:, 0] < split[m, 0], y_small, y_big)
    y_1_pred = y_1_pred.reshape(y_1_pred.shape[0], 1)

    y_1_res = y_1 - y_1_pred  # 잔차 샘플 만들기

    # 변경된 추출확률을 그래플 표시, 분류기에서 오답인 데이터셋의 추출확률 증가(표식 크기로 구분)
    if bPrint:
        display_chart_2(x_1, y_1, y_1_pred, split[m, 0])

        key = input("출력중지(c), 종료(q):")
        if key == 'q':
            break
        if key == 'c':
            bPrint = False
            continue
    y_1 = y_1_res.copy()

# %%
pred = np.zeros((n_estimator, cnt), float)
for m in range(n_estimator):
    pred[m, :] = np.where(x[:, 0] < split[m, 0], pred_small[m, 0], pred_big[m, 0])

pred_sum = pred.sum(axis=0)
pred_sum = pred_sum.reshape(pred_sum.shape[0], 1)
display_chart(x, pred_sum)
display_chart(x, y)

for i in range(10, n_estimator + 1, 10):
    pred_sum = pred[:i, :].sum(axis=0)
    pred_sum = pred_sum.reshape(pred_sum.shape[0], 1)
    display_chart(x, pred_sum)
    print(str(i) + '번째 모델')
    key = input("종료(q):")
    if key == 'q': break
# display_chart(x, y, np.full((y.shape[0],1), 4))

# %%

from sklearn.ensemble import GradientBoostingRegressor

gbc = GradientBoostingRegressor(random_state=0, n_estimators=100, learning_rate=1)
gbc.fit(x, y[:, 0])
y_pred = gbc.predict(x)
display_chart(x, y_pred)


