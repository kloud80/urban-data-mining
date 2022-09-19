# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 17:22:38 2021

@author: Kloud


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

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from graphviz import Source
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


mpl.rc('font', family='gulim')  # 한글 폰트 적용시
os.chdir('03 Dtree Ensemble/')

"""학습용 데이터셋을 불러옴"""

sdot_data_total = pd.read_csv('data/sdot학습데이터.csv', sep='|', encoding='cp949')
"""전체 Sdot 평균기온과의 온도차 평균이 높으면 1, 낮으면 0으로 종속변수 생성"""
sdot_data_total['종속'] = sdot_data_total['온도차이'].apply(lambda x: 0 if x < 0 else 1)

tmp = sdot_data_total.dtypes

tmp = sdot_data_total.mean()

""" Tree 모형 분석을 위하 주변 도로 면적비율과, 대지면적 비율 만 불러옴 (도=X, 대=y)"""
tmp = sdot_data_total[['도', '대', '종속']]

""" plot으로 고온그룹과 저온그룹을 2차원에 표시"""
x1 = np.array(tmp[tmp['종속'] == 1][['도', '대']].fillna(0).astype('float').values)
y1 = np.array(tmp[tmp['종속'] == 1]['종속'].values)
y1 = y1.reshape(y1.shape[0], 1)

x2 = np.array(tmp[tmp['종속'] == 0][['도', '대']].fillna(0).astype('float').values)
y2 = np.array(tmp[tmp['종속'] == 0]['종속'].values)
y2 = y2.reshape(y2.shape[0], 1)

plt.figure(figsize=(15, 15))
plt.scatter(x=x1[:, 0], y=x1[:, 1], marker='x', color='red', label='고온')
plt.scatter(x=x2[:, 0], y=x2[:, 1], marker='o', color='blue', label='저온')
plt.legend(fontsize=20)
plt.show()

#%%
""" 모든 입력변수를 이용한 분석"""
tmp = sdot_data_total

x = np.array(tmp[tmp.columns.drop(['종속', '시리얼번호', '온도차이', '온도비율차이'])].fillna(0).astype('float').values)
y = np.array(tmp['종속'].values)
y = y.reshape(y.shape[0], 1)

depth_test = np.array([])
for depth in range(1, 21, 1):
    tree_clf = DecisionTreeClassifier(max_depth=depth)
    tree_clf.fit(x, y)
    score_tr = tree_clf.score(x, y)

    depth_test = np.append(depth_test, [depth, score_tr])

depth_test = depth_test.reshape(int(depth_test.shape[0] / 2), 2)

print(depth_test)


#%%


#%%
"""Bagging Tree"""
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier

"""데이터 나누기"""
#데이터를 training과 testing으로 나눈다
x_train, x_test, y_train, y_test = train_test_split(x,y,                #분할할 행렬
                                                    test_size=0.25,     #검증세트의 분할 비율
                                                    random_state=1,     #순서 섞을 경우 랜덤시드 고정용
                                                    shuffle=True,       #순서 섞을지
                                                    stratify=y)         #분할시 데이터 비율 유지
print(sum(y) / len(y))
print(sum(y_train) / len(y_train))
print(sum(y_test) / len(y_test))

#%%


"""배깅트리"""
model_baggingtree = BaggingClassifier(DecisionTreeClassifier(max_depth=3),
                          n_estimators=100)

model_baggingtree.fit(x_train, y_train)
print(model_baggingtree.score(x_train, y_train))
print(model_baggingtree.score(x_test, y_test))

m = model_baggingtree[2]
dt_dot_data  = export_graphviz(m,
                               feature_names=tmp.columns.drop(['종속', '시리얼번호', '온도차이', '온도비율차이']),
                               class_names=['low', 'high'],         # 종속변수
                               rounded = True,
                               filled = True)


gp = Source(dt_dot_data)
gp.format = 'svg'
img = gp.render('dtree_render',view=True)

tr = model_baggingtree[0].tree_

"""변수중요도"""
feature_imp = np.mean([
    tree.feature_importances_ for tree in model_baggingtree.estimators_
], axis=0)
n_feature = len(tmp.columns.drop(['종속', '시리얼번호', '온도차이', '온도비율차이']))
idx = np.arange(n_feature)
sorted_idx = feature_imp.argsort()

plt.figure(figsize=(5, 25))
plt.barh(idx, feature_imp[sorted_idx], align='center')
plt.yticks(idx, tmp.columns.drop(['종속', '시리얼번호', '온도차이', '온도비율차이'])[sorted_idx])
plt.xlabel('feature importance', size=15)
plt.ylabel('feature', size=15)
plt.show()
"""변수중요도"""

#%%
from sklearn.ensemble import RandomForestClassifier

model_RF = RandomForestClassifier(max_depth = 3, n_estimators=100, random_state=1)

model_RF.fit(x_train, y_train)
print(model_RF.score(x_train, y_train))
print(model_RF.score(x_test, y_test))



m = model_RF[0]
dt_dot_data  = export_graphviz(m,
                               feature_names=tmp.columns.drop(['종속', '시리얼번호', '온도차이', '온도비율차이']),
                               class_names=['저온', '고온'],         # 종속변수
                               rounded = True,
                               filled = True)

gp = Source(dt_dot_data)
gp.format = 'svg'

img = gp.render('dtree_render',view=True)


trb = model_baggingtree[0].tree_
trr = model_RF[2].tree_


"""변수중요도"""
feature_imp = np.mean([
    tree.feature_importances_ for tree in model_RF.estimators_
], axis=0)
n_feature = len(tmp.columns.drop(['종속', '시리얼번호', '온도차이', '온도비율차이']))
idx = np.arange(n_feature)
sorted_idx = feature_imp.argsort()

plt.figure(figsize=(5, 25))
plt.barh(idx, feature_imp[sorted_idx], align='center')
plt.yticks(idx, tmp.columns.drop(['종속', '시리얼번호', '온도차이', '온도비율차이'])[sorted_idx])
plt.xlabel('feature importance', size=15)
plt.ylabel('feature', size=15)
plt.show()
"""변수중요도"""

#%%
from sklearn.inspection import permutation_importance

result = permutation_importance(model_RF, x_test, y_test, n_repeats=10,
                                random_state=42, n_jobs=2)
sorted_idx = result.importances_mean.argsort()


plt.figure(figsize=(5, 25))
plt.boxplot(result.importances[sorted_idx].T,
           vert=False, labels=tmp.columns.drop(['종속', '시리얼번호', '온도차이', '온도비율차이'])[sorted_idx])

plt.show()

