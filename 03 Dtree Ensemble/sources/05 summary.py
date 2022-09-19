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
"""학습용 데이터셋을 불러옴"""

sdot_data_total = pd.read_csv('data/sdot학습데이터.csv', sep='|', encoding='cp949')
sdot_data_total = sdot_data_total[~sdot_data_total['공'].isnull()]
"""전체 Sdot 평균기온과의 온도차 평균이 높으면 1, 낮으면 -1으로 종속변수 생성"""
sdot_data_total['종속'] = sdot_data_total['온도차이'].apply(lambda x: 0 if x < 0 else 1)

# %%

tmp = sdot_data_total

x = np.array(tmp[tmp.columns.drop(['종속', '시리얼번호', '온도차이', '온도비율차이'])].fillna(0).astype('float').values)
y = np.array(tmp['종속'].values)
y = y.reshape(y.shape[0], 1)

from sklearn.model_selection import train_test_split

"""데이터 나누기"""
# 데이터를 training과 testing으로 나눈다
x_train, x_test, y_train, y_test = train_test_split(x, y,  # 분할할 행렬
                                                    test_size=0.25,  # 검증세트의 분할 비율
                                                    random_state=1,  # 순서 섞을 경우 랜덤시드 고정용
                                                    shuffle=True,  # 순서 섞을지
                                                    stratify=y)  # 분할시 데이터 비율 유지

# %%

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

# %%

model_report = {}
model_report['Decision_tree'] = {}
model_report['Bagging_tree'] = {}
model_report['RandomForest'] = {}
model_report['Ada_boost'] = {}
model_report['GBM'] = {}

model_report['Decision_tree']['model'] = DecisionTreeClassifier()
model_report['Bagging_tree']['model'] = BaggingClassifier(DecisionTreeClassifier(), n_estimators=100)
model_report['RandomForest']['model'] = RandomForestClassifier(n_estimators=100)
model_report['Ada_boost']['model'] = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=100)
model_report['GBM']['model'] = GradientBoostingClassifier()

# %%
for key in model_report:
    start = time.time()
    model_report[key]['model'].fit(x_train, y_train)
    model_report[key]['learning time'] = time.time() - start

    y_pred = model_report[key]['model'].predict(x_train)

    model_report[key]['train_confusition_matrix'] = confusion_matrix(y_train, y_pred)
    model_report[key]['train_precision_score'] = precision_score(y_train, y_pred)
    model_report[key]['train_recall_score'] = recall_score(y_train, y_pred)
    model_report[key]['train_accuracy_score'] = accuracy_score(y_train, y_pred)
    model_report[key]['train_f1_score'] = f1_score(y_train, y_pred)

    y_pred = model_report[key]['model'].predict(x_test)

    model_report[key]['test_confusition_matrix'] = confusion_matrix(y_test, y_pred)
    model_report[key]['test_precision_score'] = precision_score(y_test, y_pred)
    model_report[key]['test_recall_score'] = recall_score(y_test, y_pred)
    model_report[key]['test_accuracy_score'] = accuracy_score(y_test, y_pred)
    model_report[key]['test_f1_score'] = f1_score(y_test, y_pred)

    if key == 'Bagging_tree':  # feature_importance 계산법 다름
        model_report[key]['feature_importances'] = np.mean([tree.feature_importances_
                                                            for tree in model_report[key]['model'].estimators_],
                                                           axis=0)
    else:
        model_report[key]['feature_importances'] = model_report[key]['model'].feature_importances_

    model_report[key]['permutation_importances'] = permutation_importance(model_report[key]['model'],
                                                                          x_test, y_test, n_repeats=5)


# %%

def display_feature_importance(feature_imp):
    n_feature = len(tmp.columns.drop(['종속', '시리얼번호', '온도차이', '온도비율차이']))
    idx = np.arange(n_feature)
    sorted_idx = feature_imp.argsort()

    plt.figure(figsize=(5, 25))
    plt.barh(idx, feature_imp[sorted_idx], align='center')
    plt.yticks(idx, tmp.columns.drop(['종속', '시리얼번호', '온도차이', '온도비율차이'])[sorted_idx])
    plt.xlabel('feature importance', size=15)
    plt.ylabel('feature', size=15)
    plt.show()


def display_permutation_importance(permute_imp):
    sorted_idx = permute_imp.importances_mean.argsort()

    plt.figure(figsize=(5, 25))
    plt.boxplot(permute_imp.importances[sorted_idx].T,
                vert=False, labels=tmp.columns.drop(['종속', '시리얼번호', '온도차이', '온도비율차이'])[sorted_idx])
    plt.show()


# %%
for key in model_report:
    print(key)
    display_feature_importance(model_report[key]['feature_importances'])
    display_permutation_importance(model_report[key]['permutation_importances'])
