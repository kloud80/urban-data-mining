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

mpl.rc('font', family='NanumGothic') #한글 폰트 적용시
os.chdir('03 Dtree Ensemble/')

"""학습용 데이터셋을 불러옴"""

sdot_data_total = pd.read_csv('data/sdot학습데이터.csv', sep='|', encoding='cp949')
"""전체 Sdot 평균기온과의 온도차 평균이 높으면 1, 낮으면 0으로 종속변수 생성"""
sdot_data_total['종속'] = sdot_data_total['온도차이'].apply(lambda x : 0 if x < 0 else 1)


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
plt.scatter(x=x1[:,0], y=x1[:,1], marker='x', color='red', label='고온')
plt.scatter(x=x2[:,0], y=x2[:,1], marker='o', color='blue', label='저온')
plt.legend(fontsize=20)
plt.show()


#%%
tmp = sdot_data_total[['도', '대', '종속']].fillna(0)
tmp['종속']

"""학습을 위하 pandas를 numpy로 변환하여 x와 y 배열 생성"""
x = np.array(tmp[['도', '대']].astype('float').values)
y = np.array(tmp['종속'].values)
y = y.reshape(y.shape[0], 1) #x배열과 shape를 같게 reshape

"""지니인덱스 계산 함수"""
def GiniIndex(y):
    total = len(y)
    G = 1
    for c in np.unique(y): #종속변수의 갯수로 loop
        # print(str(c) + "값 : " + str(np.power(np.where(y == c, 1, 0).sum() / total, 2)))
        G = G - np.power(np.where(y == c, 1, 0).sum() / total, 2)
    return G


print(str(GiniIndex(y)))
result = '고온' if y.sum() > len(y)/2 else '저온'
len(y) - y.sum()

#%%
"""입력변수를 정렬한 후 모든 구간에서 잘라서 giniindex를 계산하여 출력 """

criteria = x[:,0]
criteria = np.sort(np.unique(criteria))
total = len(y)
I = np.array([])
for f,l in zip(criteria[:-1], criteria[1:]):
    split = np.mean([f, l])
    
    s1 = y[np.where(x[:,0] < split, True, False)]
    s2 = y[np.where(x[:,0] > split, True, False)]
    
    Gini = len(s1) / total *  GiniIndex(s1) + len(s2) / total * GiniIndex(s2)
    
    I = np.append(I, np.array([f, l, split, Gini]))

I = I.reshape(int(I.shape[0]/4), 4)


plt.figure(figsize=(15, 15))
plt.scatter(x=I[:,2], y=I[:,3],  marker='o', color='grey', label='Gini')
plt.show()


#%%
""" 분할 지니인덱스 계산을 함수로 변환하여 모든 입력 변수에 대해서 계산하여 출력"""
def split_loop(x,y):
    criteria = x
    criteria = np.sort(np.unique(criteria))
    total = len(y)
    I = np.array([])
    for f,l in zip(criteria[:-1], criteria[1:]):
        split = np.mean([f, l])
        
        s1 = y[np.where(x < split, True, False)]
        s2 = y[np.where(x > split, True, False)]
        
        Gini = len(s1) / total *  GiniIndex(s1) + len(s2) / total * GiniIndex(s2)
        
        I = np.append(I, np.array([f, l, split, Gini]))
    
    I = I.reshape(int(I.shape[0]/4), 4)
    return I

I1 = split_loop(x[:,0], y)
I2 = split_loop(x[:,1], y)

plt.figure(figsize=(15, 15))
plt.scatter(x=I1[:,2], y=I1[:,3],  marker='o', color='blue', label='도로')
plt.scatter(x=I2[:,2], y=I2[:,3],  marker='x', color='red', label='대지')
plt.legend(fontsize=20)
plt.show()

I1[:,3].min()
I2[:,3].min()

split = I2[np.where(I2[:,3] == I2[:,3].min(), True, False)][0,2]

y1 = y[np.where(x[:,1] < split, True, False)]
y2 = y[np.where(x[:,1] > split, True, False)]

len(y1)
len(y2)
print(GiniIndex(y1))
result1 = '고온' if y1.sum() > len(y1)/2 else '저온'
len(y1) - y1.sum()

print(GiniIndex(y2))
result2 = '고온' if y2.sum() > len(y2)/2 else '저온'
len(y2) - y2.sum()


#%%
"""
사이킷런 라이브러리를 이용하여 Dtree 분석하기
os.environ["PATH"] += os.pathsep + r'c:\Program Files\Graphviz\bin\\'
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from graphviz import Source
from sklearn.metrics import classification_report, confusion_matrix

tree_clf = DecisionTreeClassifier(max_depth=5)
tree_clf.fit(x,y)

score_tr = tree_clf.score(x, y)


dt_dot_data  = export_graphviz(tree_clf,
                               feature_names=['도', '대'],
                               class_names=['low', 'high'],         # 종속변수
                               rounded = True,
                               filled = True)



gp = Source(dt_dot_data)

gp.format = 'svg'
img = gp.render('dtree_render',view=True)


"""변수중요도"""
feature_imp = tree_clf.feature_importances_
n_feature = len(['도', '대'])
idx = np.arange(n_feature)

plt.figure(figsize=(5, 1))
plt.barh(idx, feature_imp, align='center')
plt.yticks(idx, ['도', '대'])
plt.xlabel('feature importance', size=15)
plt.ylabel('feature', size=15)
for i, fi in zip(idx, feature_imp):
    plt.text(0.5, i,'%s' %fi, va='center', ha='center')
plt.show()

#%%
""" Depth 변화에 따른 정확도 차이 분석 = 과적합"""

depth_test = np.array([])
for depth in range(1, 21, 1) :
    tree_clf = DecisionTreeClassifier(max_depth=depth)
    tree_clf.fit(x,y)
    score_tr = tree_clf.score(x, y)
    
    depth_test = np.append(depth_test, [depth, score_tr])

depth_test = depth_test.reshape(int(depth_test.shape[0]/2), 2)

print(depth_test)


#%%
""" 이진분류 성능평가 지표"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix


tree_clf = DecisionTreeClassifier(max_depth=6)
tree_clf.fit(x,y)

y_pred = tree_clf.predict(x)

n = np.concatenate((y,  y_pred.reshape([y_pred.shape[0],1])), axis=1)

print("Confusition matrix: \n{}".format(confusion_matrix(y,y_pred)))
print("precision_score: {}".format( precision_score(y,y_pred)))
print("recall_score: {}".format( recall_score(y,y_pred)))
print("accuracy_score: {}".format( accuracy_score(y,y_pred)))
print("F1 Score: {}".format( f1_score(y,y_pred)))


#%%
#ROC 커브
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

fpr, tpr, thresholds = roc_curve(y, tree_clf.predict_proba(x)[:, 1])
fpr, tpr, thresholds

plt.plot(fpr, tpr, 'o-', label="Logistic Regression")
plt.plot([0, 1], [0, 1], 'k--', label="random guess")
# plt.plot([fallout], [recall], 'ro', ms=10)
plt.xlabel('FPR')
plt.ylabel('TPR(Recall)')
plt.title('Dtree ROC Curve')
plt.show()

rand_score = roc_auc_score(y, tree_clf.predict_proba(x)[:,1])
print('AUC : ' + str(rand_score))

