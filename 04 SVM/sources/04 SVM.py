# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 20:23:54 2021

@author: Kloud
"""
import pandas as pd
import numpy as np
import geopandas as gpd
import os, re, sys
from glob import glob
from tqdm import tqdm
import time
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance


mpl.rc('font', family='gulim') #한글 폰트 적용시
os.chdir('04 SVM/')

"""----------------------------------------------
data 소스 파일 다운로드 : https://www.dropbox.com/s/gqa6jxfvevbu5yx/data.zip?dl=0
---------------------------------------------"""
#%%

data = pd.read_csv('../data/생활인구_학습데이터.txt', sep='|', encoding='cp949')
data.dtypes
data['TOT_REG_CD'] = data['TOT_REG_CD'].astype('str')
data= data.fillna(0.0)

data['18년6월'].hist(bins=100,  figsize = (20,10), label='18년')
data['21년6월'].hist(bins=100,  figsize = (20,10), label='21년', alpha=0.5)
plt.legend()
plt.show()

print('18년6월 100명 미만 집계구 수 : ' + str(data[data['18년6월'] < 100].shape[0]))
print('21년6월 100명 미만 집계구 수 : ' + str(data[data['21년6월'] < 100].shape[0]))

data = data[data['18년6월'] >= 100]
data = data[data['21년6월'] >= 100]

g생활 = gpd.read_file('../data/results/g_생활인구.shp', encoding='cp949')
g생활 = g생활[g생활['TOT_REG_CD'].isin(data['TOT_REG_CD'])]

fig, ax = plt.subplots(1, 1, figsize=(20,15))
g생활.plot('인구차이', ax=ax, cmap='bwr', legend=True, vmin=-0.005, vmax=0.005)
plt.show()

#%%

data['인구차이'].hist(bins=1000, range=[-0.02,0.02], figsize = (20,10))
plt.show()

data['종속'] = data['인구차이'].apply(lambda x : -1 if x < -0.005 else (1 if x > 0.005 else 0))

g생활['종속'] = g생활['인구차이'].apply(lambda x : -1 if x < -0.005 else (1 if x > 0.005 else 0))


print('인구차이 작은 지역 : ' + str(data[data['종속'] == 0].shape[0]))

data = data[data['종속'] != 0]

g생활 = g생활[g생활['TOT_REG_CD'].isin(data['TOT_REG_CD'])]

fig, ax = plt.subplots(1, 1, figsize=(20,15))
g생활.plot('종속', ax=ax, cmap='bwr')
plt.show()
#%%

tmp = data.dtypes

tmp = data.mean()

tmp = data[['아파트', '단독주택', '다가구주택', '다세대주택','다중주택', '연립주택',
            '오피스텔', '사무소', '기타일반업무시설', 
            '상점', '가설점포', '소매점', '일반음식점', '기타제1종근린생활시설', '기타제2종근린생활시설', '기타판매시설',
            '학원', '고시원', '독서실', '대학교', '기타교육연구시설',
            '공',
            '종속']].copy()

tmp['주택'] = tmp['아파트'] + tmp['단독주택' ] + tmp['다가구주택' ] + tmp['다세대주택' ] + tmp['연립주택' ]

tmp['업무'] = tmp['오피스텔'] + tmp['사무소' ] + tmp['기타일반업무시설' ]

tmp['상가'] = tmp['상점'] + tmp['가설점포' ] + tmp['소매점' ] + tmp['일반음식점' ] + tmp['기타제1종근린생활시설' ] + tmp['기타제2종근린생활시설']

tmp['교육'] = tmp['학원'] + tmp['고시원' ] + tmp['독서실' ] + tmp['대학교' ] + tmp['기타교육연구시설']

tmp['공원'] = tmp['공']

tmp = tmp[['주택', '업무', '상가', '교육', '공원', '종속']]
# tmp = data[data.columns.drop(['TOT_REG_CD', 'Unnamed: 1', '18년6월','21년6월', '인구차이'])]
#%%

def SVM_margin(x, y, kernel, c, g, d) :
    classifier = SVC(kernel = kernel, C=c, gamma = g, degree = d)
    classifier.fit(x, y) 
    classifier.decision_function(x)
    classifier.predict(x)
    
    
    
    fig, ax = plt.subplots(figsize = (12,12))
    plt.scatter(x[np.where(y==-1),0][0], x[np.where(y==-1),1][0], marker = 'x', s=200, color = 'r', label = 'Negative -1')
    plt.scatter(x[np.where(y==1),0][0], x[np.where(y==1),1][0], marker = 'o', s=200, color = 'b',label = 'Positive +1')
    plt.xlim(0,6)
    plt.ylim(0,6)
    plt.xticks(np.arange(0, 6, step=1))
    plt.yticks(np.arange(0, 6, step=1))
    
    
    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = classifier.decision_function(xy).reshape(XX.shape)
    
    # plot decision boundary and margins
    ax.contour(XX, YY, Z, levels=[-1, 0, 1], alpha=0.5,
               colors=['#0000FF', '#000000', '#FF0000'], 
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    ax.scatter(classifier.support_vectors_[:, 0], classifier.support_vectors_[:, 1], s=400,
               linewidth=1, facecolors='none', edgecolors='k')
    plt.show()
    return classifier


#%%
tmp2 = tmp[['업무', '주택', '종속']]
x = np.array(tmp2[tmp2.columns.drop(['종속'])].fillna(0).astype('float').values)
y = np.array(tmp2['종속'].values)
y = y.reshape(y.shape[0], 1)


fig = plt.figure(figsize = (12,12))
plt.scatter(x[np.where(y==-1),0][0], x[np.where(y==-1),1][0], marker = 'x', s=200, color = 'r', label = 'Negative -1')
plt.scatter(x[np.where(y==1),0][0], x[np.where(y==1),1][0], marker = 'o', s=200, color = 'b',label = 'Positive +1')
plt.xlim(0,6)
plt.ylim(0,6)
plt.xticks(np.arange(0, 6, step=1))
plt.yticks(np.arange(0, 6, step=1))
plt.show()

classifier = SVM_margin(x, y, 'linear', c=1, g=1, d=0)
classifier = SVM_margin(x, y, 'poly', c=1, g=0.1, d=3)

classifier = SVM_margin(x, y, 'rbf', c=10, g=0.1, d=0)
classifier = SVM_margin(x, y, 'rbf', c=10, g=0.3, d=0)
classifier = SVM_margin(x, y, 'rbf', c=10, g=0.5, d=0)
classifier = SVM_margin(x, y, 'rbf', c=10, g=0.7, d=0)
classifier = SVM_margin(x, y, 'rbf', c=10, g=0.9, d=0)
classifier = SVM_margin(x, y, 'rbf', c=10, g=10.0, d=0)
y_pred = classifier.predict(x)

print(confusion_matrix(y,y_pred))
print('test_precision_score : ' + str(precision_score(y,y_pred))) 
print('test_recall_score : ' + str(recall_score(y,y_pred))) 
print('test_accuracy_score : ' + str(accuracy_score(y,y_pred))) 
print('test_f1_score : ' + str(f1_score(y,y_pred))) 

#%%
x = np.array(tmp[tmp.columns.drop(['종속'])].fillna(0).astype('float').values)
y = np.array(tmp['종속'].values)
y = y.reshape(y.shape[0], 1)

from sklearn.model_selection import train_test_split
"""데이터 나누기"""
#데이터를 training과 testing으로 나눈다
x_train, x_test, y_train, y_test = train_test_split(x,y,                #분할할 행렬
                                                    test_size=0.25,     #검증세트의 분할 비율
                                                    random_state=1,     #순서 섞을 경우 랜덤시드 고정용
                                                    shuffle=True,       #순서 섞을지
                                                    stratify=y)         #분할시 데이터 비율 유지


#%%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance

model_report = {}
model_report['linear'] = {}
# model_report['poly'] = {}
model_report['rbf'] = {}
# model_report['sigmoid'] = {}

C = [0.5, 1.0, 10.0]
degree = [2.0, 3.0]
gamma = [0.1, 0.5, 1.0]


for key in tqdm(model_report):
    model_report[key] = {}
    for c in C:
        if key != 'linear' :
            for g in gamma:
                if key == 'poly':
                    for d in degree:
                        attkey = 'Att_c'+str(c)+'_g'+str(g)+'_d'+str(d)
                        model_report[key][attkey] = {}
                        model_report[key][attkey]['model'] = SVC(kernel=key, C=c, gamma = g, degree = d)
                else :
                    attkey = 'Att_c'+str(c)+'_g'+str(g)
                    model_report[key][attkey] = {}
                    model_report[key][attkey]['model'] = SVC(kernel=key, C=c, gamma = g)
        else:
            attkey = 'Att_c'+str(c)
            model_report[key][attkey] = {}
            model_report[key][attkey]['model'] = SVC(kernel=key, C=c)
#%%
for key in tqdm(model_report):
    for attkey in tqdm(model_report[key]):
        start = time.time()
        
        model_report[key][attkey]['model'].fit(x_train, y_train.ravel())
        model_report[key][attkey]['learning time'] = time.time() - start
        
        y_pred = model_report[key][attkey]['model'].predict(x_train)
        
        model_report[key][attkey]['train_confusition_matrix'] = confusion_matrix(y_train,y_pred)
        model_report[key][attkey]['train_precision_score'] = precision_score(y_train,y_pred)
        model_report[key][attkey]['train_recall_score'] = recall_score(y_train,y_pred)
        model_report[key][attkey]['train_accuracy_score'] = accuracy_score(y_train,y_pred)
        model_report[key][attkey]['train_f1_score'] = f1_score(y_train,y_pred)
        
        y_pred = model_report[key][attkey]['model'].predict(x_test)
        
        model_report[key][attkey]['test_confusition_matrix'] = confusion_matrix(y_test,y_pred)
        model_report[key][attkey]['test_precision_score'] = precision_score(y_test,y_pred)
        model_report[key][attkey]['test_recall_score'] = recall_score(y_test,y_pred)
        model_report[key][attkey]['test_accuracy_score'] = accuracy_score(y_test,y_pred)
        model_report[key][attkey]['test_f1_score'] = f1_score(y_test,y_pred)

   
for key in tqdm(model_report):
    for attkey in tqdm(model_report[key]):
        print (key + ' | ' + attkey + 
               ' | train_ACC = ' + str(model_report[key][attkey]['train_accuracy_score']) + 
               ', test_ACC = ' + str(model_report[key][attkey]['test_accuracy_score']))

