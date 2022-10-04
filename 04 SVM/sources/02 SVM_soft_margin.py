# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 10:19:51 2021

@author: KLOUD
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

mpl.rc('font', family='gulim') #한글 폰트 적용시
os.chdir('04 SVM/')

#%%

def SVM_margin(x, y, c, kernel='linear') :
    classifier = SVC(kernel = kernel, C=c,)
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
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    ax.scatter(classifier.support_vectors_[:, 0], classifier.support_vectors_[:, 1], s=400,
               linewidth=1, facecolors='none', edgecolors='k')
    plt.show()
    return classifier

#%%
x = np.array([[1, 2], [1, 5], [4, 1], [3, 5], [5, 5], [5, 2]])
y = np.array([1, 1, 1, -1, -1, -1])

fig = plt.figure(figsize = (12,12))
plt.scatter(x[np.where(y==-1),0][0], x[np.where(y==-1),1][0], marker = 'x', s=200, color = 'r', label = 'Negative -1')
plt.scatter(x[np.where(y==1),0][0], x[np.where(y==1),1][0], marker = 'o', s=200, color = 'b',label = 'Positive +1')
plt.xlim(0,6)
plt.ylim(0,6)
plt.xticks(np.arange(0, 6, step=1))
plt.yticks(np.arange(0, 6, step=1))
plt.show()

classifier = SVM_margin(x, y, c=1e10)

w = classifier.coef_
b = classifier.intercept_
alphas = np.zeros(y.shape, float)
alphas[classifier.support_] = classifier.dual_coef_
alphas = alphas * y


#%%
x = np.array([[1, 2], [1, 5], [4, 1], [3, 5], [5, 5], [5, 2], [3,4]])
y = np.array([1, 1, 1, -1, -1, -1, -1])


fig = plt.figure(figsize = (12,12))
plt.scatter(x[np.where(y==-1),0][0], x[np.where(y==-1),1][0], marker = 'x', s=200, color = 'r', label = 'Negative -1')
plt.scatter(x[np.where(y==1),0][0], x[np.where(y==1),1][0], marker = 'o', s=200, color = 'b',label = 'Positive +1')
plt.xlim(0,6)
plt.ylim(0,6)
plt.xticks(np.arange(0, 6, step=1))
plt.yticks(np.arange(0, 6, step=1))
plt.show()


classifier = SVM_margin(x, y, c=1e10)
classifier = SVM_margin(x, y, c=10)
classifier = SVM_margin(x, y, c=1)
classifier = SVM_margin(x, y, c=0.000001)
classifier.coef_
classifier.support_vectors_


#%%
x = np.array([[1, 2], [1, 5], [4, 1], [3, 5], [5, 5], [5, 2], [3,3]])
y = np.array([1, 1, 1, -1, -1, -1, -1])

fig = plt.figure(figsize = (12,12))
plt.scatter(x[np.where(y==-1),0][0], x[np.where(y==-1),1][0], marker = 'x', s=200, color = 'r', label = 'Negative -1')
plt.scatter(x[np.where(y==1),0][0], x[np.where(y==1),1][0], marker = 'o', s=200, color = 'b',label = 'Positive +1')
plt.xlim(0,6)
plt.ylim(0,6)
plt.xticks(np.arange(0, 6, step=1))
plt.yticks(np.arange(0, 6, step=1))
plt.show()

classifier = SVM_margin(x, y, c=2)
classifier = SVM_margin(x, y, c=10)
classifier = SVM_margin(x, y, c=100)

#%%
x = np.array([[1, 2], [1, 5], [4, 1], [3, 5], [5, 5], [5, 2], [2,3]])
y = np.array([1, 1, 1, -1, -1, -1, -1])

fig = plt.figure(figsize = (12,12))
plt.scatter(x[np.where(y==-1),0][0], x[np.where(y==-1),1][0], marker = 'x', s=200, color = 'r', label = 'Negative -1')
plt.scatter(x[np.where(y==1),0][0], x[np.where(y==1),1][0], marker = 'o', s=200, color = 'b',label = 'Positive +1')
plt.xlim(0,6)
plt.ylim(0,6)
plt.xticks(np.arange(0, 6, step=1))
plt.yticks(np.arange(0, 6, step=1))
plt.show()

classifier = SVM_margin(x, y, c=1, kernel='linear')
classifier = SVM_margin(x, y, c=100, kernel='rbf')
classifier = SVM_margin(x, y, c=0.1, kernel='poly')
classifier = SVM_margin(x, y, c=100, kernel='sigmoid')

classifier = SVM_margin(x, y, c=100)


