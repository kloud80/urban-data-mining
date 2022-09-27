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

mpl.rc('font', family='gulim') #한글 폰트 적용시
os.chdir('04 SVM/')


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

#%%
""" 2차계획법 라이브러리 cvxopt 활용  https://cvxopt.org/ """
#Importing with custom names to avoid issues with numpy / sympy matrix
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

#Initializing values and computing H. Note the 1. to force to float type
m,n = x.shape
y = y.reshape(-1,1) * 1.
x_dash = y * x
H = np.dot(x_dash , x_dash.T) * 1.

#Converting into cvxopt format
P = cvxopt_matrix(H)
q = cvxopt_matrix(-np.ones((m, 1)))
G = cvxopt_matrix(-np.eye(m))
h = cvxopt_matrix(np.zeros(m))
A = cvxopt_matrix(y.reshape(1, -1))
b = cvxopt_matrix(np.zeros(1))

#Setting solver parameters (change default to decrease tolerance) 
cvxopt_solvers.options['show_progress'] = True
cvxopt_solvers.options['abstol'] = 1e-10
cvxopt_solvers.options['reltol'] = 1e-10
cvxopt_solvers.options['feastol'] = 1e-10

#Run solver
sol = cvxopt_solvers.qp(P, q, G, h, A, b)
alphas = np.array(sol['x'])
disp = alphas.reshape([1, 6])
# disp = w.reshape([1, 2])
# disp = b.reshape([1, 6])

#%%
""" 구한 알파를 통해 w와 b를 계산한다 """
maximum = alphas.reshape([1,6])


w = np.zeros(x.shape[1])
for ai, xi, yi in zip(maximum[0,:], x, y):
    w += ai * yi * xi
    
bs = np.empty((0), float)
support_vectors_ = np.empty((0,2), float)
for ai, xi, yi in zip(maximum[0,:], x, y):
    if ai > 0.000000001 : #threshold
        bs = np.append(bs, yi - np.dot(w.T, xi))
        support_vectors_= np.append(support_vectors_, np.array([xi]), axis=0)

b = bs.sum() / len(bs)


# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T

Z = ((w *  xy).sum(axis=1) + b).reshape(XX.shape)


fig, ax = plt.subplots(figsize = (12,12))
plt.scatter(x[np.where(y==-1),0][0], x[np.where(y==-1),1][0], marker = 'x', s=200, color = 'r', label = 'Negative -1')
plt.scatter(x[np.where(y==1),0][0], x[np.where(y==1),1][0], marker = 'o', s=200, color = 'b',label = 'Positive +1')
plt.xlim(0,6)
plt.ylim(0,6)
plt.xticks(np.arange(0, 6, step=1))
plt.yticks(np.arange(0, 6, step=1))

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])

# plot support vectors
ax.scatter(support_vectors_[:, 0], support_vectors_[:, 1], s=500,
           linewidth=1, facecolors='none', edgecolors='k')
plt.show()


#%%

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', C=1e10)
classifier.fit(x, y) 

classifier.coef_
w
classifier.intercept_
b


alphas2 = np.zeros(y.shape, float)
alphas2[classifier.support_] = classifier.dual_coef_.reshape(alphas2[classifier.support_].shape)
alphas2 = alphas2 * y

alphas2
alphas




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


fig, ax = plt.subplots(figsize = (12,12))
plt.scatter(x[np.where(y==-1),0][0], x[np.where(y==-1),1][0], marker = 'x', s=200, color = 'r', label = 'Negative -1')
plt.scatter(x[np.where(y==1),0][0], x[np.where(y==1),1][0], marker = 'o', s=200, color = 'b',label = 'Positive +1')
plt.xlim(0,6)
plt.ylim(0,6)
plt.xticks(np.arange(0, 6, step=1))
plt.yticks(np.arange(0, 6, step=1))

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])

# plot support vectors
ax.scatter(classifier.support_vectors_[:, 0], classifier.support_vectors_[:, 1], s=400,
           linewidth=1, facecolors='none', edgecolors='k')
plt.show()
