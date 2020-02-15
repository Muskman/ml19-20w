#!/usr/bin/env python

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import numpy as np
from random import randint

df = np.loadtxt("train")

Y = df[:,0]

X = df[:,1:]
np.shape(X)

w = np.zeros((1000,))
alpha = 0.05
w1 = np.zeros((1000,))

for i in range(100):
    for j in range(1000):
        if w[j] > 0:
            w1[j] = 1
        elif w[j] == 0: 
            w1[j] = 0
        else :
            w1[j] = -1
        
    tr = np.subtract(Y,np.matmul(X,w))
    
    tr1 = (np.matmul(np.transpose(X),tr))
    tr1 *= 2
    tr1 = w1 - tr1
    tr1 *= alpha
    w = np.subtract(w, tr1)

b = np.argsort(w)
w

c = b[0:979]
c

for i in range(979):#equating every element of w[i] to zero except maximum 20 elements 
    j = c[i]
    w[j] = 0

w


Wnew = np.loadtxt("wAstTrain")

sum = 0
count = 0

for i in range(1000):
    if(Wnew[i]!=0 and w[i] != 0):
        sum += abs(Wnew[i] - w[i])
        count += 1

sum

count