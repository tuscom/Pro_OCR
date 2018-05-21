# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 01:51:36 2017

@author: FUNNICLOWN
"""

import numpy as np

def predict(theta, pixel_data, label_text):
    """pixel_data→yp & 精度計算"""
#    pixel_data=["width", "height"], label_text="nothing"
    "数値変換"
    a0 = pixel_data.flatten()
#    a0 = np.reshape(a0, (1, np.shape(a0)[0]))
    a0 = np.r_[1, a0] #=(1+総ピクセル)
    a0_scaling = np.reshape(a0, (1, np.shape(a0)[0]))
    a0_scaling = scaling(a0_scaling)
    a0 = np.reshape(a0_scaling, np.shape(a0)[0])

    g = sigmoid(h(theta, a0))
    
    g_max_index = np.argmax(g)

    predict_label = label_text[g_max_index]
    
    return predict_label
    
def sigmoid(z):

    g = 1 / (1 + np.exp(-z))
    return g

def h(theta, X):
    "指標となる値に変換"
    "logisticとは少し違う"

    return np.inner(X, theta)

def scaling(x):
    "操作可能パラメータ"
    d_switch = 0 #1 = 定義域使用。 0 = 標準偏差使用    
    n = np.shape(x)[1]
    m = np.shape(x)[0]
                
    if d_switch == 1:
        s = ran(x)
    else:
        s = std(x)
    """
    s = np.reshape(s, (1, s.shape[0]))
    """
    ave = x[:, 1:n+1].mean(axis = 1) # ave = (m,), x[:, 1:n+1] = (m, n[0])
    ave = np.reshape(ave, (m, 1))    
    x[:, 1:n+1] = (x[:, 1:n+1] - ave) / s
    
    return x

def ran(x):
    m = np.shape(x)[0]
    n = np.shape(x)[1]
    max = x[:, 1:n+1].max(axis = 1) # x[:, 1:n+1].max(axis=1) = (260, )
    min = x[:, 1:n+1].min(axis = 1)
    range = max - min
    range = np.reshape(range, (m, 1))

    return range

def std(x):
    m = np.shape(x)[0]
    n = np.shape(x)[1]
    
    st = x[:, 1:n+1].std(axis = 1)
    st = np.reshape(st, (m, 1))
    
    return st
