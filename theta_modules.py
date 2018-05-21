# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 02:28:01 2017

@author: FUNNICLOWN
"""

"theta_modules"

import numpy as np
import time
import matplotlib.pyplot as plt

from accuracy import scaling

def h(theta, X):
    "指標となる値に変換"
    "logisticとは少し違う"
    return np.inner(X, theta)

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

def del_J(a0_one, yp, yt, theta, lam):
    m = np.shape(yp)[0]
    n = [np.shape(theta)[0], np.shape(theta)[1]]
    del_j = np.zeros((n[1], n[0])) #θと同じ(n[1] × n[0])になるはず    
    del_j = (1./m) * (yp - yt)
    del_j = np.inner(del_j.T, a0_one.T)
    
    theta_mask = theta
    theta_mask[:, 0] = 0
    del_j += (lam / m) * theta_mask
        
    return del_j

def initial_theta(L_in, L_out):
    "絶対値がεinitより小さくなるはず"
    sigma_init = np.sqrt(6) / (np.sqrt(L_in) + np.sqrt(L_out))
    print("εinit = " + str(sigma_init))
    
    initial_theta = np.random.rand(L_out, L_in) * 2 * sigma_init - sigma_init
    
    "初めだけ保存"
    file_name = "theta.npz"
    np.savez(file_name, theta=initial_theta)
    
def a0_to_yp(a0, theta):
    m = np.shape(a0)[0]
    g = sigmoid(h(theta, a0))
    
    p_max = g.max(axis=1)
    p_max = np.reshape(p_max, (m, 1))
    "最大値以外０"
    g_copy = g
    g_copy[g < p_max] = 0
    g_copy[g >= p_max] = 1

    yp = g_copy
    
    return yp

def update_theta(theta, a0_train, yt_train, lam, alpha, update_ite, theta_file_name):
    start = time.time()
    a0_train = scaling(a0_train)
    
    
    """θ更新"""
    for l in range(update_ite):
        yp = sigmoid(h(theta, a0_train)) #yp計算
        del_j = del_J(a0_train, yp, yt_train, theta, lam)
        theta -= alpha * del_j
    
    """精度計算"""
    yp = a0_to_yp(a0_train, theta)
    
    m = np.shape(a0_train)[0]
    bit_hit = yp==yt_train
    "各アルファベット評価"
    alpha_hit = np.all(bit_hit, axis=1)
    "正当要素抽出"
    hit = float(alpha_hit[alpha_hit == True].size)
#    print("正答数 : " + str(int(hit)) + "個" + " / " + str(m) + "個")
    m = float(m)
    ratio = 100. * (hit / m)
    print("精度 : " + str(ratio) + "%")
    
    """精度記録""" #どんな時も初期値は必要
#    file_name = "accuracy.npz"
#    accuracy_file = np.load(file_name)
#    update_store = accuracy_file["update_store"]
#    accuracy_store = accuracy_file["accuracy_store"]    
#    
#    print(np.append(update_store, update_store[-1]+update_ite))
#    update_store = np.append(update_store, update_store[-1]+update_ite)
#    accuracy_store = np.append(accuracy_store, ratio)
#
#    np.savez(file_name, update_store=update_store, accuracy_store=accuracy_store)
#    np.savez(theta_file_name, theta=theta)
    
    "グラフ保存"
#    fig = plt.figure()
#    plt.title("accuracy change")
#    plt.xlabel("update iteration")
#    plt.ylabel("accuracy")
#    plt.plot(update_store, accuracy_store)
#    plt.savefig("accuracy_change.png")
    
    elapsed_time = time.time() - start
    print("update_theta's time : " + str(elapsed_time))
    return theta
