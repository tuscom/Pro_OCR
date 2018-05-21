# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 01:28:45 2017

@author: FUNNICLOWN
"""

"J_modules"

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numba
import time #処理時間計測用

from text_painting import Text_painting

class calculate_J():
    def __init__(self, A, theta, alpha, **kwargs): #初期宣言でなんとかなるものを**kwargsに入れる        
        self.A = A # = (total_m × n0, total_m × n1)
        self.A0 = self.A[0]
        self.Yt = self.A[1]
        
        "訓練データをtrain, cv, testに分割する"

        self.m = float(np.shape(self.Yt)[0])
        self.m_set = [0.6*self.m, 0.2*self.m, 0.2*self.m]
                
        self.n = [np.shape(self.A0)[1], np.shape(self.Yt)[1]]
        self.theta = theta
        self.alpha = alpha
        
        self.kwargs = {"scaling_switch":1,
                       "lam":np.array([0, 0.01, 0.1]),
                       "random_select":10,
                       "update_ite":50,
                       "save_train_switch":1}
        self.kwargs.update(kwargs)
        
        self.scaling_switch = self.kwargs["scaling_switch"]
        self.lam = self.kwargs["lam"]
        self.random_select = self.kwargs["random_select"]
        self.update_ite = self.kwargs["update_ite"]
        self.save_train_switch = self.kwargs["save_train_switch"]
        

    def sigmoid(self, z):
        g = 1 / (1 + np.exp(-z))
        #    print("sigmoidの範囲 : "+str(np.min(g))+"～"+str(np.max(g)))

        return g

    def h(self, theta, X):
        "指標となる値に変換"
        "logisticとは少し違う"
        return np.inner(X, theta)

    def scaling(self, x):
        "操作可能パラメータ"
        d_switch = 0 #1 = 定義域使用。 0 = 標準偏差使用
        
        n = np.shape(x)[1]
        m = np.shape(x)[0]
        
        print("x_before = ...")
        print(x)
        
        if d_switch == 1:
            s = self.ran(x)
        else:
            s = self.std(x)
        """
        s = np.reshape(s, (1, s.shape[0]))
        """
        ave = x[:, 1:n+1].mean(axis = 1) # ave = (m,), x[:, 1:n+1] = (m, n[0])
        ave = np.reshape(ave, (m, 1))    
        print("average = ...")
        print(ave[:3, 0])
        print("...")
        
        x[:, 1:n+1] = (x[:, 1:n+1] - ave) / s
        
        print("x_after = ...")
        print(x)
        return x

    def ran(self, x):
        m = np.shape(x)[0]
        n = np.shape(x)[1]
        max = x[:, 1:n+1].max(axis = 1) # x[:, 1:n+1].max(axis=1) = (260, )
        min = x[:, 1:n+1].min(axis = 1)

        range = max - min
        range = np.reshape(range, (m, 1))
        print("Domain = " + str(range[:3, 0]))
        print("...")

        return range

    def std(self, x):
        m = np.shape(x)[0]
        n = np.shape(x)[1]
        
        st = x[:, 1:n+1].std(axis = 1)
        st = np.reshape(st, (m, 1))
        print("standard_deviation = " + str(st[:3, 0]))
        print("...")
        
        return st

    def devide_data(self, a0, random_switch=1):
        "dataをtrain, cv, testに分ける"
        
        m_set = [0.6*self.m, 0.2*self.m, 0.2*self.m]
        
        all_data = a0
        all_data = np.c_[all_data, a0]
        
        if random_switch:
            np.random.shuffle(all_data)
        
        "boundary 作成"
        n_boundary = np.zeros(np.shape(self.n)[0] - 1)
        n_boundary[0] = self.n[0]
        m_boundary = np.zeros(2)
        m_boundary[0] = m_set[0]
        m_boundary[1] = m_boundary[0] + m_set[1]        
        
        for i in range(np.shape(n_boundary)[0]-1):
            n_boundary[i+1] = n_boundary[i] + self.n[i+1]
            
        all_data2 = [0, 0, 0]
        all_data = np.vsplit(all_data, m_boundary)
        
        for i in range(len(all_data2)):
            all_data2[i] = np.hsplit(all_data[i], n_boundary)
            
        return all_data2
        
    def del_J(self, a0_one, yp, yt, theta, lam):
        m = np.shape(yp)[0]
        n = [np.shape(theta)[0], np.shape(theta)[1]]
        del_j = np.zeros((n[1], n[0])) #θと同じ(n[1] × n[0])になるはず。
        
        del_j = (1./m) * (yp - yt)
        del_j = np.inner(del_j.T, a0_one.T)
    
        theta_mask = theta
        theta_mask[:, 0] = 0
        del_j += (lam / m) * theta_mask
        
        return del_j

#    @numba.jit
    def J_one(self, theta, yp_one, yt_one,lam, j_save, r, i, j, k, l):
        m = np.shape(yp_one)[0]
        
        j_one = (1. / m) * np.sum(-yt_one * np.log(yp_one) - (1 - yt_one) * np.log(1 - yp_one), axis=0)
        j_one += (lam / m) * np.sum(theta[1:] ** 2)
        j_save[r][i][j][k][l] = j_one
        
        return j_save

#    @numba.jit
    def J_update_ite(self, theta, a0_train, yt_train, a0_one, yt_one, lam, j_save, r, i, j, k):
        
        for l in range(self.update_ite):
            yp = self.sigmoid(self.h(theta, a0_train)) #yp計算
            del_j = self.del_J(a0_train, yp, yt_train, theta, lam)
            theta -= self.alpha * del_j
            "更新のためのtrainでのj、値記録のためのtrain, test, cvでのjが必要"
            yp_one = self.sigmoid(self.h(theta, a0_one))
            
#            j_save[r][i][j][k].append([])
            j_save = self.J_one(theta, yp_one, yt_one, lam, j_save, r, i, j, k, l)
            
        print("done update_ite")            
            
        return j_save
    
#    @numba.jit
    def J_m_train(self, lam, a0_3, a_train, a0_one, yt_one, j_save, r, i, j):
        for k in range(int(self.m_set[0])):
            if self.save_train_switch:
                "特に制約なし"
            else:
                k = self.m_set[0]
            
#            j_save[r][i][j].append([])
            a0_train = a0_3[0][:k+1, :]
            yt_train = a_train[1][:k+1, :]
            j_save = self.J_update_ite(self.theta, a0_train, yt_train, a0_one, yt_one, lam, j_save, r, i, j, k)

        print("done m_train")            
            
        return j_save
        
#    @numba.jit
    def J_lambda(self, a0_3, a_train, a0_one, yt_one, j_save, r, i):
        for j in range (len(self.lam)):
#            j_save[r][i].append([])
            j_save = self.J_m_train(self.lam[j], a0_3, a_train, a0_one, yt_one, j_save, r, i, j)
            
        print("done lambda")            
            
        return j_save
        
#    @numba.jit
    def J_train_cv_test(self, a0_3, a_train, yt_3, j_save, r):
        for i in range(3):
            j_save = self.J_lambda(a0_3, a_train, a0_3[i], yt_3[i], j_save, r, i)

        print("done train_cv_test")            
        
        return j_save
        
#    @numba.jit
    def J_random_select(self):
        j_save = []
        #先に型を作ってみる
        j_save = np.zeros((self.random_select, 3, len(self.lam), self.m_set[0], self.update_ite, self.n[1]))
        
        if self.scaling_switch:
            self.A[0] = self.scaling(self.A[0])            
            
        all_data = np.c_[self.A[0], self.A[1]]
        for r in range(self.random_select):
            np.random.shuffle(all_data)
            all_data2 = np.hsplit(all_data, [self.n[0]])
            a0_3 = np.vsplit(all_data2[0], [self.m_set[0], self.m_set[0]+self.m_set[1]])
            yt_3 = np.vsplit(all_data2[1], [self.m_set[0], self.m_set[0]+self.m_set[1]])
            
            a_train = [a0_3[0], yt_3[0]]
            
            j_save = self.J_train_cv_test(a0_3, a_train, yt_3, j_save, r)
            
        print("done random_select")
            
        return j_save

    def j_one(self, theta, yp_one, yt_one,lam):
        m = np.shape(yp_one)[0]
        
        j_one = (1. / m) * np.sum(-yt_one * np.log(yp_one) - (1 - yt_one) * np.log(1 - yp_one), axis=0)
        j_one += (lam / m) * np.sum(theta[1:] ** 2)
        
        j_one = np.average(j_one)
        
        return j_one


    def j_ite(self, lam):
        start = time.time()
        
        "j_iteのグラフ描画専門"

        x_ite = np.arange(1, self.update_ite+1, 1)

        self.A[0] = self.scaling(self.A[0])        
        
        j_save = np.zeros((self.random_select, 3, self.update_ite))
        
        all_data = np.c_[self.A[0], self.A[1]]
        for r in range(self.random_select):
            print("ite updating...")
            np.random.shuffle(all_data)
            all_data2 = np.hsplit(all_data, [self.n[0]])
            a0_3 = np.vsplit(all_data2[0], [self.m_set[0], self.m_set[0]+self.m_set[1]])
            yt_3 = np.vsplit(all_data2[1], [self.m_set[0], self.m_set[0]+self.m_set[1]])
            
            a_train = [a0_3[0], yt_3[0]]
            
            for i in range(3):
                a0_train = a_train[0]
                yt_train = a_train[1]
                a0_one = a0_3[i]
                yt_one = yt_3[i]
                theta = self.theta
                for l in range(self.update_ite):
                    yp = self.sigmoid(self.h(theta, a0_train)) #yp計算
                    del_j = self.del_J(a0_train, yp, yt_train, theta, lam)
                    theta -= self.alpha * del_j
                    "更新のためのtrainでのj、値記録のためのtrain, test, cvでのjが必要"
                    yp_one = self.sigmoid(self.h(theta, a0_one))
            
                    j_save[r, i, l] = self.j_one(theta, yp_one, yt_one, lam)
                    

        j_save = np.average(j_save, axis=0)
        print(np.shape(j_save))

        "グラフ描画"
        fig = plt.figure()
        plt.title("j_iteration")
        plt.xlabel("update iteration")
        plt.ylabel("j")
        plt.plot(x_ite, j_save[0], label="train")
        plt.plot(x_ite, j_save[1], label="cv")
        plt.plot(x_ite, j_save[2], label="test")
        plt.legend()
        plt.savefig("j_ite.png")

        elapsed_time = time.time() - start
        print("j_ite's time : " + str(elapsed_time))

    def j_m_train(self, lam):
        start = time.time()
        "j_iteのグラフ描画専門"

        self.A[0] = self.scaling(self.A[0])        
        
        j_save = np.zeros((self.random_select, 3, self.m_set[0], self.update_ite))
        
        all_data = np.c_[self.A[0], self.A[1]]
        for r in range(self.random_select):
            np.random.shuffle(all_data)
            all_data2 = np.hsplit(all_data, [self.n[0]])
            a0_3 = np.vsplit(all_data2[0], [self.m_set[0], self.m_set[0]+self.m_set[1]])
            yt_3 = np.vsplit(all_data2[1], [self.m_set[0], self.m_set[0]+self.m_set[1]])
            
            a_train = [a0_3[0], yt_3[0]]
            
            for i in range(3):
                a0_train = a_train[0]
                yt_train = a_train[1]
                a0_one = a0_3[i]
                yt_one = yt_3[i]
                theta = self.theta
                
                for k in range(int(self.m_set[0])):
            
                    a0_train = a0_3[0][:k+1, :]
                    yt_train = a_train[1][:k+1, :]

                    for l in range(self.update_ite):
                        yp = self.sigmoid(self.h(theta, a0_train)) #yp計算
                        del_j = self.del_J(a0_train, yp, yt_train, theta, lam)
                        theta -= self.alpha * del_j
                        "更新のためのtrainでのj、値記録のためのtrain, test, cvでのjが必要"
                        yp_one = self.sigmoid(self.h(theta, a0_one))
            
                        j_save[r, i, k, l] = self.j_one(theta, yp_one, yt_one, lam)

        j_save = np.average(j_save, axis=0)
        print(np.shape(j_save))

        elapsed_time = time.time() - start
        print("j_m_train's time : " + str(elapsed_time))



    def j_lam(self):
        start = time.time()
        
        self.A[0] = self.scaling(self.A[0])        
        
        j_save = np.zeros((self.random_select, 3, len(self.lam)))
        
        all_data = np.c_[self.A[0], self.A[1]]
        for r in range(self.random_select):
            np.random.shuffle(all_data)
            all_data2 = np.hsplit(all_data, [self.n[0]])
            a0_3 = np.vsplit(all_data2[0], [self.m_set[0], self.m_set[0]+self.m_set[1]])
            yt_3 = np.vsplit(all_data2[1], [self.m_set[0], self.m_set[0]+self.m_set[1]])
            
            a_train = [a0_3[0], yt_3[0]]
            
            for i in range(3):
                a0_train = a_train[0]
                yt_train = a_train[1]
                a0_one = a0_3[i]
                yt_one = yt_3[i]
                theta = self.theta
                
                for j in range (len(self.lam)):
                    lam = self.lam[j]

                    for l in range(self.update_ite):
                        yp = self.sigmoid(self.h(theta, a0_train)) #yp計算
                        del_j = self.del_J(a0_train, yp, yt_train, theta, lam)
                        theta -= self.alpha * del_j
                        "更新のためのtrainでのj、値記録のためのtrain, test, cvでのjが必要"
                        yp_one = self.sigmoid(self.h(theta, a0_one))
            
                    j_save[r, i, j] = self.j_one(theta, yp_one, yt_one, lam)

        j_save = np.average(j_save, axis=0)
        print(np.shape(j_save))
        print(np.shape(lam))

        fig = plt.figure()
        plt.title("j_lambda")
        plt.xlabel("lambda")
        plt.ylabel("j")
        plt.plot(self.lam, j_save[0], label="train")
        plt.plot(self.lam, j_save[1], label="cv")
        plt.plot(self.lam, j_save[2], label="test")
        plt.legend()
        plt.savefig("j_lambda.png")

        elapsed_time = time.time() - start
        print("j_lam's time : " + str(elapsed_time))

    


def load_x(file_name):
    "データ作成"
    x = np.load(file_name)
    return x

def convert_x(file_name):
    "x→a変換"
    xt = load_x(file_name)
    
    "データについて"
    x_shape = np.shape(xt)
    print("フォント数 ： " + str(x_shape[0]))
    print("文字数 : " + str(x_shape[1]))
    print("画像枚数 : " + str(x_shape[0] * x_shape[1]))
    print("縦 pixel数 : " + str(x_shape[2]))
    print("横 pixel数 : " + str(x_shape[3]))
    print("RGB : " + str(x_shape[4]))
    print("np.shape(xt) : " + str(x_shape))

    "変数名をlayer用に変換"    
    m = x_shape[0] * x_shape[1]
    n = [0, 0]
    n[0] = x_shape[2] * x_shape[3] + 1
    n[1] = x_shape[1]

    xtR = xt[:, :, :, :, 0]
    xtR = np.reshape(xtR, (x_shape[0], x_shape[1], x_shape[2] * x_shape[3]))
    xtR = np.reshape(xtR, (x_shape[0] * x_shape[1], np.shape(xtR)[2]))
    print("データX : (総画像枚数, 総pixel数） = " + str(np.shape(xtR)))
    
    a0 = np.c_[np.ones((m, 1)), xtR]
    
    """training_datay変換"""
    yt = np.zeros((m, x_shape[1]))
    for i in range(m):
        j = np.mod(i, x_shape[1])
        yt[i, j] = 1

        
    return [a0, yt]

def initial_theta(L_in, L_out):
    "絶対値がεinitより小さくなるはず"
    sigma_init = np.sqrt(6) / (np.sqrt(L_in) + np.sqrt(L_out))
    print("εinit = " + str(sigma_init))
    
    initial_theta = np.random.rand(L_out, L_in) * 2 * sigma_init - sigma_init
    return initial_theta

def main():
    start = time.time()
    file_name = "alphabet.npy"

    [a0, yt] = convert_x(file_name)
    
#    "一回限り"
#    #m * n0の形で保存
#    np.savez("alphabet.npz", a0=a0, yt=yt)
    
    L_in = np.shape(a0)[1]
    L_out = np.shape(yt)[1]
    
    theta = initial_theta(L_in, L_out) #総更新回数とともに保存し、累積的に更新できるようにする
    
    best_lam = 0
    
    update_ite = 50
    lam = [0, 0.01, 0.1, 1, 3]
    calc_J = calculate_J([a0, yt], theta, alpha=0.001, update_ite=update_ite, lam=lam)
    
#    calc_J.j_ite(best_lam)
    calc_J.j_lam()
#    calc_J.j_m_train(best_lam)
    
#    j_save = calc_J.J_random_select()
#    print(np.shape(j_save))
#
#    "j_変形"
#    j_custom = np.average(j_save, axis=0)
#    j_custom = np.average(j_custom, axis=4)
#    print(np.shape(j_custom))
#
#    "J_ite描画"
#    x_iteration = np.arange(1, update_ite+1, 1)
#    
#    best_lam_index = lam.index(best_lam)
#    
##    j_ite_custom = j_custom[:][:][-1][:]
##    print(np.shape(j_ite_custom))
#    j_ite_train = j_custom[0][best_lam_index][-1][:]
#    j_ite_cv = j_custom[1][best_lam_index][-1][:]
#    j_ite_test = j_custom[2][best_lam_index][-1][:]
#    print(np.shape(x_iteration), np.shape(j_ite_train))
#    
#    plt.plot(x_iteration, j_ite_train)    
#    plt.plot(x_iteration, j_ite_cv)
#    plt.plot(x_iteration, j_ite_test)
#    
#    plt.title("J_iteration")
#    plt.ylabel("J")
#    plt.xlabel("iteration")
#    
#    plt.show()
#    
#    "J_m描画"
#
#    x_m_train = np.arange(1, np.shape(j_custom)[2]+1, 1)    
#    
#    j_m_custom = np.array(j_custom)
#    j_m_train = j_custom[0, best_lam_index, :, -1]
#    j_m_cv = j_custom[1, best_lam_index, :, -1]
#    j_m_test = j_custom[2, best_lam_index, :, -1]
#    print("m_train")
#    print(np.shape(j_m_train), np.shape(x_m_train))
#    
#    plt.plot(x_m_train, j_m_train)
#    plt.plot(x_m_train, j_m_cv)
#    plt.plot(x_m_train, j_m_test)
#    
#    plt.title("J_m_train")
#    plt.ylabel("J")
#    plt.xlabel("m_train")    
#    
#    plt.show()
#    
#    
#    "J_lam描画"
#    x_lam = lam
#    
#    j_lam_custom = np.array(j_custom)
#    j_lam_train = j_lam_custom[0, :, -1, -1]
#    j_lam_cv = j_lam_custom[1, :, -1, -1]
#    j_lam_test = j_lam_custom[2, :, -1, -1]
#    print(np.shape(j_lam_train))
#    
#    plt.plot(x_lam, j_lam_train)
#    plt.plot(x_lam, j_lam_cv)
#    plt.plot(x_lam, j_lam_test)
#    
#    plt.title("J_lambda")
#    plt.ylabel("J")
#    plt.xlabel("lambda")
#    
#    plt.show()
#
#
    elapsed_time = time.time() - start
    print("elapsed_time : " + str(elapsed_time))
    raw_input("end program")
    
#main()