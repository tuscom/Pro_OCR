# coding: cp932
"""
Created on Tue Nov 07 11:59:27 2017

@author: 7317058
"""

"neural_forward"

from PIL import Image
import pygame
import numpy as np
import matplotlib.pyplot as plt

def train_X(file_name):
    "�f�[�^�쐬"
    x = np.load(file_name)
    return x
    
def initial_theta(L_in, L_out):
    "��Βl����init��菬�����Ȃ�͂�"
    sigma_init = np.sqrt(6) / (np.sqrt(L_in) + np.sqrt(L_out))
    print("��init = " + str(sigma_init))
    
    initial_theta = np.random.rand(L_out, L_in) * 2 * sigma_init - sigma_init
    return initial_theta

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
#    print("sigmoid�͈̔� : "+str(np.min(g))+"�`"+str(np.max(g)))

    return g

def h(theta, X):
    "�w�W�ƂȂ�l�ɕϊ�"
    "logistic�Ƃ͏����Ⴄ"
    return np.inner(X, theta)

def J(g, y, theta, lam=0.0):
    "cost�v�Z"
    m = float(np.shape(y)[0])
    
    theta_mask = theta[0]
    theta_mask[:, 0] = 0
    j = (1./m) * np.sum(-y * np.log(g) - (1 - y) * np.log(1 - g), axis=0)
    j += (lam / (2*m)) * np.sum(theta_mask**2, axis=1)
    
    return j

def del_J(a, y, theta, lam=0.0):
    m = np.shape(a[0])[0]
    n = [np.shape(a[0])[1], np.shape(a[1])[1]]
    del_j = np.zeros((n[1], n[0])) #�ƂƓ���(n[1] �~ n[0])�ɂȂ�͂��B

    del_j = (1./m) * (a[1] - y)
    del_j = np.inner(del_j.T, a[0].T)
    
    theta_mask = theta[0]
    theta_mask[:, 0] = 0
    del_j += (lam / m) * theta_mask
#    print(np.shape(del_j))
#    print(np.shape(theta[0]))
    
    return del_j

def scaling(x):
    "����\�p�����[�^"
    d_switch = 0 #1 = ��`��g�p�B 0 = �W���΍��g�p
    
    n = np.shape(x)[1]
    m = np.shape(x)[0]
    
    print("x_before = ...")
    print(x)
    
    if d_switch == 1:
        s = ran(x)
    else:
        s = std(x)
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

def ran(x):
    m = np.shape(x)[0]
    n = np.shape(x)[1]
    max = x[:, 1:n+1].max(axis = 1) # x[:, 1:n+1].max(axis=1) = (260, )
    min = x[:, 1:n+1].min(axis = 1)

    range = max - min
    range = np.reshape(range, (m, 1))
    print("Domain = " + str(range[:3, 0]))
    print("...")

    return range

def std(x):
    m = np.shape(x)[0]
    n = np.shape(x)[1]

    st = x[:, 1:n+1].std(axis = 1)
    st = np.reshape(st, (m, 1))
    print("standard_deviation = " + str(st[:3, 0]))
    print("...")
    
    return st

def Neural_Forward():
    "One_vs_all(no hidden layer)"
    
    "����\�p�����[�^"
    iteration = 100 #�X�V�񐔐ݒ�
    thre = 0.5 #threshold�ݒ�
    alpha = 0.001 #�w�K���ݒ�
    scaling_switch = 1 #scaling�����邩�ǂ����̐ݒ�
    ncol = 5 #�v���b�g�O���t�̗񐔎w��
    
    """�f�[�^X�擾"""
    file_name = "alphabet.npy"
    
    xt = train_X(file_name)
    "�f�[�^�ɂ���"
    x_shape = np.shape(xt)
    print("�t�H���g�� �F " + str(x_shape[0]))
    print("������ : " + str(x_shape[1]))
    print("�摜���� : " + str(x_shape[0] * x_shape[1]))
    print("�c pixel�� : " + str(x_shape[2]))
    print("�� pixel�� : " + str(x_shape[3]))
    print("RGB : " + str(x_shape[4]))
    print("np.shape(xt) : " + str(x_shape))
    
    """�摜�\�����Ă݂�"""
    test_img = Image.new("RGB", (x_shape[3], x_shape[2]))
    for i in range(x_shape[3]):
        for j in range(x_shape[2]):
            test_img.putpixel((i, j), tuple(xt[0, 0, j, i]))
#    test_img.show()
    
    "�ϐ�����layer�p�ɕϊ�"    
    m = x_shape[0] * x_shape[1]
    n = [0, 0]
    n[0] = x_shape[2] * x_shape[3] + 1
    n[1] = x_shape[1]
    
    """training_dataX�ϊ�"""
    "RGB����"
    xtR = xt[:, :, :, :, 0]
    print("�f�[�^X : (�t�H���g��, �A���t�@�x�b�g��, �cpixel��, ��pixel��) = " + str(np.shape(xtR)))
    xtG = xt[:, :, :, :, 1]
    xtB = xt[:, :, :, :, 2]
    
    "�s�N�Z�����P��ɂ���"
    xtR = np.reshape(xtR, (x_shape[0], x_shape[1], x_shape[2] * x_shape[3]))
    print("�f�[�^X : (�t�H���g��, �A���t�@�x�b�g��, ��pixel��) = " + str(np.shape(xtR)))
    "�摜�����̂P��ɂ���"
    xtR = np.reshape(xtR, (x_shape[0] * x_shape[1], np.shape(xtR)[2]))
    print("�f�[�^X : (���摜����, ��pixel���j = " + str(np.shape(xtR)))
    "2������reshape�͍s�A��̏��Ɍ���"    
    
    """training_datay�ϊ�"""
    yt = np.zeros((m, x_shape[1]))
    for i in range(m):
        j = np.mod(i, x_shape[1])
        yt[i, j] = 1
    "������a, b, c, ... , z�ƑΉ����Ă���"
    print("yt = ...")
    print(yt)
    print("np.shape(yt) : " + str(np.shape(yt)))
    
    """�ƌ���"""
    "�Ə����l�ݒ�"
    "bias�����܂܂�Ă���"
    theta = [0]
    theta[0] = initial_theta(n[0], n[1])
    print("theta[0]'s shape : " + str(np.shape(theta[0])))
    
    """�s��a"""
    a = [0, 0]
    a[0] = np.c_[np.ones((m, 1)), xtR]
    a[1] = [[0 for i in range(n[1])] for k in range(m)]
    
    print("a[0]'s shape : " + str(np.shape(a[0])))
    print("a[1]'s shape : " + str(np.shape(a[1])))
    
    "scaling"
    if scaling_switch == 1:
        a[0] = scaling(a[0])
    
    
    """cost�v�Z & �ƍX�V"""
    j_ite = [[0 for i in range(iteration+1)] for k in range(n[1])]
    j_ite = np.array(j_ite)
    #cost�L�^�p�ϐ��Bcost��logistic�̐�(n[1])�̐������}���ł���
    #j_ite = (n[1], iteration+1) ���X�V���Ă��Ȃ�J���L�^
    print("cost's shape : " + str(np.shape(j_ite)))

    "�����l�L�^"
    a[1] = sigmoid(h(theta[0], a[0]))
    j_ite[:, 0] = J(a[1], yt, theta) #j_ite��� = (n[1],)
    print("j_ite_first = ...")
    print(j_ite[:, 0])
    print("theta_first = ...")
    print(theta)
    print("del_j_ite's shape : " + str(np.shape(theta[0])))
    del_j = del_J(a, yt, theta)
    
    for i in range(iteration):
        theta[0] -= alpha * del_j
        a[1] = sigmoid(h(theta[0], a[0]))
        
        j_ite[:, i+1] = J(a[1], yt, theta) #cost�l�L�^
        
        del_j = del_J(a, yt, theta)
        
    print("j_last = ...")
    print(j_ite[:, iteration])
    print("theta_last = ...")
    print(theta)
    
    """�v���b�g"""
    "cost_iteration����"
#    ax = [0 for i in range(n[1])]
#    nrow = int(np.ceil(np.divide(float(n[1]), float(ncol))))
#    fig, ax = plt.subplots(nrows=nrow, ncols=ncol)
#    
#    
#    it = np.arange(0, iteration+1)
#    plot_count = 0
#    for i in range(nrow):
#        for k in range(ncol):
#            ax[i, k].plot(it, j_ite[plot_count, :])
#            plot_count += 1
#            
#            if plot_count >= n[1]:
#                break
    fig = plt.figure()
    j_ite_ave = np.average(j_ite, axis=0)
    print(np.shape(j_ite_ave))
    plt.plot(np.arange(0, iteration+1, 1), j_ite_ave)
    plt.show()
    
    plt.show()
    
    """���x�v�Z"""
    "���x���ϊ�"
    p_max = a[1].max(axis=1)
    p_max = np.reshape(p_max, (m, 1))
#    print(np.argmax(a[2], axis=1))
    "�ő�l�ȊO�O"
    a1_copy = a[1]
    a1_copy[a[1] < p_max] = 0
    a1_copy[a[1] >= p_max] = 1

    "�e�r�b�g�]��"
    yp = a1_copy
    bit_hit = yp==yt
    "�e�A���t�@�x�b�g�]��"
    alpha_hit = np.all(bit_hit, axis=1)
    "�����v�f���o"
    hit = float(alpha_hit[alpha_hit == True].size)
    print("������ : " + str(int(hit)) + "��" + " / " + str(m) + "��")
    m = float(m)
    ratio = 100. * (hit / m)
    print("���x : " + str(ratio) + "%")

    
Neural_Forward()
raw_input("end program")

"""
�ۑ����ׂ��f�[�^
�Ea[0], yt
�Ej_ite_ave
"""