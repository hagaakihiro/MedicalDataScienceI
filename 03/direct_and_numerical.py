#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Input data; E3_output2.csv
# Output data; Weights in polynomial fit
#---
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import sys

from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

# Analytical solution
def direct_weight_optimize(y, basis, lamb):
    prevec = np.dot(basis, y)
    premat = np.linalg.inv(lamb * np.identity(len(prevec)) + np.dot(basis,basis.T))
    ww = np.dot(premat,prevec)
    return ww

# basis set
def basis_set_calc(num_basis, x_n):
    # set basis function
    for idata in range(0,num_basis):
        if idata == 0:
            basis = x_n**idata
        else:
            basis = np.append(basis, x_n**idata, axis=0)
    basis = np.reshape(basis, (num_basis, len(x_n)))
    return basis

def gauss_basis_set_calc(num_basis, x_n ,mu, sigma):
    # set basis function
    for idata in range(0,num_basis):
        if idata == 0:
            basis = x_n**idata
        else:
            basis_0 = np.exp(-(x_n - mu[idata])**2 / (2*sigma[idata]**2))
            basis = np.append(basis, basis_0, axis=0)
    basis = np.reshape(basis, (num_basis, len(x_n)))
    return basis

# Objective function (loss function)
def objectivefunction(w, y, x, basis, lamb):
    y_pre = np.dot(w, basis)
    oval = 0.5*np.sum(np.square(y - y_pre))
    penalty = 0.5*lamb*np.sum(np.square(w))
    return oval + penalty

# Derivative of objective function
def gradient(w, y, x, basis, lamb):
    premat = np.dot(basis,basis.T)
    g_pre = (np.dot(premat,w) - np.dot(basis,y))
    g_penalty = lamb * w
    grad = g_pre + g_penalty
    return grad
    
#---- main ----
if __name__ == '__main__':
    # input data
    df = pd.read_csv("output2.csv", encoding="SHIFT-JIS")
    print(df) # x, sin_2px, noise, total
    
    # input parameters
    num_basis = 10 # num. of polynomials
    num_basis += 1 # for constant term
    lamb = np.double(0.0005)
    
    basis = basis_set_calc(num_basis,df["x"])
    #mu = np.linspace(0, 1, num_basis)
    #sigma = np.ones(num_basis)
    #basis = gauss_basis_set_calc(num_basis,df["x"],mu,sigma)
    print(basis.shape)

    # data to be fitted
    y_true = df["total"]
    w = np.zeros(num_basis)
    #sys.exit()

    #direct_w = direct_weight_optimize(y_true,basis,lamb)
    result = minimize(objectivefunction, x0=w, args=(y_true, df["x"], basis, lamb), jac=gradient, tol=10**(-10), method='SLSQP')
    numerical_w = result.x
    #print(numerical_w)
    #sys.exit()
    # dataの表示
    plt.plot(df["x"],y_true, "ro", label="data")
    
    # 上記のdf["total"]に対するフィッティング精度（RMSE）を算出
    basis = basis_set_calc(num_basis,df["x"])
    fitted_1 = np.dot(numerical_w,basis)
    rmse_val = np.sum((fitted_1-df["total"])**2)/(len(fitted_1))
    # グラフにRMSEを記載
    rmse_d = "rmse_d = %f" % rmse_val
    plt.text(0.6,0.5, rmse_d)
    # xを0~1で100点に刻む（x = 0, 0.01, 0.02,,,,1)
    x = np.linspace(0, 1, 100)
    # 上記のxに対してsin 2*pi*xをプロット
    plt.plot(x,np.sin(2*np.pi*x), label="true")
    #basis = gauss_basis_set_calc(num_basis,x,mu,sigma)
    # 上記のxに対してフィッティング結果を表示
    basis = basis_set_calc(num_basis,x)
    fitted_2 = np.dot(numerical_w,basis)
    plt.plot(x,fitted_2, label="predict_numerical", color="green")
    
    plt.ylim(-1.5,1.5)
    #plt.plot(data[:,0], data[:,4])
    plt.legend()
    plt.show(block=True)
