#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Input data; E3_output2.csv
# Output data; Weights in gauss fit
#---
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import sys

import numpy.linalg as LA

from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

# Analytical solution: Eq.(3.15)
def direct_weight_optimize(y, basis, lamb):
    prevec = np.dot(basis, y)
    #premat = np.linalg.inv(lamb * np.identity(len(prevec)) + np.dot(basis,basis.T))
    #ww = np.dot(premat,prevec)
    ww = np.linalg.solve(lamb * np.identity(len(prevec)) + np.dot(basis,basis.T),prevec)
    return ww

# Covariant S : Eq.(3.59)
def cov_s(n, basis, lamb):
    premat = np.linalg.inv(lamb * np.identity(n) + np.dot(basis,basis.T))
    variance = np.dot(basis.T,np.dot(premat,basis))
    return variance

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

# basis set for gauss
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

# basis set for polynomial
def basis_set_calc(num_basis, x_n):
    # set basis function
    for idata in range(0,num_basis):
        if idata == 0:
            basis = x_n**idata
        else:
            basis = np.append(basis, x_n**idata, axis=0)
    basis = np.reshape(basis, (num_basis, len(x_n)))
    return basis

#---- main ----
if __name__ == '__main__':
    
    # select basis used.
    #k_basis = "gauss"
    k_basis = "polynomial"
    
    # input data generation
    #num_data = 25
    #rang = 1.0/(num_data-1)
    #df = pd.DataFrame(np.arange(num_data)*rang,columns=["x"])
    #df["sin_2px"]=df.x.apply(lambda x: math.sin(2*x*math.pi))
    #df["noise"]=np.random.normal(0,0.3,num_data)
    #df["total"]=df[["sin_2px","noise"]].sum(axis=1)
    #df.to_csv("k-fold.csv", index=None, encoding="SHIFT-JIS")
    df = pd.read_csv("k-fold.csv", encoding="SHIFT-JIS")
    #print(df) # x, sin_2px, noise, total
    
    # plot (to see input data)
    plt.plot(df["x"],df["total"], "ro", label="data")
    x = np.linspace(0, 1, 100)
    plt.plot(x,np.sin(2*np.pi*x), label="true")
    plt.legend()
    plt.show(block=True)
    plt.close()
    
    #sys.exit()
    
    # input parameters
    num_basis = 10 # num. of basis
    num_basis += 1 # for constant term
    
    if k_basis == "gauss":
        # design matrix for gaussian basis
        mu = np.linspace(0, 1, num_basis)
        sigma = np.ones(num_basis)*0.3
        basis = gauss_basis_set_calc(num_basis,df["x"],mu,sigma)
    elif k_basis == "polynomial":
        # design matrix for gaussian basis
        basis = basis_set_calc(num_basis, df["x"])
    #print(basis.shape)

    # data to be fitted
    y_true = df["total"]
    
    # eigen value
    eig , eig_vec = LA.eig(np.dot(basis , basis.T))
    # initial value
    alpha = 0.0001
    beta = 1
    # max num. of iteration
    num = 20
    for ite in range(num):
        gam = np.sum(eig/(alpha/beta+eig))
        direct_w = direct_weight_optimize(y_true,basis,alpha/beta)
        mTm = np.dot(direct_w.T , direct_w)
        alpha = gam / mTm
        
        fitted = np.dot(direct_w,basis)
        beta = (len(y_true)-gam)/mean_squared_error(fitted, y_true)
        print("ite=", ite, alpha, beta, gam)
    lamb = alpha/beta
    print("lamb = ", lamb)
    plt.plot(df["x"],y_true, "ro", label="data")
    x = np.linspace(0, 1, 100)
    plt.plot(x,np.sin(2*np.pi*x), label="true")
    
    if k_basis == "gauss":
        basis = gauss_basis_set_calc(num_basis,x,mu,sigma)
    elif k_basis == "polynomial":
        basis = basis_set_calc(num_basis,x)
    fitted_2 = np.dot(direct_w,basis)
    plt.plot(x,fitted_2, label="predict",color="green")
    rmse_d = "rmse_d = %f" % mean_squared_error(fitted_2, np.sin(2*np.pi*x))
    plt.text(0.565,0.4, rmse_d)
    plt.ylim(-1.5,1.5)
    plt.legend()
    plt.show(block=True)
    #sys.exit()


    # predictive distribution
    s = cov_s(num_basis, basis, lamb)
    s = np.diag(s)
    #beta = (1/0.3)**2
    ss = np.sqrt((1+s)/beta)
    #print(np.diag(ss))
    plt.plot(df["x"],y_true, "ro", label="data")
    plt.plot(x,fitted_2+ss, ls=":", color="red")
    plt.plot(x,fitted_2-ss, ls=":", color="red")
    plt.plot(x,fitted_2, color="red")
    plt.show(block=True)


