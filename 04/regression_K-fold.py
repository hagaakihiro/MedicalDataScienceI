#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Input data; k-fold.csv.csv
# Output data; Weights in fit
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
    df_all = pd.read_csv("k-fold.csv", encoding="SHIFT-JIS")
    #print(df) # x, sin_2px, noise, total
    
    # plot (to see input data)
    plt.plot(df_all["x"],df_all["total"], "ro", label="data")
    x = np.linspace(0, 1, 100)
    plt.plot(x,np.sin(2*np.pi*x), label="true")
    plt.legend()
    plt.show(block=True)
    plt.close()
    #sys.exit()
    # k-fold
    kf = 5
    data_min_lamb = []
    for k in range(kf):
        df_valid = df_all[k::kf] # k-fold
        #df_valid = df_all.sample(kf) # random sampling
        df = df_all.drop(df_valid.index)
        print(df_valid)
        print(df)
        #sys.exit()
    
        # input parameters
        num_basis = 10 # num. of basis
        num_basis += 1 # for constant term
        # Hyper parameter
        lamb = np.double(  1.0  ) # initial value
        
        
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
        #print(w)

        min_mse = 1000000
        for L in range(100):
            w = np.zeros(num_basis)
            lamb *= 0.8
            
            # optimization
            # Use Analytical solution
            direct_w = direct_weight_optimize(y_true,basis,lamb)
            #print(direct_w)
        
            # Validation
            x = df_valid["x"]
            y = df_valid["total"]
            if k_basis == "gauss":
                basis_valid = gauss_basis_set_calc(num_basis,x,mu,sigma)
            elif k_basis == "polynomial":
                basis_valid = basis_set_calc(num_basis,x)
            fitted = np.dot(direct_w,basis_valid)
            mse_value = mean_squared_error(fitted, y)
            if mse_value < min_mse:
                min_lamb = lamb
                min_mse = mse_value
        data_min_lamb.append(min_lamb)
        #print(min_lamb, min_mse)
    print(data_min_lamb)
    lamb = np.mean(data_min_lamb)
    print("lamb =", lamb)
    direct_w = direct_weight_optimize(y_true,basis,lamb)
    #sys.exit()
    # optimized weight is given by "direct_w" or "result.x"
    # plot result:
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
    #plt.plot(data[:,0], data[:,4])
    plt.legend()
    plt.show(block=True)
    sys.exit()

