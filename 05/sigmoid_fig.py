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

#from scipy.optimize import minimize
#from sklearn.metrics import mean_squared_error

# Analytical solution
def direct_weight_optimize(y, basis, lamb):
    prevec = np.dot(basis, y)
    premat = np.linalg.inv(lamb * np.identity(len(prevec)) + np.dot(basis,basis.T))
    ww = np.dot(premat,prevec)
    return ww

# basis set
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

#---- main ----
if __name__ == '__main__':

    a0 = 90
    alp = 0.08
    x = np.linspace(0, 400, 100)
    y = 0.5/(1+np.exp(-alp*(a0-x)))+1
    plt.plot(x,y,color="red")
    plt.legend()
    plt.show(block=True)
    sys.exit()
    
    
    
