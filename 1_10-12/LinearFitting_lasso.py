import random
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from scipy.linalg import svd
# Analytical solution
def direct_weight_optimize(y, basis, lamb):
    prevec = np.dot(basis, y)
    #premat = np.linalg.inv(lamb * np.identity(len(prevec)) + np.dot(basis,basis.T))
    #ww = np.dot(premat,prevec)
    ww = np.linalg.solve(np.dot(basis,basis.T)+lamb * np.identity(len(prevec)),prevec)
    return ww

# basis set
def ridge_multi_basis_set_calc(num_basis, df, input_name):
    # set basis function
    ic = 0
    for name0 in input_name:
        if ic == 0:
            basis = df[name0]
            ic = 1
        else:
            basis = np.append(basis, df[name0], axis=0)
    basis = np.reshape(basis, (num_basis, len(df)))
    return basis

def multi_basis_set_calc(num_basis, df, input_name):
    # set basis function
    basis = np.ones(len(df))
    for name0 in input_name:
        basis = np.append(basis, df[name0], axis=0)
    basis = np.reshape(basis, (num_basis, len(df)))
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

def lasso_objectivefunction(w, y, basis, lamb):
    y_pre = np.dot(w, basis)
    oval = 0.5*np.sum(np.square(y - y_pre))
    penalty = 0.5*lamb*np.sum(np.abs(w))
    return oval + penalty

def load_data(file_path, data_type):
    if data_type == 0:
        # カンマ区切り（CSV）の場合
        df = pd.read_csv(file_path)
    elif data_type == 1:
        # タブ区切りの場合
        df = pd.read_csv(file_path, sep="\t")
    else:
        # スペース区切りの場合
        df = pd.read_csv(file_path, delim_whitespace=True)
    return df

#---- main ----
if __name__ == '__main__':

    file_path = "prostate.data"
    data_type = 1 # 0 comma, 1 tab, 2 space
    df = load_data(file_path, data_type)
    # 1列目を除く
    df = df.iloc[:, 1:]
    # データの確認
    if df is not None:
        print(df.head())  # 最初の5行を表示
    
    input_name = ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp',
       'gleason', 'pgg45']
    
    # Normalization as z-value
    for name0 in input_name:
        df[name0] = (df[name0]-df[name0].mean())/df[name0].std()
    df_train = df[df["train"]=="T"]
    df_test = df[df["train"]=="F"]
    #print(df_train)
    #print(df_test)
    #sys.exit()
    
    #plt.scatter(df["x"],df["t"])
    #plt.show()
    #sys.exit()

    # input parameters
    num_basis = len(input_name)
    num_basis += 1 # for constant term
    lamb = np.double(10)
    
    # LASSOも説明変数を中心化しておく。w0は予測変数（y or t）の平均で与えられる
    basis = multi_basis_set_calc(num_basis,df_train,input_name)
    print(basis.shape)

    """
    V, singular_values, Vdagger = svd(basis.T)
    print(singular_values**2)
    dof = np.sum(singular_values**2/(singular_values**2+lamb))
    print("Effective DOF: ", dof)
    #sys.exit()
    """
    # data to be fitted
    #w0 = df_train["lpsa"].mean()
    y_true = df_train["lpsa"]# - w0
    w = np.zeros(num_basis)
    #sys.exit()

    #direct_w = direct_weight_optimize(y_true,basis,lamb)
    # Numerical solution : LASSOでは解析解は得られないため、数値的に解く必要がある
    result = minimize(lasso_objectivefunction, x0=w, args=(y_true, basis, lamb), tol=0, options={"maxiter":500}, method='Powell')
    #print(w0)
    print(result.x)
    #sys.exit()
    direct_w = result.x
    # 上記のdf_testに対するフィッティング精度（RMSE）を算出
    basis_test = multi_basis_set_calc(num_basis,df_test,input_name)
    fitted_1 = np.dot(direct_w,basis_test) #+ w0
    rmse_val = np.sqrt(np.sum((fitted_1-df_test["lpsa"])**2)/(len(fitted_1)))
    # グラフにRMSEを記載
    rmse_d = "rmse_d = %f" % rmse_val
    print(rmse_d)
    # true-pred fig.
    plt.plot(np.linspace(0,5,2),np.linspace(0,5,2),ls="-",color="black",lw=0.5)
    plt.scatter(df_test["lpsa"], fitted_1)
    plt.xlabel("lpsa")
    plt.ylabel("pred")
    plt.text(3,1,rmse_d)
    plt.ylim(0,5)
    plt.xlim(0,5)
    plt.show()
    # Table
    input_name0 = ["intercept"]
    input_name0.extend(input_name)
    input_name0 = pd.DataFrame([input_name0]).T
    direct_w0 = pd.DataFrame(direct_w)

    #direct_w0 = [w0]
    #direct_w0.extend(direct_w)
    #direct_w0 = pd.DataFrame([direct_w0]).T
    #beta_std0 = pd.DataFrame(beta_std)
    output_df = pd.concat([input_name0, direct_w0],axis=1)
    output_df.columns = ["Params", "Coef."]
    print(output_df)
