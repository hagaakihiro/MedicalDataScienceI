import random
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import sys

#from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

# Analytical solution
def direct_weight_optimize(y, basis, lamb):
    prevec = np.dot(basis, y)
    #premat = np.linalg.inv(lamb * np.identity(len(prevec)) + np.dot(basis,basis.T))
    #ww = np.dot(premat,prevec)
    ww = np.linalg.solve(np.dot(basis,basis.T),prevec)
    return ww

# basis set
def multi_basis_set_calc(num_basis, df, input_name):
    # set basis function
    basis = np.ones(len(df))
    for name0 in input_name:
        basis = np.append(basis, df[name0], axis=0)
    basis = np.reshape(basis, (num_basis, len(df)))
    return basis


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
    print(df_train)
    #print(df_test)
    #sys.exit()
    
    #plt.scatter(df["x"],df["t"])
    #plt.show()
    #sys.exit()

    # input parameters
    num_basis = len(input_name)
    num_basis += 1 # for constant term
    lamb = np.double(0.0)
    
    basis = multi_basis_set_calc(num_basis,df_train,input_name)
    print(basis.shape)

    # data to be fitted
    y_true = df_train["lpsa"]
    w = np.zeros(num_basis)
    #sys.exit()

    direct_w = direct_weight_optimize(y_true,basis,lamb)
    print(direct_w)
    #sys.exit()
    
    # 上記のdf_testに対するフィッティング精度（RMSE）を算出
    basis_test = multi_basis_set_calc(num_basis,df_test,input_name)
    fitted_1 = np.dot(direct_w,basis_test)
    rmse_val = np.sqrt(np.sum((fitted_1-df_test["lpsa"])**2)/(len(fitted_1)))
    # RMSEを記載
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

    # (Xt X)の逆行列を求める BishopのS_N, ここでbeta = rmse
    V = np.linalg.inv(np.dot(basis,basis.T))
    #print(V)
    # 求めた係数の標準誤差
    beta_std = []
    for i in range(len(V)):
        beta_std.append(np.sqrt(V[i,i]*rmse_val))
    #print(beta_std)
    
    # Table
    input_name0 = ["intercept"]
    input_name0.extend(input_name)
    input_name0 = pd.DataFrame([input_name0]).T
    direct_w0 = pd.DataFrame(direct_w)
    beta_std0 = pd.DataFrame(beta_std)
    output_df = pd.concat([input_name0, direct_w0, beta_std0, direct_w0/beta_std0],axis=1)
    output_df.columns = ["Params", "Coef.", "RMSE", "Z-score"]
    print(output_df)
