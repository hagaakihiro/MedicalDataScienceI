import random
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import sys

#from scipy.optimize import minimize
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
    # num_basis += 1 # for constant term
    lamb = np.double(24.2)
    
    basis = ridge_multi_basis_set_calc(num_basis,df_train,input_name)
    print(basis.shape)


    V, singular_values, Vdagger = svd(basis.T)
    print(singular_values**2)
    dof = np.sum(singular_values**2/(singular_values**2+lamb))
    print("Effective DOF: ", dof)
    #sys.exit()
    # data to be fitted
    w0 = df_train["lpsa"].mean()
    y_true = df_train["lpsa"] - w0
    w = np.zeros(num_basis)
    #sys.exit()

    direct_w = direct_weight_optimize(y_true,basis,lamb)
    print(w0)
    print(direct_w)
    #sys.exit()
    
    # 上記のdf_testに対するフィッティング精度（RMSE）を算出
    basis_test = ridge_multi_basis_set_calc(num_basis,df_test,input_name)
    fitted_1 = np.dot(direct_w,basis_test) + w0
    rmse_val = np.sum((fitted_1-df_test["lpsa"])**2)/(len(fitted_1))
    # グラフにRMSEを記載
    rmse_d = "rmse_d = %f" % rmse_val
    print(rmse_d)

    # Table
    input_name0 = ["intercept"]
    input_name0.extend(input_name)
    input_name0 = pd.DataFrame([input_name0]).T
    direct_w0 = [w0]
    direct_w0.extend(direct_w)
    direct_w0 = pd.DataFrame([direct_w0]).T
    #beta_std0 = pd.DataFrame(beta_std)
    output_df = pd.concat([input_name0, direct_w0],axis=1)
    output_df.columns = ["Params", "Coef."]
    print(output_df)
