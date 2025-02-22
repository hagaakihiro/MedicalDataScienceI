#!/usr/bin/env python3

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from sklearn import svm


if __name__ == '__main__':
    # Data loading
    dataname = "producted_data.csv"
    df = pd.read_csv(dataname, encoding="SHIFT-JIS")
    
    # Data plot in 2D
    df_1 = df[df["class"] == 1]
    df_2 = df[df["class"] == 2]
    
    plt.plot(df_1["x1"], df_1["x2"], "ro", label="class 1")
    plt.plot(df_2["x1"], df_2["x2"], "bo", label="class 2")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 学習データの前処理
    x_train = np.array(df[["x1", "x2"]].values)
    y_train = np.array(df["class"])
    
    # 識別モデルの構築 -- support vector machine
    # Default C=1.0, kernel="rbf"
    clf = svm.SVC(C=1.0, kernel="rbf")
    #clf = svm.NuSVC()
    clf.fit(x_train, y_train)
    
    #モデルの識別精度
    pre = clf.predict(x_train)
    print(pre)
    
    #分割表の作成
    num_class = 2
    re_t = np.zeros(num_class*num_class,dtype ='int')
    table = np.reshape(re_t, (num_class, num_class))*0
    for i in range(len(y_train)):
        table[y_train[i]-1,pre[i]-1] += 1
    print(table)


    
    # 2D boundary
    #ac_score = metrics.accuracy_score(y_train, pre)
    #print(ac_score)

    #print(x_train)
    i0 = 0
    for j in range(-20,20):
        x2 = j*0.1
        for i in range(-20,20):
            x1 = i*0.1
            d_data = pd.DataFrame([[x1, x2]])
            if i0 == 0:
                df2 = d_data
                i0 = 1
            else:
                df2 = pd.concat([df2,d_data])
    df2.columns = ["x1", "x2"]
    #print(df2)
    x_pre = np.array(df2[["x1", "x2"]].values)
    y_pre = clf.predict(x_pre)
    df2["y_pre"] = y_pre
    print(df2)

    x = np.arange(-2,2,0.1)
    y = np.arange(-2,2,0.1)
    X, Y = np.meshgrid(x,y)
    Z = np.array(df2["y_pre"]).reshape(40,40)

    plt.contour(X, Y, Z, levels=[1.5], colors = "black", linestyles="-",linewidths=3)

    df3 = pd.read_csv("True_boundary_rough.csv", encoding="SHIFT-JIS")
    ZZ = np.array(df3["pdf"]).reshape(40,40)
    
    plt.contour(X, Y, ZZ, levels=[0.5])

    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.legend()

    df3_0 = df3[df3["pdf"]<0.5]
    df3_1 = df3[df3["pdf"]>=0.5]
    plt.scatter(df3_0["x1"], df3_0["x2"], color = "red", s=0.5)
    plt.scatter(df3_1["x1"], df3_1["x2"], color = "blue", s=0.5)

    plt.tight_layout()
    #plt.show()
    plt.savefig("SVM_result.png")
    
