#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
import sys

if __name__ == '__main__':
    # Data loading
    dataname = "producted_data.csv"
    df = pd.read_csv(dataname, encoding="SHIFT-JIS")
    # Data loading -- end
    
    # Data plot
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
    #sys.exit()
    # Data plot -- end
    
    # Clasification
    x_train = np.array(df[["x1", "x2"]].values)
    y_train = np.array(df["class"])
    # 識別モデルの構築 -- Logistic regression model
    lr = LogisticRegression(penalty="none", max_iter=1000)
    #lr = LogisticRegression(penalty="l2", C=1.0)
    lr.fit(x_train, y_train)
    
    #w1,w2
    print("w1 = ", lr.coef_[0][0], "w2 = ", lr.coef_[0][1])
    w1 = lr.coef_[0][0]
    w2 = lr.coef_[0][1]
    #w0
    print("w0 = ", lr.intercept_[0])
    w0 = lr.intercept_[0]
    
    #モデルの識別精度
    pre = lr.predict(x_train)
    print(pre)
    #sys.exit()
    num_class = 2
    re_t = np.zeros(num_class*num_class,dtype ='int')
    table = np.reshape(re_t, (num_class, num_class))*0
    for i in range(len(y_train)):
        table[y_train[i]-1,pre[i]-1] += 1
    print(table)

    # 2D boundary
    # w1*x1 + w2*x2 + w0 = 0
    x1 = np.linspace(-2,2,101)
    y1 = (-lr.coef_[0][0]*x1 - lr.intercept_[0])/lr.coef_[0][1]
    plt.plot(x1,y1,color="black")
    
    x = np.arange(-2,2,0.1)
    y = np.arange(-2,2,0.1)
    X, Y = np.meshgrid(x,y)
    
    # 実際の境界（データ生成の際に使った確率分布の境界）
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
    plt.savefig("Logistic_result.png")
    
