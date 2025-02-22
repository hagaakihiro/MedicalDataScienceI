#!/usr/bin/env python3

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import svm

import sys

if __name__ == '__main__':
    # Data loading
    dataname = "iris_data.csv"
    df = pd.read_csv(dataname, encoding="SHIFT-JIS")
    num_data = len(df)
    data_split = [0.6,0.2,0.2]
    #print(df)

    kfold = 5
    df_test = df.iloc[0::kfold,:]
    df_drop = df.drop(index=df_test.index)

    # Data Splitting 3: 5-fold
    for k in range(kfold):
        df_validation = df_drop.iloc[k::kfold,:]
        df_train = df_drop.drop(index=df_validation.index)

        print(k, "train:", len(df_train), "setosa:", len(df_train[df_train["Species"]=="setosa"]))
        #print(df_train.index)
        print(k, "validation:", len(df_validation),"setosa:", len(df_validation[df_validation["Species"]=="setosa"]))
        #print(df_validation.index)
        print(k, "test:", len(df_test),"setosa:", len(df_test[df_test["Species"]=="setosa"]))
        #print(df_test.index)

        """
        # Preprocessing
        x_train = np.array(df_train[["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]].values)
        #x_train = np.array(df[["Sepal Length"]].values)
        #x_train = np.array(df[["Sepal Width"]].values)
        #x_train = np.array(df[["Petal Length"]].values)
        #x_train = np.array(df[["Petal Width"]].values)
        y_train = np.array(df_train["class"])
    
        # Model -- Logistic Regression
        lr = LogisticRegression(penalty="l2", C=1.0)
        lr.fit(x_train, y_train)
        # Model -- support vector machine
        # Default C=1.0, kernel="rbf"
        #clf = svm.SVC(C=1, kernel="rbf")
        #clf = svm.NuSVC()
        #clf.fit(x_train, y_train)
    
        # Model prediction
        x_validation = np.array(df_validation[["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]].values)
        y_validation = np.array(df_validation["class"])
        pre = lr.predict(x_validation)
        #pre = clf.predict(x_train)
        print(pre)
        #print(lr.coef_)
    
        # table
        num_class = 3
        re_t = np.zeros(num_class*num_class,dtype ='int')
        table = np.reshape(re_t, (num_class, num_class))*0
        for i in range(len(y_validation)):
            table[y_validation[i]-1,pre[i]-1] += 1
        print(table)
        print(table[0,0]+table[1,1]+table[2,2])
        """
    
