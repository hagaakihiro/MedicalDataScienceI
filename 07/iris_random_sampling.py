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
    """
    # Data Splitting 1: random sampling
    num_train = np.int16(num_data * data_split[0])
    num_validation = np.int16(num_data * data_split[1])
    num_test = num_data - num_train - num_validation
    df_train = df.sample(num_train)
    df_drop = df.drop(index=df_train.index)
    df_validation = df_drop.sample(num_validation)
    df_test = df_drop.drop(index=df_validation.index)
    
    print("train:", len(df_train), "setosa:", len(df_train[df_train["Species"]=="setosa"]))
    print(df_train.index)
    print("validation:", len(df_validation),"setosa:", len(df_validation[df_validation["Species"]=="setosa"]))
    print(df_validation.index)
    print("test:", len(df_test),"setosa:", len(df_test[df_test["Species"]=="setosa"]))
    print(df_test.index)
    #sys.exit()
    # -- end
    """
    # Data Splitting 2: random sampling with same class ratio
    df_setosa = df[df["Species"]=="setosa"]
    df_versicolor = df[df["Species"]=="versicolor"]
    df_virginica = df[df["Species"]=="virginica"]

    num_setosa = len(df_setosa)
    num_versicolor = len(df_versicolor)
    num_virginica = len(df_virginica)
    
    # setosa
    num_setosa_train = np.int16(num_setosa * data_split[0])
    num_setosa_validation = np.int16(num_setosa * data_split[1])
    num_setosa_test = num_setosa - num_setosa_train - num_setosa_validation
    df_setosa_train = df_setosa.sample(num_setosa_train)
    df_setosa_drop = df_setosa.drop(index=df_setosa_train.index)
    df_setosa_validation = df_setosa_drop.sample(num_setosa_validation)
    df_setosa_test = df_setosa_drop.drop(index=df_setosa_validation.index)

    # versicolor
    num_versicolor_train = np.int16(num_versicolor * data_split[0])
    num_versicolor_validation = np.int16(num_versicolor * data_split[1])
    num_versicolor_test = num_versicolor - num_versicolor_train - num_versicolor_validation
    df_versicolor_train = df_versicolor.sample(num_versicolor_train)
    df_versicolor_drop = df_versicolor.drop(index=df_versicolor_train.index)
    df_versicolor_validation = df_versicolor_drop.sample(num_versicolor_validation)
    df_versicolor_test = df_versicolor_drop.drop(index=df_versicolor_validation.index)

    # virginica
    num_virginica_train = np.int16(num_virginica * data_split[0])
    num_virginica_validation = np.int16(num_virginica * data_split[1])
    num_virginica_test = num_virginica - num_virginica_train - num_virginica_validation
    df_virginica_train = df_virginica.sample(num_virginica_train)
    df_virginica_drop = df_virginica.drop(index=df_virginica_train.index)
    df_virginica_validation = df_virginica_drop.sample(num_virginica_validation)
    df_virginica_test = df_virginica_drop.drop(index=df_virginica_validation.index)

    # Combine
    df_train = pd.concat([df_setosa_train, df_versicolor_train, df_virginica_train])
    df_validation = pd.concat([df_setosa_validation, df_versicolor_validation, df_virginica_validation])
    df_test = pd.concat([df_setosa_test, df_versicolor_test, df_virginica_test])
    print("train:", len(df_train), "setosa:", len(df_train[df_train["Species"]=="setosa"]))
    print(df_train.index)
    print("validation:", len(df_validation),"setosa:", len(df_validation[df_validation["Species"]=="setosa"]))
    print(df_validation.index)
    print("test:", len(df_test),"setosa:", len(df_test[df_test["Species"]=="setosa"]))
    print(df_test.index)
    #sys.exit()
    # -- end

    """
    # Data Splitting 3: 5-fold
    kfold = 5
    for k in range(kfold):
        df_test = df.iloc[k::kfold,:]
        kv = k + 1
        if kv == kfold:
            kv = 0
        df_validation = df.iloc[kv::kfold,:]
        
        df_drop = df.drop(index=df_test.index)
        df_train = df_drop.drop(index=df_validation.index)

        print(k, "train:", len(df_train), "setosa:", len(df_train[df_train["Species"]=="setosa"]))
        #print(df_train.index)
        print(k, "validation:", len(df_validation),"setosa:", len(df_validation[df_validation["Species"]=="setosa"]))
        #print(df_validation.index)
        print(k, "test:", len(df_test),"setosa:", len(df_test[df_test["Species"]=="setosa"]))
        #print(df_test.index)
    sys.exit()

    """
    # Preprocessing
    x_train = np.array(df_train[["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]].values)
    #x_train = np.array(df[["Sepal Length"]].values)
    #x_train = np.array(df[["Sepal Width"]].values)
    #x_train = np.array(df[["Petal Length"]].values)
    #x_train = np.array(df[["Petal Width"]].values)
    y_train = np.array(df_train["class"])
    
    # Model -- Logistic Regression
    lr = LogisticRegression(penalty="l2", C=0.1)
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
    # 訓練結果
    pre_train = lr.predict(x_train)
    re_t = np.zeros(num_class*num_class,dtype ='int')
    table = np.reshape(re_t, (num_class, num_class))*0
    for i in range(len(y_train)):
        table[y_train[i]-1,pre_train[i]-1] += 1
    print(table)
    print("train accuracy: ", (table[0,0]+table[1,1]+table[2,2])/len(y_train))

    # 検証結果
    re_t = np.zeros(num_class*num_class,dtype ='int')
    table = np.reshape(re_t, (num_class, num_class))*0
    for i in range(len(y_validation)):
        table[y_validation[i]-1,pre[i]-1] += 1
    print(table)
    print("validation accuracy: ", (table[0,0]+table[1,1]+table[2,2])/len(y_validation))
    
    
    # テスト結果
    x_test = np.array(df_test[["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]].values)
    y_test = np.array(df_test["class"])
    pre_test = lr.predict(x_test)
    re_t = np.zeros(num_class*num_class,dtype ='int')
    table = np.reshape(re_t, (num_class, num_class))*0
    for i in range(len(y_test)):
        table[y_test[i]-1,pre_test[i]-1] += 1
    print(table)
    print("test accuracy: ", (table[0,0]+table[1,1]+table[2,2])/len(y_test))
