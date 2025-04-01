#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.stats import norm
import math
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

def Class_Sampling(m1_1, s1_1, m2_1, s2_1, num_1, m1_2, s1_2, m2_2, s2_2, num_2, m3_1, s3_1, m3_2, s3_2):
    
    c1_data_x1 = []
    c1_data_x2 = []
    c1_data_x3 = []

    c1_data_x1=np.random.normal(m1_1,s1_1,num_1)
    c1_data_x2=np.random.normal(m2_1,s2_1,num_1)
    c1_data_x3=np.random.normal(m3_1,s3_1,num_1)


    c1_data_x1 = np.append(c1_data_x1, np.random.normal(m1_2,s1_2,num_2))
    c1_data_x2 = np.append(c1_data_x2, np.random.normal(m2_2,s2_2,num_2))
    c1_data_x3 = np.append(c1_data_x3, np.random.normal(m3_2,s3_2,num_2))

    return c1_data_x1, c1_data_x2, c1_data_x3
# Class 1 sampling -- end

"""
# Bayes = 0.5
x0 = 0
for j in range(-20,20):
    x2 = j*0.1
    for i in range(-20,20):
        x1 = i*0.1
        X_C1 = 0.5*norm.pdf(x1,m1_1,s1_1)*norm.pdf(x2,m2_1,s2_1)+ 0.5*norm.pdf(x1,m1_2,s1_2)*norm.pdf(x2,m2_2,s2_2)

        X_C2 = 0.8*norm.pdf(x1,m1_21,s1_21)*norm.pdf(x2,m2_21,s2_21)+0.2*norm.pdf(x1,m1_22,s1_22)*norm.pdf(x2,m2_22,s2_22)

        X = X_C2/(X_C1+X_C2)
        d_data = pd.DataFrame([[x1,x2,X]])
        if X > 0.49 and X < 0.51:
            #print(x1,x2,X)
            if x0 == 1:
                df2 = pd.concat([df2, d_data])
            else:
                df2 = d_data
            x0 = 1
df2.columns = ["x1","x2","pdf"]
print(df2)
#df2["pdf"] = np.array(X)
#print(df2)
#df3 = df2[df2["pdf"]>0.45 and df2["pdf"]<-0.55]
#print(df3)
"""

if __name__ == '__main__':
    
    input_name = "CT-to-ED.csv"
    df = pd.read_csv(input_name)
    print(df)
    #sys.exit()
    
    output_name = "CT-to-ED_3D"
    
    output_name_csv = output_name + ".png"
    #df.to_csv(output_name_csv,index=False)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # 3Dプロット
    x = df["CT_value_1"]
    y = df["CT_value_2"]
    z = df["Physical_Density"]
    dx = dy = 0.5  # 棒の幅
    dz = df["Physical_Density"]  # 棒の高さ
    #ax.bar3d(x, y, z, dx, dy, dz, color='b', alpha=0.7)
    ax.scatter3D(x, y, z, "ro", label="Density")
    #ax.plot_trisurf(x, y, z, cmap="viridis", edgecolor="none", alpha=0.8)
    
    #plt.xlabel("CT_low")
    #plt.ylabel("CT_high")
    #plt.plot(df2["x1"], df2["x2"], ".", color="black")
    #plt.xlim(-2,2)
    #plt.ylim(-2,2)
    #plt.legend()
    plt.tight_layout()
    plt.show()

