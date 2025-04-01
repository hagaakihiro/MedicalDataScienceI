#!/usr/bin/env python3

import numpy as np
#import sklearn.cross_validation as crv
import pandas as pd
import math
import random
import matplotlib.pyplot as plt


numdata = 100
df = pd.DataFrame(np.arange(0,numdata)/(numdata),columns=["x"])
df["sin_2px"]=df.x.apply(lambda x: math.sin(2*x*math.pi))
df["noise"]=np.random.normal(0,0.3,numdata)
df["total"]=df[["sin_2px","noise"]].sum(axis=1)
plt.plot(df["x"],df["total"])
plt.show()
df.to_csv("test.csv",index=None)
#print(df.noise)

