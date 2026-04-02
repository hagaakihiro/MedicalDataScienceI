# -*- coding: utf-8 -*-
#import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

df = pd.read_csv("output.csv", encoding="SHIFT-JIS")
print(df)
df["total"] = df["sin_2px"] + df["noise"]
print(df)
#df.to_csv("output2.csv", index=None, encoding="SHIFT-JIS")




