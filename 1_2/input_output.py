import pandas as pd

df = pd.read_csv("output.csv", encoding="SHIFT-JIS")
print(df)
df["total"] = df["sin_2px"]+df["noise"]
print(df)
df.to_csv("output2.csv", index=None, encoding="SHIFT-JIS")
