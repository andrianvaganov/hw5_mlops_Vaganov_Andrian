import pandas as pd
df = pd.read_csv("https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")
df.dropna(inplace=True)
df.to_csv("energydata_complete_v1.csv")