import pandas as pd

df = pd.read_stata("data2.dta")
df.to_excel("data_yx.xlsx", index=False)