import pandas as pd

df = pd.read_csv('./data/WSD/7.csv')

df_train = df
df_train = df_train.fillna(0)
import numpy as np
value = np.asarray(df['value'])
print(value)
print(df)
print(df_train)