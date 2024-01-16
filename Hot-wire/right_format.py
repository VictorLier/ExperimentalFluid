import pandas as pd
import numpy as np


#import data
df = pd.read_csv('Hot-wire/0_1D.csv')
df_2 = pd.read_csv('Hot-wire/7D.csv')

df['Radial position'] = np.around(df['Radial position']*47.33/1000,3)
df_2['Radial position'] = np.around(df_2['Radial position']*47.33/1000,3)

#add column with ,
df.insert(0, '0.005', None)


df_2.insert(0, '0.7', None)

print(df)


#transpose data
df = df.T
df_2 = df_2.T

# add row with one number





#save data
df.to_csv('Hot-wire/0_1D_new.csv', index=True, header=False)
df_2.to_csv('Hot-wire/7D_new.csv', index=True, header=False)

