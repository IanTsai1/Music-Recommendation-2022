import pandas as pd
import numpy as np
from datetime import datetime


df = pd.read_csv('./data/members.csv',nrows=5000)
#test_songs = pd.read_csv('fix_test.csv')

for i in range(len(df['registration_init_time'])):
    month = str(df['registration_init_time'][i])
    year = str(df['registration_init_time'][i])
    df.loc[i,'month'] = int(month[5])
    df.loc[i, 'year'] = int(year[:4])


for i in range(len(df['registration_init_time'])):
    d = str(df['registration_init_time'][i])
    d = '-'.join([d[:4], d[4:6], d[6:]])
    df.loc[i, 'registration_init_time'] = d



for i in range(len(df['registration_init_time'])):
    d1 = datetime.strptime(df['registration_init_time'][i], "%Y-%m-%d")
    d2 = datetime.strptime(df['expiration_date'][i], "%Y-%m-%d")
    df.loc[i,'purchased_days'] = abs((d2 - d1).days)


print(df['registration_init_time'].dtypes)

#df.to_csv('updated.csv', index=False)