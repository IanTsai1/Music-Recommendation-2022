import featuretools as ft
import pandas as pd
#from datetime import datetime
import datetime
import timeit
import time
import re


train_songs = pd.read_csv('fixfix_train.csv')

for i in range(len(train_songs['registration_init_time'])):
    date = str(train_songs['registration_init_time'][i])
    d = '-'.join([date[:4], date[4:6], date[6:]])
    train_songs.loc[i,'reg_month'] = int(date[5])
    train_songs.loc[i, 'reg_year'] = int(date[:4])
    train_songs.loc[i, 'registration_init_time'] = d

train_songs.to_csv('fixfix_train.csv', index=False)
print("YAY")
'''
print("19")

for i in range(len(train_songs['expiration_date'])):
    date = str(train_songs['expiration_date'][i])
    d = '-'.join([date[:4], date[4:6], date[6:]])
    train_songs.loc[i, 'exp_month'] = int(date[5])
    train_songs.loc[i, 'exp_year'] = int(date[:4])
    train_songs.loc[i, 'expiration_date'] = d

print("line28")

for i in range(len(train_songs['registration_init_time'])):
    d1 = datetime.strptime(train_songs['registration_init_time'][i], "%Y-%m-%d")
    d2 = datetime.strptime(train_songs['expiration_date'][i], "%Y-%m-%d")
    train_songs.loc[i,'purchased_days'] = abs((d2 - d1).days)

'''



df = pd.read_csv('testtest.csv')
start_time = time.time()


for i in range(len(df['registration_init_time'])):
    df.loc[i, 'month'] = int(str(df['registration_init_time'][i])[5])
    df.loc[i, 'year'] = int(str(df['registration_init_time'][i])[:4])

for i in range(len(df['registration_init_time'])):
    month = str(df['registration_init_time'][i])
    year = str(df['registration_init_time'][i])
    df.loc[i, 'month'] = int(month[5])
    df.loc[i, 'year'] = int(year[:4])

for i in range(len(df['registration_init_time'])):
    d = str(df['registration_init_time'][i])
    d = '-'.join([d[:4], d[4:6], d[6:]])
    df.loc[i, 'registration_init_time'] = d

for i in range(len(df['expiration_date'])):
    d = str(df['expiration_date'][i])
    d = '-'.join([d[:4], d[4:6], d[6:]])
    df.loc[i, 'expiration_date'] = d

for i in range(len(df['registration_init_time'])):
    pp = re.compile(r'(\d{4})-(\d{2})-(\d{2})')
    datetime.datetime(*map(int, pp.match(row).groups()))

def with_indexing(dstr):
    return datetime.datetime(*map(int, [dstr[:4], dstr[5:7], dstr[8:10]]))

for index, row in df.iterrows():
    df.loc[index,'expiration_date'] = with_indexing(int(row))
    print(row)

for i in range(len(df['registration_init_time'])):
    d1 = datetime.strptime(df['registration_init_time'][i], "%Y-%m-%d")
    d2 = datetime.strptime(df['expiration_date'][i], "%Y-%m-%d")
    df.loc[i,'purchased_days'] = abs((d2 - d1).days)

print("--- %s seconds ---" % (time.time() - start_time))




