from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import random
from scipy import stats
df = pd.read_csv("testtest.csv")

df.replace('?', np.nan) #maybe nan

def fillNaN_gender(df):
    gender = ['male','female']
    for i in range(len(df)):
        if  pd.isnull(df[i])== True:
            df.at[i] = gender[random.randint(0, 1)]

def find_mean(df):
    df.replace(0,np.nan,inplace=True) #removes 0 values so it won't affect mean
    df.fillna(df.mean(), inplace=True)


def label_encoding(df1):
    col_list = df1.select_dtypes(include=['object']).columns.to_list()
    col_list.append('registration_init_time')
    col_list.append('expiration_date')
    labelencoder = LabelEncoder()
    for items in col_list:
        df1[items] = labelencoder.fit_transform(df1[items])

df['source_screen_name'].fillna(df['source_screen_name'].mode().iloc[0], inplace=True)
df['source_system_tab'].fillna(df['source_system_tab'].mode().iloc[0], inplace=True)
df['source_type'].fillna(df['source_type'].mode().iloc[0], inplace=True)
df['artist_name'].fillna(df['artist_name'].mode().iloc[0], inplace=True) #maybe find a better method
df['composer'].fillna(df['composer'].mode().iloc[0], inplace=True) #maybe find a better method
df['lyricist'].fillna(df['lyricist'].mode().iloc[0], inplace=True) #maybe find a better method
df['genre_ids'].fillna(df['genre_ids'].mode().iloc[0], inplace=True)
df['city'].fillna(df['city'].mode().iloc[0], inplace=True)
df['registered_via'].fillna(df['registered_via'].mode().iloc[0], inplace=True)
df['song_length'].fillna(df['song_length'].mean(), inplace=True) #HERE
df['language'].fillna(df['language'].mode().iloc[0], inplace=True)

df = df[df['msno'].notna()]
df = df[df['song_id'].notna()]
df = df[df['target'].notna()]
df = df[df['registration_init_time'].notna()] #label encode dates later
df = df[df['expiration_date'].notna()]

fillNaN_gender(df['gender'])
find_mean(df['bd'])

#labelencoder = LabelEncoder()
#df['registration_init_time'] = labelencoder.fit_transform(df['registration_init_time'])
#df['expiration_date'] = labelencoder.fit_transform(df['expiration_date'])


label_encoding(df)
df= df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
df.to_csv('test_train.csv', index=False)




'''
x_train = df.drop(['target'], axis=1)
y_train = df['target'].values

model = RandomForestClassifier(random_state=42)
model.fit(x_train,y_train)
'''

