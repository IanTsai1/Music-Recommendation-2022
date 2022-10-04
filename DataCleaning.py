import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import missingno as msno
from random import randint, choice
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
import time


#train_songs = pd.read_csv('fix_train.csv')
#test_songs = pd.read_csv('fix_test.csv')


#data reading and merging

songs_df = pd.read_csv("./data/songs.csv")
members_df = pd.read_csv("./data/members.csv")
train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")


#merging train data, uses MSNO to find the corresponding values
t_m = pd.merge(train_df, members_df, on='msno', how='inner') #inner matches correspoding on value
train_songs = pd.merge(t_m,songs_df,on='song_id',how='outer')
train_songs = train_songs.replace('?', np.nan) #replace ? mark with nan
del t_m #deletes all object listed

#merging test data

t_m = pd.merge(test_df, members_df, on='msno', how='inner') #inner matches correspoding on value
test_songs = pd.merge(t_m,songs_df,on='song_id',how='outer')
test_songs = test_songs.replace('?',np.nan) #replace ? mark with nan
del songs_df, members_df, train_df, test_df, t_m



#view NaN rows
fig = msno.matrix(test_songs)
fig_copy = fig.get_figure()
fig_copy.savefig('plot1.png', bbox_inches = 'tight') #white lines = missing value, pic

#shows columns that has nan values
nan_cols = [i for i in train_songs.columns if train_songs[i].isnull().any()]  
print(nan_cols)


#functions

#imputing missing data
def fillNaN_gender(df):
    gender = ['male','female']
    for i in range(len(df)):
        if  pd.isnull(df[i])== True:
            df.at[i] = gender[random.randint(0, 1)]

def find_mean(df):
    df.replace(0,np.nan,inplace=True) #removes 0 values so it won't affect mean
    df.fillna(df.mean(), inplace=True)

#string encoding
def label_encoding(df1):
    col_list = df1.select_dtypes(include=['object']).columns.to_list()
    col_list.append('registration_init_time')
    col_list.append('expiration_date')
    labelencoder = LabelEncoder()
    for items in col_list:
        df1[items] = labelencoder.fit_transform(df1[items])



#SimpleImputer is equivalent to fillna()
imp_frequent= SimpleImputer(missing_values='NaN', strategy='most_frequent') #try 'Nan' if doesn't work
imp_frequent.fit(train_songs['artist_name'])
train_songs['aritst_name'] = imp_frequent.transform(train_songs['artist_name'])



#train songs data cleaning
train_songs['source_screen_name'].fillna(train_songs['source_screen_name'].mode().iloc[0], inplace=True)
train_songs['source_system_tab'].fillna(train_songs['source_system_tab'].mode().iloc[0], inplace=True)
train_songs['source_type'].fillna(train_songs['source_type'].mode().iloc[0], inplace=True)
train_songs['artist_name'].fillna(train_songs['artist_name'].mode().iloc[0], inplace=True) #maybe find a better method
train_songs['composer'].fillna(train_songs['composer'].mode().iloc[0], inplace=True) #maybe find a better method
train_songs['lyricist'].fillna(train_songs['lyricist'].mode().iloc[0], inplace=True) #maybe find a better method
train_songs['genre_ids'].fillna(train_songs['genre_ids'].mode().iloc[0], inplace=True)
train_songs['city'].fillna(train_songs['city'].mode().iloc[0], inplace=True)
train_songs['registered_via'].fillna(train_songs['registered_via'].mode().iloc[0], inplace=True)
train_songs['song_length'].fillna(train_songs['song_length'].mean(), inplace=True) #HERE
train_songs['language'].fillna(train_songs['language'].mode().iloc[0], inplace=True)

train_songs = train_songs[train_songs['msno'].notna()]
train_songs = train_songs[train_songs['song_id'].notna()]
train_songs = train_songs[train_songs['target'].notna()]
train_songs = train_songs[train_songs['registration_init_time'].notna()] #label encode dates later
train_songs = train_songs[train_songs['expiration_date'].notna()]

fillNaN_gender(train_songs['gender'])
find_mean(train_songs['bd'])
train_songs = train_songs[~(test_songs['bd']<=10)]

#feature engineering
train_songs['reg_month'] = train_songs['registration_init_time'].apply(lambda x: int(str(x)[4:6]))
train_songs['reg_year'] = train_songs['registration_init_time'].apply(lambda x: int(str(x)[:4]))
train_songs['exp_month'] = train_songs['expiration_date'].apply(lambda x: int(str(x)[4:6]))
train_songs['exp_year'] = train_songs['expiration_date'].apply(lambda x: int(str(x)[:4]))


#test songs data cleaning

test_songs['source_screen_name'].fillna(test_songs['source_screen_name'].mode().iloc[0], inplace=True)
test_songs['source_system_tab'].fillna(test_songs['source_system_tab'].mode().iloc[0], inplace=True)
test_songs['source_type'].fillna(test_songs['source_type'].mode().iloc[0], inplace=True)
test_songs['artist_name'].fillna(test_songs['artist_name'].mode().iloc[0], inplace=True) #maybe find a better method
test_songs['composer'].fillna(test_songs['composer'].mode().iloc[0], inplace=True) #maybe find a better method
test_songs['lyricist'].fillna(test_songs['lyricist'].mode().iloc[0], inplace=True) #maybe find a better method
test_songs['genre_ids'].fillna(test_songs['genre_ids'].mode().iloc[0], inplace=True)
test_songs['city'].fillna(test_songs['city'].mode().iloc[0], inplace=True)
test_songs['registered_via'].fillna(test_songs['registered_via'].mode().iloc[0], inplace=True)
test_songs['song_length'].fillna(test_songs['song_length'].mean(), inplace=True) #HERE
test_songs['language'].fillna(test_songs['language'].mode().iloc[0], inplace=True)

test_songs = test_songs[test_songs['msno'].notna()]
test_songs = test_songs[test_songs['song_id'].notna()]
test_songs = test_songs[test_songs['id'].notna()]
test_songs = test_songs[test_songs['registration_init_time'].notna()] #label encode dates later
test_songs = test_songs[test_songs['expiration_date'].notna()]

fillNaN_gender(test_songs['gender'])
find_mean(test_songs['bd'])
test_songs = test_songs[~(test_songs['bd']<=10)]

test_songs['reg_month'] = test_songs['registration_init_time'].apply(lambda x: int(str(x)[4:6]))
test_songs['reg_year'] = test_songs['registration_init_time'].apply(lambda x: int(str(x)[:4]))
test_songs['exp_month'] = test_songs['expiration_date'].apply(lambda x: int(str(x)[4:6]))
test_songs['exp_year'] = test_songs['expiration_date'].apply(lambda x: int(str(x)[:4]))

#lable encoding
label_encoding(train_songs)
train_songs['membership_length'] = train_songs['registration_init_time'] - train_songs['expiration_date']
train_songs = train_songs[(np.abs(stats.zscore(train_songs)) < 3).all(axis=1)] #remove outliers
train_songs.to_csv('newfeatures_train.csv', index=False)

#more data cleaning for cells with multiple values
train_songs['genre_ids'] = train_songs['genre_ids'].str.split('|').str[0] #get the first genre if there's multiple
train_songs['composer'] = train_songs['composer'].  str.split('|').str[0]
train_songs['lyricist'] = train_songs['lyricist'].str.split('|').str[0]
test_songs['composer'] = test_songs['composer'].str.split('|').str[0]
test_songs['lyricist'] = test_songs['lyricist'].str.split('|').str[0]

#remove ages that's less than 10
test_songs = test_songs[~(test_songs['bd']<=10)]
train_songs = train_songs[~(train_songs['bd']<=10)]



label_encoding(test_songs)
test_songs['membership_length'] = abs(test_songs['registration_init_time'] - test_songs['expiration_date'])
test_songs = test_songs[(np.abs(stats.zscore(test_songs)) < 3).all(axis=1)] #remove outliers
test_songs.to_csv('newfeatures_test.csv', index=False)


















