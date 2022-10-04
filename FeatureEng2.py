import pandas as pd

train_songs = pd.read_csv('newfeatures_train.csv')
test_songs = pd.read_csv('testtest.csv')



#membership_length
train_songs['membership_length'] = train_songs['registration_init_time'] - train_songs['expiration_date']


#month, year using apply and function
'''
def month(x):
    date = str(x)
    month = date[5]
    return int(month)

def year(x):
    date = str(x)
    year = date[:4]
    return int(year)

train_songs['reg_month'] = train_songs['registration_init_time'].apply(month)
train_songs['reg_year'] = train_songs['registration_init_time'].apply(year)
'''


#month, year using apply and lambda
train_songs['reg_month'] = train_songs['registration_init_time'].apply(lambda x: int(str(x)[5]))
train_songs['reg_year'] = train_songs['registration_init_time'].apply(lambda x: int(str(x)[:4]))
train_songs['exp_month'] = train_songs['expiration_date'].apply(lambda x: int(str(x)[5]))
train_songs['exp_year'] = train_songs['expiration_date'].apply(lambda x: int(str(x)[:4]))





train_songs.to_csv('newnewfeatures_train.csv', index=False)
train_songs.to_csv('newnewfeatures_test.csv', index=False)
