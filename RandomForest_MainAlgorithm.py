import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import time


train_songs = pd.read_csv('newfeatures_train.csv',nrows=1000000)
test_songs = pd.read_csv('newfeatures_test.csv')
x_train = train_songs.drop(['target','language'], axis=1)
y_train = train_songs['target'].values

start_time = time.time()

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


model = RandomForestClassifier(max_depth=110, max_features='sqrt', n_estimators=1100, bootstrap = True, min_samples_split = 2, min_samples_leaf = 1)

#RandomForestClassifier(n_estimators = 1000, min_samples_split = 2, min_samples_leaf = 1, max_features = 'sqrt',max_depth = 110, bootstrap=True)

model.fit(x_train,y_train) #doesn't accept strings need encoding


#feature importance
'''
importances = model.feature_importances_
sorted_idx = model.feature_importances_.argsort()
train_songs = train_songs.drop(['target'], axis=1)
df_list = list(train_songs.columns)
plt.barh(df_list,importances)
plt.show()
'''

#RandomizedSearchCV
'''
rf_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(x_train, y_train)
best_random = rf_random.best_params_
print(best_random) #{'n_estimators': 1000, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 110, 'bootstrap': True}
'''

# Grid search
'''
param_grid = {
    'bootstrap': [True],
    'max_depth': [100, 110, 120],
    'max_features': ['sqrt'],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [1, 2, 3],
    'n_estimators': [900,1000,1100]
}


rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                          cv = 3, n_jobs = -1, verbose = 2)
grid_search.fit(x_train, y_train)
best_grid = grid_search.best_estimator_ #RandomForestClassifier(max_depth=110, max_features='sqrt', n_estimators=1100)
print(best_grid)
'''


X_test = test_songs.drop(columns=['id','language'], axis=1)
ids = test_songs['id'].values

del test_songs

#pred_test = grid_search.predict(X_test)
#pred_test = rf_random.predict(X_test)
pred_test = model.predict(X_test)

#submission file
sub = pd.DataFrame()
sub['id'] = ids.astype('int')
sub['target'] = pred_test
sub.to_csv('submission.csv', index = False , float_format ='%.5f')

