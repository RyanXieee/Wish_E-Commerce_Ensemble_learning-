#!/usr/bin/env python
# coding: utf-8

# # Feature Selection with Autocorrelation (Pearson Correlation Coefficients)

# In[3]:


import gc
import time
import numpy as np
import pandas as pd
from datetime import datetime


# In[4]:


train = pd.read_csv(r'C:\Users\tianr\Desktop\interview\elo\train.csv')
test = pd.read_csv(r'C:\Users\tianr\Desktop\interview\elo\test.csv')


# In[ ]:


# Extraction of feature name
features = train.columns.tolist()
features.remove("card_id")
features.remove("target")
featureSelect = features[:]

# Calculate the correlation coefficient
corr = []
for fea in featureSelect:
    corr.append(abs(train[[fea, 'target']].fillna(0).corr().values[0][1]))

# Take feature to modeling
se = pd.Series(corr, index=featureSelect).sort_values(ascending=False)
feature_select = ['card_id'] + se[:300].index.tolist()

# output results
train_RF = train[feature_select + ['target']]
test_RF = test[feature_select]


# 1.GridSearchCV

# In[5]:


from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


# In[6]:


def param_grid_search(train):
    
    # Step 1.Create grid search space
    print('param_grid_search')
    features = train.columns.tolist()
    features.remove("card_id")
    features.remove("target")
    parameter_space = {
    "n_estimators": [79, 80, 81], 
    "min_samples_leaf": [29, 30, 31],
    "min_samples_split": [2, 3],
    "max_depth": [9, 10],
    "max_features": ["auto", 80]
}
    # Step 2.conduct grid search
    print("Tuning hyper-parameters for mse")
    # Instantiating Random Forest Models
    clf = RandomForestRegressor(
        criterion="mse",
        n_jobs=15,
        random_state=22)
    
    grid = GridSearchCV(clf, parameter_space, cv=2, scoring="neg_mean_squared_error")
    grid.fit(train[features].values, train['target'].values)
    
    # Step 3.output result
    print("best_params_:")
    print(grid.best_params_)
    means = grid.cv_results_["mean_test_score"]
    stds = grid.cv_results_["std_test_score"]


# 2.RandomForest

# In[7]:


grid = param_grid_search(train_RF)


# In[8]:


grid.best_estimator_.predict(test[features])


# In[9]:


test['target'] = grid.best_estimator_.predict(test[features])
test[['card_id', 'target']].to_csv("result/submission_randomforest.csv", index=False)

