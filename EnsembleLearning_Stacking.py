#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from hyperopt import hp, fmin, tpe
from numpy.random import RandomState
from sklearn.metrics import mean_squared_error


# # LightGBM

# In[10]:


train = pd.read_csv('preprocess/train.csv')
test = pd.read_csv('preprocess/test.csv')


# In[18]:


import lightgbm as lgb


# In[19]:


from hyperopt import hp, fmin, tpe


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


# In[12]:


#call back function
def params_append(params):
    params['feature_pre_filter'] = False
    params['objective'] = 'regression'
    params['metric'] = 'rmse'
    params['bagging_seed'] = 2020
    return params


# In[13]:


def param_hyperopt(train):
   
    # Part 1.Delineate feature names, remove ID columns and label columns
    label = 'target'
    features = train.columns.tolist()
    features.remove('card_id')
    features.remove('target')
    
    # Part 2.Encapsulating training data
    train_data = lgb.Dataset(train[features], train[label])
    

    def hyperopt_objective(params):
        
        params = params_append(params)
        print(params)
        res = lgb.cv(params, train_data, 1000,
                     nfold=2,
                     stratified=False,
                     shuffle=True,
                     metrics='rmse',
                     early_stopping_rounds=20,
                     verbose_eval=False,
                     show_stdv=False,
                     seed=2020)
        return min(res['rmse-mean']) 

   # Hyperparametric space
    params_space = {
        'learning_rate': hp.uniform('learning_rate', 1e-2, 5e-1),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
        'feature_fraction': hp.uniform('feature_fraction', 0.5, 1),
        'num_leaves': hp.choice('num_leaves', list(range(10, 300, 10))),
        'reg_alpha': hp.randint('reg_alpha', 0, 10),
        'reg_lambda': hp.uniform('reg_lambda', 0, 10),
        'bagging_freq': hp.randint('bagging_freq', 1, 10),
        'min_child_samples': hp.choice('min_child_samples', list(range(1, 30, 5)))
    }
    
    # Part 5.TPE hyperparameter search
    params_best = fmin(
        hyperopt_objective,
        space=params_space,
        algo=tpe.suggest,
        max_evals=30,
        rstate=RandomState(2020))
    
    return params_best


# In[14]:


best_clf = param_hyperopt(train_LGBM)


# In[15]:


best_clf


# - Single Model Prediction

# In[16]:


best_clf = params_append(best_clf)
label = 'target'
features = train_LGBM.columns.tolist()
features.remove('card_id')
features.remove('target')
lgb_train = lgb.Dataset(train_LGBM[features], train_LGBM[label])


# In[18]:


bst = lgb.train(best_clf, lgb_train)


# In[19]:


bst.predict(train_LGBM[features])


# In[20]:


np.sqrt(mean_squared_error(train_LGBM[label], bst.predict(train_LGBM[features])))


# In[21]:


test_LGBM['target'] = bst.predict(test_LGBM[features])
test_LGBM[['card_id', 'target']].to_csv("result/submission_LGBM.csv", index=False)


# In[23]:


test_LGBM[['card_id', 'target']].head(5)


# In[85]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy import sparse


# In[11]:


import xgboost as xgb
from sklearn.feature_selection import f_regression
from numpy.random import RandomState
from bayes_opt import BayesianOptimization


# In[13]:


train = pd.read_csv('preprocess/train.csv')
test = pd.read_csv('preprocess/test.csv')


# In[15]:


# callback function
def params_append(params):
    params['objective'] = 'reg:squarederror'
    params['eval_metric'] = 'rmse'
    params["min_child_weight"] = int(params["min_child_weight"])
    params['max_depth'] = int(params['max_depth'])
    return params

def param_beyesian(train):
   
    train_y = pd.read_csv("data/train.csv")['target']
    sample_index = train_y.sample(frac=0.1, random_state=2020).index.tolist()
    train_data = xgb.DMatrix(train.tocsr()[sample_index, :
                             ], train_y.loc[sample_index].values, silent=True)
    
    def xgb_cv(colsample_bytree, subsample, min_child_weight, max_depth,
               reg_alpha, eta,
       
        params = {'objective': 'reg:squarederror',
                  'early_stopping_round': 50,
                  'eval_metric': 'rmse'}
        params['colsample_bytree'] = max(min(colsample_bytree, 1), 0)
        params['subsample'] = max(min(subsample, 1), 0)
        params["min_child_weight"] = int(min_child_weight)
        params['max_depth'] = int(max_depth)
        params['eta'] = float(eta)
        params['reg_alpha'] = max(reg_alpha, 0)
        params['reg_lambda'] = max(reg_lambda, 0)
        print(params)
        cv_result = xgb.cv(params, train_data,
                           num_boost_round=1000,
                           nfold=2, seed=2,
                           stratified=False,
                           shuffle=True,
                           early_stopping_rounds=30,
                           verbose_eval=False)
        return -min(cv_result['test-rmse-mean'])
    
    xgb_bo = BayesianOptimization(
        xgb_cv,
        {'colsample_bytree': (0.5, 1),
         'subsample': (0.5, 1),
         'min_child_weight': (1, 30),
         'max_depth': (5, 12),
         'reg_alpha': (0, 5),
         'eta':(0.02, 0.2),
         'reg_lambda': (0, 5)}
    )
    xgb_bo.maximize(init_points=21, n_iter=5)  # init_points表示初始点，n_iter代表迭代次数（即采样数）
    print(xgb_bo.max['target'], xgb_bo.max['params'])
    return xgb_bo.max['params']

def train_predict(train, test, params):
    """

    :param train:
    :param test:
    :param params:
    :return:
    """
    train_y = pd.read_csv("data/train.csv")['target']
    test_data = xgb.DMatrix(test)

    params = params_append(params)
    kf = KFold(n_splits=5, random_state=2020, shuffle=True)
    prediction_test = 0
    cv_score = []
    prediction_train = pd.Series()
    ESR = 30
    NBR = 10000
    VBE = 50
    for train_part_index, eval_index in kf.split(train, train_y):
        train_part = xgb.DMatrix(train.tocsr()[train_part_index, :],
                                 train_y.loc[train_part_index])
        eval = xgb.DMatrix(train.tocsr()[eval_index, :],
                           train_y.loc[eval_index])
        bst = xgb.train(params, train_part, NBR, [(train_part, 'train'),
                                                          (eval, 'eval')], verbose_eval=VBE,
                        maximize=False, early_stopping_rounds=ESR, )
        prediction_test += bst.predict(test_data)
        eval_pre = bst.predict(eval)
        prediction_train = prediction_train.append(pd.Series(eval_pre, index=eval_index))
        score = np.sqrt(mean_squared_error(train_y.loc[eval_index].values, eval_pre))
        cv_score.append(score)
    print(cv_score, sum(cv_score) / 5)
    pd.Series(prediction_train.sort_index().values).to_csv("preprocess/train_xgboost.csv", index=False)
    pd.Series(prediction_test / 5).to_csv("preprocess/test_xgboost.csv", index=False)
    test = pd.read_csv('data/test.csv')
    test['target'] = prediction_test / 5
    test[['card_id', 'target']].to_csv("result/submission_xgboost.csv", index=False)
    return


# In[16]:


best_clf = param_beyesian(train_x)


# In[17]:


train_predict(train_x, test_x, best_clf)


# In[81]:


oof_rf  = pd.read_csv('./preprocess/train_randomforest.csv')
predictions_rf  = pd.read_csv('./preprocess/test_randomforest.csv')

oof_lgb  = pd.read_csv('./preprocess/train_lightgbm.csv')
predictions_lgb  = pd.read_csv('./preprocess/test_lightgbm.csv')

oof_xgb  = pd.read_csv('./preprocess/train_xgboost.csv')
predictions_xgb  = pd.read_csv('./preprocess/test_xgboost.csv')


# In[101]:


oof_rf.head(5)


# In[103]:


predictions_lgb.head(5)


# In[104]:


oof_rf.shape, oof_lgb.shape


# In[105]:


predictions_rf.shape, predictions_lgb.shape


# In[80]:


def stack_model(oof_1, oof_2, oof_3, predictions_1, predictions_2, predictions_3, y):
   
    train_stack = np.hstack([oof_1, oof_2, oof_3])
    test_stack = np.hstack([predictions_1, predictions_2, predictions_3])
    predictions = np.zeros(test_stack.shape[0])
    from sklearn.model_selection import RepeatedKFold
    folds = RepeatedKFold(n_splits=5, n_repeats=2, random_state=2020)
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_stack, y)):
        print("fold n°{}".format(fold_+1))
        trn_data, trn_y = train_stack[trn_idx], y[trn_idx]
        val_data, val_y = train_stack[val_idx], y[val_idx]
        print("-" * 10 + "Stacking " + str(fold_+1) + "-" * 10)
        # Using Bayesian regression as a model for fusion of results（final model）
        clf = BayesianRidge()
        clf.fit(trn_data, trn_y)
        predictions += clf.predict(test_stack) / (5 * 2)
    
    return predictions


# In[82]:


target = train['target'].values


# In[83]:


predictions_stack  = stack_model(oof_rf, oof_lgb, oof_xgb, 
                                 predictions_rf, predictions_lgb, predictions_xgb, target)


# In[78]:


predictions_stack


# In[79]:


sub_df = pd.read_csv('data/sample_submission.csv')
sub_df["target"] = predictions_stack
sub_df.to_csv('predictions_stack1.csv', index=False)

