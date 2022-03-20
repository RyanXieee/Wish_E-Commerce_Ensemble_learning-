#!/usr/bin/env python
# coding: utf-8

# In[2]:


import gc
import time
import numpy as np
import pandas as pd
from datetime import datetime


# In[3]:


transaction = pd.read_csv(r'C:\Users\tianr\Desktop\interview\elo\transaction_d_pre.csv')


# In[7]:


numeric_cols = ['authorized_flag',  'category_1', 'installments',
       'category_3',  'month_lag','purchase_month','purchase_day',
       'purchase_amount', 'category_2', 
       'purchase_month', 'purchase_hour_section', 'purchase_day',
       'most_recent_sales_range', 'most_recent_purchases_range', 'category_4']
categorical_cols = ['city_id', 'merchant_category_id', 'merchant_id', 'state_id', 'subsector_id']


# In[ ]:


# Create dictionary
aggs = {}
for col in numeric_cols:
    aggs[col] = ['nunique', 'mean', 'min', 'max','var','skew', 'sum']
for col in categorical_cols:
    aggs[col] = ['nunique']    
aggs['card_id'] = ['size', 'count']
cols = ['card_id']

# Statistics calculation with groupby
for key in aggs.keys():
    cols.extend([key+'_'+stat for stat in aggs[key]])

df = transaction[transaction['month_lag']<0].groupby('card_id').agg(aggs).reset_index()
df.columns = cols[:1] + [co+'_hist' for co in cols[1:]]

df2 = transaction[transaction['month_lag']>=0].groupby('card_id').agg(aggs).reset_index()
df2.columns = cols[:1] + [co+'_new' for co in cols[1:]]
df = pd.merge(df, df2, how='left',on='card_id')

df2 = transaction.groupby('card_id').agg(aggs).reset_index()
df2.columns = cols
df = pd.merge(df, df2, how='left',on='card_id')
del transaction
gc.collect()

# Generate training set and test set
train = pd.merge(train, df, how='left', on='card_id')
test =  pd.merge(test, df, how='left', on='card_id')
del df
train.to_csv(r'C:\Users\tianr\Desktop\interview\elo\train_groupby.csv', index=False)
test.to_csv(r'C:\Users\tianr\Desktop\interview\elo\test_groupby.csv', index=False)

gc.collect()


# In[ ]:


train = pd.merge(train_dict, train_groupby, how='left', on='card_id').fillna(0)
test = pd.merge(test_dict, test_groupby, how='left', on='card_id').fillna(0)

