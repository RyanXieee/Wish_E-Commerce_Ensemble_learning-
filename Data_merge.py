#!/usr/bin/env python
# coding: utf-8

# In[2]:


import gc
import time
import numpy as np
import pandas as pd
from datetime import datetime


# In[3]:


train = pd.read_csv(r'C:\Users\tianr\Desktop\interview\elo\train.csv')
test =  pd.read_csv(r'C:\Users\tianr\Desktop\interview\elo\test.csv')
merchant = pd.read_csv(r'C:\Users\tianr\Desktop\interview\elo\merchants.csv')
new_transaction = pd.read_csv(r'C:\Users\tianr\Desktop\interview\elo\new_merchant_transactions.csv')
history_transaction = pd.read_csv(r'C:\Users\tianr\Desktop\interview\elo\historical_transactions.csv')


# In[4]:


def change_object_cols(se):
    value = se.unique().tolist()
    value.sort()
    return se.map(pd.Series(range(len(value)), index=value)).values


# In[5]:


se_map = change_object_cols(train['first_active_month'].append(test['first_active_month']).astype(str))
train['first_active_month'] = se_map[:train.shape[0]]
test['first_active_month'] = se_map[train.shape[0]:]


# In[5]:


train.to_csv(r'C:\Users\tianr\Desktop\interview\elo\train_pre.csv', index=False)
test.to_csv(r'C:\Users\tianr\Desktop\interview\elo\test_pre.csv', index=False)


# In[6]:


del train
del test
gc.collect()


# 1.mechants data preprocessing

# In[7]:


# Discrete cols category_cols and continuous cols numeric_cols according to business meaning
category_cols = ['merchant_id', 'merchant_group_id', 'merchant_category_id',
       'subsector_id', 'category_1',
       'most_recent_sales_range', 'most_recent_purchases_range',
       'category_4', 'city_id', 'state_id', 'category_2']
numeric_cols = ['numerical_1', 'numerical_2',
     'avg_sales_lag3', 'avg_purchases_lag3', 'active_months_lag3',
       'avg_sales_lag6', 'avg_purchases_lag6', 'active_months_lag6',
       'avg_sales_lag12', 'avg_purchases_lag12', 'active_months_lag12']


# In[8]:


for col in ['category_1', 'most_recent_sales_range', 'most_recent_purchases_range', 'category_4']:
    merchant[col] = change_object_cols(merchant[col])
    


# In[9]:


merchant[category_cols] = merchant[category_cols].fillna(-1)


# In[10]:


inf_cols = ['avg_purchases_lag3', 'avg_purchases_lag6', 'avg_purchases_lag12']
merchant[inf_cols] = merchant[inf_cols].replace(np.inf, merchant[inf_cols].replace(np.inf, -99).max().max())


# In[11]:


for col in numeric_cols:
    merchant[col] = merchant[col].fillna(merchant[col].mean())


# In[12]:


# Remove duplicate columns from the transaction data
duplicate_cols = ['merchant_id', 'merchant_category_id', 'subsector_id', 'category_1', 'city_id', 'state_id', 'category_2']
merchant = merchant.drop(duplicate_cols[1:], axis=1)
merchant = merchant.loc[merchant['merchant_id'].drop_duplicates().index.tolist()].reset_index(drop=True)


# 2.transcation data preprocessing

# In[13]:


transaction = pd.concat([new_transaction, history_transaction], axis=0, ignore_index=True)
del new_transaction
del history_transaction
gc.collect()


# In[14]:


numeric_cols = [ 'installments', 'month_lag', 'purchase_amount']
category_cols = ['authorized_flag', 'card_id', 'city_id', 'category_1',
       'category_3', 'merchant_category_id', 'merchant_id', 'category_2', 'state_id',
       'subsector_id']
time_cols = ['purchase_date']


# In[15]:


for col in ['authorized_flag', 'category_1', 'category_3']:
    transaction[col] = change_object_cols(transaction[col].fillna(-1).astype(str))
transaction[category_cols] = transaction[category_cols].fillna(-1)
transaction['category_2'] = transaction['category_2'].astype(int)


# In[17]:


#Information extraction for time periods (morning, afternoon, evening, early morning)
transaction['purchase_month'] = transaction['purchase_date'].apply(lambda x:'-'.join(x.split(' ')[0].split('-')[:2]))
transaction['purchase_hour_section'] = transaction['purchase_date'].apply(lambda x: x.split(' ')[1].split(':')[0]).astype(int)//6
transaction['purchase_day'] = transaction['purchase_date'].apply(lambda x: datetime.strptime(x.split(" ")[0], "%Y-%m-%d").weekday())//5                                                                    
del transaction['purchase_date']


# In[18]:


transaction['purchase_month'] = change_object_cols(transaction['purchase_month'].fillna(-1).astype(str))


# 3.data merge

# In[19]:


cols = ['merchant_id', 'most_recent_sales_range', 'most_recent_purchases_range', 'category_4']
transaction = pd.merge(transaction, merchant[cols], how='left', on='merchant_id')

numeric_cols = ['purchase_amount', 'installments']

category_cols = ['authorized_flag', 'city_id', 'category_1',
       'category_3', 'merchant_category_id','month_lag','most_recent_sales_range',
                 'most_recent_purchases_range', 'category_4',
                 'purchase_month', 'purchase_hour_section', 'purchase_day']

id_cols = ['card_id', 'merchant_id']

transaction[cols[1:]] = transaction[cols[1:]].fillna(-1).astype(int)
transaction[category_cols] =transaction[category_cols].fillna(-1).astype(str)


# In[20]:


transaction.to_csv(r'C:\Users\tianr\Desktop\interview\elo\transaction_d_pre.csv', index=False)


# In[ ]:




