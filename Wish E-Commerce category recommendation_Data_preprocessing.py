#!/usr/bin/env python
# coding: utf-8

# Background:Wish is one of the largest ecommerce marketplaces in the world.In order to provide customers with precise promotions or discounts, we used data on some clothing transactions to build machine learning models to understand the most important aspects and preferences in their customers'lifecycle.We conducted Ensemble method to predict loyalty score of customers of the platform.

#  In the hypercompetitive market,loyal customers convert friends and family into customers just through word of mouth, and they contribute higher revenues for brands

# Metrics To Measure Customer Loyalty:
#     Net Promoter Score:Promoters. These are customers with a score of 9 or 10. They are your biggest fans, and are likely to not only buy from you again, but also recommend you to others.
# 
# Passives. Passives, or customers with a score of 7 to 8, may be satisfied but they lack the enthusiasm to recommend you to others. They wouldn’t be opposed to offers from your competitors.
# 
# Detractors. If a customer rates the service with a score of 6 or lower, they are considered ‘detractors’. They are dissatisfied customers who can damage your brand by communicating their negative experience to others, thereby impeding your growth.
# Repurchase Ratio
# Upsell Ratio
# Customer Lifetime Value
# The Customer Lifetime Value (CLV) is an understanding of the total revenue attributed by the entire relationship (including future purchases) with a customer. 

# # Exploratory Data Analysis

# In[4]:


import os
import numpy as np
import pandas as pd


# In[5]:


pd.read_excel(r'C:\Users\tianr\Desktop\interview\elo\Data_Dictionary.xlsx',header=2, sheet_name='train')


# In[6]:


import gc


# In[7]:


train=pd.read_csv(r'C:\Users\tianr\Desktop\interview\elo\train.csv')
test=pd.read_csv(r'C:\Users\tianr\Desktop\interview\elo\test.csv')


# In[8]:


(train.shape,test.shape)


# In[9]:


train.head(5)


# In[10]:


train.info()


# 1.Data quality analysis

# In[11]:


# Check the training set id for duplicates
train['card_id'].nunique() == train.shape[0]


# In[12]:


test['card_id'].nunique() == test.shape[0]


# In[14]:


#Find missing values by column and summarize
train.isnull().sum()


# In[15]:


test.isnull().sum()


# In[16]:


statistics = train['target'].describe()
statistics


# In[17]:


# Check outlier
import seaborn as sns
import matplotlib.pyplot as plt


# In[18]:


sns.set()
sns.histplot(train['target'], kde=True)


# In[19]:


(train['target'] < -30).sum()


# In[20]:


statistics.loc['mean'] - 3 * statistics.loc['std']


# In[23]:


#Consistency Analysis
features = ['first_active_month','feature_1','feature_2','feature_3']
train_count = train.shape[0]
test_count = test.shape[0]


# In[24]:


for feature in features:
    (train[feature].value_counts().sort_index()/train_count).plot()
    (test[feature].value_counts().sort_index()/test_count).plot()
    plt.legend(['train','test'])
    plt.xlabel(feature)
    plt.ylabel('ratio')
    plt.show()


# # Data Exploration and Data Cleaning

# 1.Merchant Data Interpretation and Exploration

# In[25]:


merchant = pd.read_csv(r'C:\Users\tianr\Desktop\interview\elo\merchants.csv',header=0)


# In[26]:


merchant.head(5)


# In[27]:


df = pd.read_excel(r'C:\Users\tianr\Desktop\interview\elo\Data_Dictionary.xlsx', header=2, sheet_name='merchant')


# In[28]:


df


# In[29]:


merchant.isnull().sum()


# In[30]:


category_cols = ['merchant_id', 'merchant_group_id', 'merchant_category_id',
       'subsector_id', 'category_1',
       'most_recent_sales_range', 'most_recent_purchases_range',
       'category_4', 'city_id', 'state_id', 'category_2']
numeric_cols = ['numerical_1', 'numerical_2',
     'avg_sales_lag3', 'avg_purchases_lag3', 'active_months_lag3',
       'avg_sales_lag6', 'avg_purchases_lag6', 'active_months_lag6',
       'avg_sales_lag12', 'avg_purchases_lag12', 'active_months_lag12']


# In[31]:


merchant[category_cols].nunique()


# In[32]:


merchant[category_cols].dtypes


# In[33]:


merchant['category_2'].unique()


# In[34]:


merchant['category_2']=merchant['category_2'].fillna(-1)


# In[35]:


merchant[category_cols].isnull().sum()


# In[36]:


def change_object_cols(se):
    value = se.unique().tolist()
    value.sort()
    return se.map(pd.Series(range(len(value)), index=value)).values


# In[37]:


merchant['category_1']


# In[38]:


change_object_cols(merchant['category_1'])


# In[39]:


for col in ['category_1', 'most_recent_sales_range', 'most_recent_purchases_range', 'category_4']:
    merchant[col] = change_object_cols(merchant[col])


# In[40]:


#continuous variables exploration


# In[41]:


merchant[numeric_cols].dtypes


# In[42]:


merchant[numeric_cols].isnull().sum()


# In[43]:


merchant[numeric_cols].describe()


# In[44]:


#Infinite values
inf_cols = ['avg_purchases_lag3', 'avg_purchases_lag6', 'avg_purchases_lag12']
merchant[inf_cols] = merchant[inf_cols].replace(np.inf, merchant[inf_cols].replace(np.inf, -99).max().max())


# In[45]:


merchant[numeric_cols].describe()


# In[46]:


# missing value
for col in numeric_cols:
    merchant[col] = merchant[col].fillna(merchant[col].mean())


# In[47]:


merchant[numeric_cols].describe()


# 2.Transaction Data Interpretation and Exploration

# In[49]:


history_transaction = pd.read_csv(r'C:\Users\tianr\Desktop\interview\elo\historical_transactions.csv',header = 0)


# In[50]:


history_transaction.info()


# In[51]:


pd.read_excel(r'C:\Users\tianr\Desktop\interview\elo\Data_Dictionary.xlsx', header=2, sheet_name='history')


# In[52]:


new_transaction =pd.read_csv(r'C:\Users\tianr\Desktop\interview\elo\new_merchant_transactions.csv',header=0)


# In[53]:


new_transaction.head(5)


# In[54]:


# check duplicates between merchant data and transactions


# In[55]:


duplicate_cols = []

for col in merchant.columns:
    if col in new_transaction.columns:
        duplicate_cols.append(col)
        
print(duplicate_cols)


# In[56]:


new_transaction[duplicate_cols].drop_duplicates().shape


# In[57]:


new_transaction['merchant_id'].nunique()


# In[58]:


# Data preprocessing
numeric_cols = ['installments', 'month_lag', 'purchase_amount']
category_cols = ['authorized_flag', 'card_id', 'city_id', 'category_1',
       'category_3', 'merchant_category_id', 'merchant_id', 'category_2', 'state_id',
       'subsector_id']
time_cols = ['purchase_date']

assert len(numeric_cols) + len(category_cols) + len(time_cols) == new_transaction.shape[1]


# In[59]:


new_transaction[category_cols].isnull().sum()


# In[60]:


for col in ['authorized_flag', 'category_1', 'category_3']:
    new_transaction[col] = change_object_cols(new_transaction[col].fillna(-1).astype(str))
    
new_transaction[category_cols] = new_transaction[category_cols].fillna(-1)


# In[61]:


new_transaction[category_cols].dtypes

