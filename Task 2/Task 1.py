#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
import pickle


# In[2]:


df = pd.read_csv("AirQuality.csv",sep = ';')


# In[3]:


df.head()


# In[4]:


df.info()
print("----------------------------------------------------------------")
print("Total Null values per column :-\n",df.isna().sum())
print("----------------------------------------------------------------")
print("Shape of the dataset = ",df.shape)


# In[5]:


#Since all the values of these columns are missing, simply drop them
df.drop(['Unnamed: 15', 'Unnamed: 16'],axis= 1, inplace = True)
df.head()


# In[6]:


df.tail(115)


# In[7]:


df.dropna(inplace = True)


# In[8]:


df.head()


# In[9]:


def remove_comma(col):
    df[col] = df[col].str.replace(',','.')
cols = ['CO(GT)','C6H6(GT)','T','RH','AH']
for col in cols:
    remove_comma(col)


# In[10]:


df.head()


# In[11]:


df['Time'].unique()


# In[12]:


df['Time'] = df['Time'].apply(lambda x: int(x.split('.')[0]))
df['Month'] = df['Date'].apply(lambda x: int(x.split('/')[1]))
df['Day'] = df['Date'].apply(lambda x: int(x.split('/')[0]))


# In[13]:


df.drop("Date",axis = 1,inplace = True)


# In[14]:


df.head()


# In[15]:


df.dtypes


# In[16]:


cols = ['CO(GT)','C6H6(GT)','T','RH','AH']
for col in cols:
    df[col] = df[col].astype(float)


# In[17]:


df.head()


# In[18]:


df.describe()


# In[19]:


df.dtypes


# In[20]:


plt.figure(figsize = (18,8))
df.boxplot()


# In[21]:


mean_by_time = df.groupby(['Time']).mean()
for col in df.columns[1:]:
    Q1 = df[col].sort_values().quantile(0.25)
    Q3 = df[col].sort_values().quantile(0.75)
    IQR = Q3-Q1
    outliers = df[(df[col]<(Q1-1.5*IQR)) | (df[col]>(Q3+1.5*IQR))]
    for ind in outliers.index:
        time = df['Time'][ind]
        df[col][ind] = mean_by_time[col][time]


# In[22]:


plt.figure(figsize = (18,8))
df.boxplot()


# In[26]:


df.drop(['Time','Month','Day'],axis = 1,inplace = True)


# In[27]:


df.head()


# In[28]:


plt.figure(figsize = (13,10))
sns.heatmap(df.corr(),annot = True,cmap = 'viridis')


# In[29]:


#We can see PT08.S1(CO),C6H6(GT) and PT08.S2(NMHC) can be dependent variables


# In[30]:


cols = ['PT08.S1(CO)','C6H6(GT)','PT08.S2(NMHC)']

col_scores = {}

for col in cols:
    X_col = df.drop(col,axis = 1)
    Y_col = df[col]

    X_col_train,X_col_test,y_col_train,y_col_test = train_test_split(X_col,Y_col,test_size = 0.2)

    lr_col = LinearRegression()
    lr_col.fit(X_col_train,y_col_train)
    col_scores[col] = lr_col.score(X_col_test,y_col_test)
    
scores = pd.DataFrame.from_dict(col_scores,orient = 'index',columns = ['Scores'])
scores


# In[31]:


#NMHC provides best result and is since the target variable



# In[32]:


X = df.drop('PT08.S2(NMHC)',axis = 1)
Y = df['PT08.S2(NMHC)']

print(X.shape)
display(X)

lr = LinearRegression()
lr.fit(X,Y)


# In[ ]:
pickle.dump(lr,open('model.pkl','wb'))


# %%
