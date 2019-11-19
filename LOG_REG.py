#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv(r'C:\Users\BISWA\Downloads\Social Networks Add.csv')


# In[3]:


data.columns


# In[4]:


data.info()


# In[5]:


data.head()


# In[6]:


data.describe()


# In[7]:


x = data.iloc[:,[2,3]]
y = data.iloc[:,[4]]


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.25, random_state =0)


# In[10]:


from sklearn.linear_model import LogisticRegression


# In[11]:


model = LogisticRegression()


# In[12]:


from sklearn.preprocessing import StandardScaler


# In[13]:


sc = StandardScaler()


# In[14]:


X_train = sc.fit_transform(x_train)


# In[15]:


X_test = sc.transform(x_test)


# In[16]:


model.fit(X_train,y_train)


# In[17]:


y_pre = model.predict(X_test)


# In[18]:


y_pre


# In[19]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[20]:


confusion_matrix(y_test,y_pre)


# In[24]:


print(len(y_test))


# In[21]:


print(accuracy_score(y_test,y_pre))


# In[22]:


sns.heatmap(pd.DataFrame(confusion_matrix(y_test,y_pre)))
plt.show()

