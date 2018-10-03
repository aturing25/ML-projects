
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[3]:


train=pd.read_csv("C:\\Users\\HIMANSHU\\Desktop\\Data Science\\ANALYTICS VIDYA\\CodeFest\\cf_train.csv")
test=pd.read_csv("C:\\Users\\HIMANSHU\\Desktop\\Data Science\\ANALYTICS VIDYA\\CodeFest\\cf_test.csv")


# In[4]:


train.head()


# In[5]:


from sklearn.preprocessing import LabelEncoder
model=LabelEncoder()
train["Tag"]=model.fit_transform(train["Tag"].astype(str))
test["Tag"]=model.fit_transform(test["Tag"].astype(str))


# In[6]:


corr=train.corr()
print(corr["Upvotes"].sort_values(ascending=False))


# In[7]:


ID_train=train["ID"]
train.drop("ID",axis=1, inplace=True)
train.drop("Tag",axis=1, inplace=True)
test.drop("Tag",axis=1, inplace=True)


# In[8]:


Y_train=train["Upvotes"]
X_train=train.drop("Upvotes", axis=1)


# In[18]:


from sklearn.ensemble import RandomForestRegressor as RF
my_model=RF(n_estimators=100,min_samples_split=10)
my_model.fit(X_train,Y_train)


# In[19]:


ID_test=test["ID"]
X_test=test.drop("ID",axis=1)


# In[20]:


predict=my_model.predict(X_test)


# In[21]:


predict=list(map(int,predict))


# In[22]:


predict=list(map(int,predict))


# In[24]:


my_submission = pd.DataFrame({'ID': ID_test,'Upvotes':predict})
my_submission=my_submission[["ID","Upvotes"]]

my_submission.to_csv("C:\\Users\\HIMANSHU\\Desktop\\Data Science\\ANALYTICS VIDYA\\CodeFest\\sol6.csv", index=False)

