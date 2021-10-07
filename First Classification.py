#!/usr/bin/env python
# coding: utf-8

# My First Classification Model

#             by Bridget Moh Yi Min

# This is done following <i>Data Professor YouTube channel, http://youtube.com/dataprofessor </i>. If you are interested , do check him out!

# # 1) Import Libraries
# 

# In[1]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


# # 2) Load wine dataset

# In[2]:


wine = datasets.load_wine()


# # 3) Input features

# The ***wine*** data set contains 13 input features and 1 output variable (the class label).

# In[3]:


print(wine.feature_names)


# In[4]:


print(wine.target_names)


# # 4) Glimpse of the data

# In[6]:


wine.data


# In[7]:


wine.target


# # 5) Assigning input and output data

# In[8]:


X= wine.data
Y= wine.target


# 5.1 Looking at data dimension

# In[9]:


X.shape


# In[10]:


Y.shape


# # 6) Build classification model using random forest

# In[11]:


clf = RandomForestClassifier()


# In[12]:


clf.fit(X, Y)


# # 7) Feature Importance

# In[14]:


print(clf.feature_importances_)


# # 8) Making predictions

# In[15]:


X[0]


# In[18]:


print(clf.predict([[1.423e+01, 1.710e+00, 2.430e+00, 1.560e+01, 1.270e+02, 2.800e+00,
       3.060e+00, 2.800e-01, 2.290e+00, 5.640e+00, 1.040e+00, 3.920e+00,
       1.065e+03]]))


# In[23]:


print(clf.predict(X[[0]]))


# In[24]:


print(clf.predict_proba(X[[0]]))


# In[22]:


clf.fit(wine.data, wine.target_names[wine.target])


# # 9) Data split (80/20 ratio)

# In[26]:


X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.2)


# In[28]:


X_train.shape


# In[29]:


Y_train.shape


# In[30]:


X_test.shape


# In[31]:


Y_test.shape


# # 10) Rebuild the RandomForest model 

# In[32]:


clf.fit(X_train,Y_train)


# a) Perform prediction on single sample from the data set

# In[33]:


print(clf.predict([[1.423e+01, 1.710e+00, 2.430e+00, 1.560e+01, 1.270e+02, 2.800e+00,
       3.060e+00, 2.800e-01, 2.290e+00, 5.640e+00, 1.040e+00, 3.920e+00,
       1.065e+03]]))


# In[34]:


print(clf.predict_proba([[1.423e+01, 1.710e+00, 2.430e+00, 1.560e+01, 1.270e+02, 2.800e+00,
       3.060e+00, 2.800e-01, 2.290e+00, 5.640e+00, 1.040e+00, 3.920e+00,
       1.065e+03]]))


# b) Performs prediction on the test set

# *Predicted wine class labels*

# In[35]:


print(clf.predict(X_test))


# *Actual wine class labels*

# In[36]:


print(Y_test)


# # 11) Model Performance

# In[37]:


print(clf.score(X_test, Y_test))

