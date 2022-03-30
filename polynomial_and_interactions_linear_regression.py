#!/usr/bin/env python
# coding: utf-8

# In[44]:


import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
a=load_boston()

X_train,X_test,y_train,y_test=tts(a.data,a.target,random_state=15)

mms=MinMaxScaler()
mms.fit(X_train)
rescaled_X_train=mms.transform(X_train)
rescaled_X_test=mms.transform(X_test)

lr=LinearRegression()
lr.fit(X_train,y_train)
print('raw_data_test_score (lr):',lr.score(X_test,y_test))

lr=LinearRegression()
lr.fit(rescaled_X_train,y_train)
print('scaled_data_test_score(lr):',lr.score(rescaled_X_test,y_test))


pf=PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
pf.fit(rescaled_X_train)
poly_X_train=pf.transform(rescaled_X_train)
poly_X_test=pf.transform(rescaled_X_test)

lr=LinearRegression()
lr.fit(poly_X_train,y_train)
print('scaled_poly_data_test_score(lr):',lr.score(poly_X_test,y_test))

r=Ridge(random_state=11,alpha=0.0001)
r.fit(poly_X_train,y_train)
print('scaled_poly_data_test_score(ridge):',r.score(poly_X_test,y_test))

rfr=RandomForestRegressor(random_state=2)
rfr.fit(X_train,y_train)
print('raw_data_test_score(random_forest):',rfr.score(X_test,y_test))


# In[ ]:




