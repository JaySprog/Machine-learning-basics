import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
plt.ion()

a=pd.read_csv(r"C:\users\jaysu\Desktop\DS&ML\priority_1\ram_prices.csv")

plt.semilogy(a.date,a.price,label='Real Data')
plt.xlabel('Year')
plt.ylabel('Price in $ per Megabyte')

data=np.array(a)
data_train=np.array(a[a.date<2000])
data_test=np.array(a[a.date>=2000])

X_train=(np.array(data_train[:,0])).reshape(-1,1)
y_train=np.log((np.array(data_train[:,1])).reshape(-1,1))
X_test=(np.array(data_test[:,0])).reshape(-1,1)
y_test=np.log((np.array(data_test[:,1])).reshape(-1,1))

dtr=DecisionTreeRegressor(random_state=45)
dtr.fit(X_train,y_train)
dtr_predict=dtr.predict(data[:,0].reshape(-1,1))
plt.plot(a.date,np.exp(dtr_predict),linestyle='dashed',
         label='Decision Tree Regressor Prediction')

lr=LinearRegression()
lr.fit(X_train,y_train)
lr_predict=lr.predict(data[:,0].reshape(-1,1))
plt.plot(a.date,np.exp(lr_predict),linestyle='dashed',label='Linear Regressor Prediction')


plt.legend()








