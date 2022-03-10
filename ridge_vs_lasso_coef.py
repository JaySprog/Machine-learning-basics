import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge as r
from sklearn.linear_model import Lasso as lasso
plt.ion()

lb=load_boston()
X_train,X_test,y_train,y_test=tts(lb.data,lb.target,random_state=2)

a1=r(alpha=0.1)
a1.fit(X_train,y_train)
a2=lasso(alpha=1,max_iter=100000)
a2.fit(X_train,y_train)
a3=lasso(alpha=0.01,max_iter=100000)
a3.fit(X_train,y_train)
a4=lasso(alpha=0.0001,max_iter=100000)
a4.fit(X_train,y_train)

plt.plot(a1.coef_,'o', label='ridge_alpha=0.1')
plt.plot(a2.coef_,'s', label='lasso_alpha=1')
plt.plot(a3.coef_,'^',label='lasso_alpha=0.01')
plt.plot(a4.coef_,'+', label='lasso_alpha=0.0001')
plt.grid()
plt.legend()
