import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression 
plt.ion()

a=load_breast_cancer()

X_train,X_test,y_train,y_test=tts(a.data,a.target,random_state=0)

bc_loreg=LogisticRegression(C=0.001,max_iter=100000)
bc_loreg.fit(X_train,y_train)
coef=bc_loreg.coef_
plt.plot(coef.T,'o', label='C=0.001')

bc_loreg=LogisticRegression(C=1,max_iter=100000)
bc_loreg.fit(X_train,y_train)
coef=bc_loreg.coef_
plt.plot(coef.T,'s',label='C=1')

bc_loreg=LogisticRegression(C=100,max_iter=100000)
bc_loreg.fit(X_train,y_train)
coef=bc_loreg.coef_
plt.plot(coef.T,'^',label='C=100')

plt.xticks(np.arange(30), a.feature_names,rotation=90)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.grid()
plt.legend()
plt.subplots_adjust(bottom=0.445)


