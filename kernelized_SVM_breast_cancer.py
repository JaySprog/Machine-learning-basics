import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split as tts
from sklearn.svm import SVC
import matplotlib.pyplot as plt
plt.ion()

a=load_breast_cancer()
X_train,X_test,y_train,y_test=tts(a.data,a.target,random_state=12)
svc=SVC(random_state=9)
svc.fit(X_train,y_train)
print('training_accuracy:',svc.score(X_train,y_train))
print('test_accuracy:',svc.score(X_test,y_test))

plt.plot(X_train.min(axis=0), 'o', label="min")
plt.plot(X_train.max(axis=0), '^', label="max")
plt.legend()
plt.xlabel("Feature index")
plt.ylabel("Feature magnitude")
plt.yscale("log")



min_on_training=X_train.min(axis=0)
range_on_training = (X_train - min_on_training).max(axis=0)
X_train_scaled=(X_train-min_on_training)/range_on_training
X_test_scaled=(X_test-min_on_training)/range_on_training

svc=SVC(C=5,random_state=31)
svc.fit(X_train_scaled, y_train)
print('training_accuracy:',svc.score(X_train_scaled,y_train))
print('test_accuracy:',svc.score(X_test_scaled,y_test))
