import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
plt.ion()

a=load_breast_cancer()
X_train,X_test,y_train,y_test=tts(a.data,a.target,random_state=45)
gbc=GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,max_depth=2,random_state=37)
gbc.fit(X_train,y_train)
#print('training_accuracy:',gbc.score(X_train,y_train))
#print('test_accuracy:',gbc.score(X_test,y_test))

#print(gbc.predict_proba(X_test).shape)
#print(gbc.decision_function(X_test))
#print(gbc.predict(X_test))

a=load_iris()
X_train,X_test,y_train,y_test=tts(a.data,a.target,random_state=5)
gbc=GradientBoostingClassifier(random_state=3)
gbc.fit(X_train,y_train)
#print('training_accuracy:',gbc.score(X_train,y_train))
#print('test_accuracy:',gbc.score(X_test,y_test))

print((gbc.predict_proba(X_test)))
print(gbc.decision_function(X_test))
print(gbc.predict(X_test))
