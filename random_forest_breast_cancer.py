import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
plt.ion()

a=load_breast_cancer()
X_train,X_test,y_train,y_test=tts(a.data,a.target,random_state=21)
rfc=RandomForestClassifier(n_estimators=100,random_state=37)
rfc.fit(X_train,y_train)
print('training_accuracy:',rfc.score(X_train,y_train))
print('test_accuracy:',rfc.score(X_test,y_test))

plt.barh(np.arange(a.data.shape[1]),rfc.feature_importances_, align='center')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.yticks(ticks=np.arange(a.data.shape[1]), labels=a.feature_names)
plt.subplots_adjust(left=0.345)
