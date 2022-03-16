import pandas as pd
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
plt.ion()

a=make_moons(noise=0.3,random_state=3)
n_samples=(a[0].shape)[0]
for i in range (n_samples):
    if a[1][i]==0:
        plt.scatter(a[0][i][0],a[0][i][1],color='green')
    if a[1][i]==1:
        plt.scatter(a[0][i][0],a[0][i][1],color='orange')
        
X_train,X_test,y_train,y_test=tts(a[0],a[1],stratify=a[1],random_state=10)
rfc=RandomForestClassifier(n_estimators=100,random_state=25)
rfc.fit(X_train,y_train)
print(rfc.score(X_train,y_train))
print(rfc.score(X_test,y_test))

plt.xlabel('first_feature')
plt.ylabel('second_feature')
    

