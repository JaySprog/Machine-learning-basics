import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.model_selection import train_test_split as tts
plt.ion()

a=load_breast_cancer()
X_train,X_test,y_train,y_test=tts(a.data,a.target,random_state=1)

score_array_train=[]
score_array_test=[]
k_value=[]
for k in range(1,21):
    shi=knc(n_neighbors=k)
    shi.fit(X_train,y_train)
    score_array_test.append(shi.score(X_test,y_test))
    score_array_train.append(shi.score(X_train,y_train))
    k_value.append(k)
                             

plt.plot(k_value,score_array_train,label="training accuracy")
plt.plot(k_value,score_array_test,label="test accuracy")
plt.grid()
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
