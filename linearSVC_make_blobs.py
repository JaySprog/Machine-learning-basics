import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
plt.ion()


X,y=make_blobs(random_state=0)

lsvc=LinearSVC(random_state=1,max_iter=100000)
lsvc.fit(X,y)
print(lsvc.coef_)
print(lsvc.intercept_)
shi=np.array([[1,3],[-2,2],[1,2],[0,0]])
print(lsvc.predict(shi))







