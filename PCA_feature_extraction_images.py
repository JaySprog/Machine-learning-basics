import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split as tts
plt.ion()

a=fetch_lfw_people(min_faces_per_person=20)
#print(a.target.shape)
#print(a.target_names.shape)
#print(a.target_names)
#print(np.bincount(a.target))

X_train,X_test,y_train,y_test=tts(a.data,a.target,stratify=a.target,random_state=1)
#knc=KNeighborsClassifier(n_neighbors=1)
#knc.fit(X_train, y_train)
#print(knc.score(X_test, y_test))

#pca = PCA(n_components=1000, whiten=True, random_state=13)
#pca.fit(X_train)
#X_train_pca = pca.transform(X_train)
#X_test_pca = pca.transform(X_test)

#print(X_train_pca.shape)
#print(X_test_pca.shape)

#knc=KNeighborsClassifier(n_neighbors=1)
#knc.fit(X_train_pca, y_train)
#print(knc.score(X_test_pca, y_test))
#print(pca.components_.shape)

#plt.imshow(a.images[0])
#plt.figure()
#print(a.images[0].shape)
#plt.imshow(pca.components_[0].reshape(62,47))
#plt.figure()
#plt.imshow(pca.components_[1].reshape(62,47))

nmf= NMF(n_components=15,random_state=43,max_iter=10000)
nmf.fit(X_train)
X_train_nmf = nmf.transform(X_train)
X_test_nmf = nmf.transform(X_test)

print(X_train_nmf.shape)
print(X_test_nmf.shape)


plt.imshow(nmf.components_[0].reshape(62,47))
plt.figure()
plt.imshow(nmf.components_[1].reshape(62,47))



























