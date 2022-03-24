import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.ion()

a=load_breast_cancer()
n_samples=a.target.shape[0]
print(a.target)
ss=StandardScaler()
ss.fit(a.data)
X_scaled=ss.transform(a.data)

pca=PCA(n_components=2,random_state=13)
pca.fit(X_scaled)
X_pca=pca.transform(X_scaled)

for i in range(n_samples):
    if a.target[i]==0:
        plt.scatter(X_pca[:,0][i],X_pca[:,1][i],color='green',
                    marker='o')
    if a.target[i]==1:
        plt.scatter(X_pca[:,0][i],X_pca[:,1][i],color='orange',
                    marker='^')


plt.xlabel("First principal component")
plt.ylabel("Second principal component")
plt.gca().set_aspect("equal")
print(pca.components_)
print(pca.components_.shape)

plt.figure()
plt.imshow(pca.components_, cmap='viridis')
plt.yticks(range(2),('component:1','component:2'))
plt.xlabel("Features")
plt.ylabel("Principal Components")
plt.colorbar()
plt.subplots_adjust(right=0.952,left=0.217)

