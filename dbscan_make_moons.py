import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
plt.ion()

a=make_moons(n_samples=200,random_state=13, noise=0.05)
plt.scatter(a[0][:,0],a[0][:,1])
plt.figure()

dbscan=DBSCAN(eps=0.3,min_samples=10)
dbscan.fit(a[0])

for i in range(200):
    if dbscan.labels_[i]==0:
        plt.scatter(a[0][:,0][i],a[0][:,1][i],color='red')
    if dbscan.labels_[i]==1:
        plt.scatter(a[0][:,0][i],a[0][:,1][i],color='blue')
    if dbscan.labels_[i]==-1:
        plt.scatter(a[0][:,0][i],a[0][:,1][i],color='grey')

