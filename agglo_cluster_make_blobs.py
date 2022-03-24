import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, ward
plt.ion()

a=make_blobs(n_samples=200,random_state=13)
plt.scatter(a[0][:,0],a[0][:,1])
plt.figure()

ac=AgglomerativeClustering(n_clusters=3,linkage='ward')
ac.fit(a[0])
print(ac.labels_)

for i in range(200):
    if ac.labels_[i]==0:
        plt.scatter(a[0][:,0][i],a[0][:,1][i],color='red')
    if ac.labels_[i]==1:
        plt.scatter(a[0][:,0][i],a[0][:,1][i],color='blue')
    if ac.labels_[i]==2:
        plt.scatter(a[0][:,0][i],a[0][:,1][i],color='green')

plt.figure()
linkage_array = ward(a[0])
dendrogram(linkage_array,truncate_mode='lastp')
plt.xlabel("Sample index")
plt.ylabel("Cluster distance")
