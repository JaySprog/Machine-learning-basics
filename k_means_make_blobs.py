import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
plt.ion()

a=make_blobs(random_state=23)
plt.scatter(a[0][:,0],a[0][:,1])
plt.figure()

kmeans=KMeans(n_clusters=3,random_state=12,max_iter=300)
kmeans.fit(a[0])
print(kmeans.labels_)
print(kmeans.predict(a[0]))
print(kmeans.cluster_centers_)

for i in range(100):
    if kmeans.labels_[i]==0:
        plt.scatter(a[0][:,0][i],a[0][:,1][i],color='red')
    if kmeans.labels_[i]==1:
        plt.scatter(a[0][:,0][i],a[0][:,1][i],color='blue')
    if kmeans.labels_[i]==2:
        plt.scatter(a[0][:,0][i],a[0][:,1][i],color='green')

plt.plot(kmeans.cluster_centers_[0][0],kmeans.cluster_centers_[0][1],
            marker='^',markeredgewidth=4,markeredgecolor='grey')
plt.plot(kmeans.cluster_centers_[1][0],kmeans.cluster_centers_[1][1],
            marker='^',markeredgewidth=4,markeredgecolor='grey')
plt.plot(kmeans.cluster_centers_[2][0],kmeans.cluster_centers_[2][1],
            marker='^',markeredgewidth=4,markeredgecolor='grey')


kmeans=KMeans(n_clusters=2,random_state=31,max_iter=300)
kmeans.fit(a[0])
plt.figure()
for i in range(100):
    if kmeans.labels_[i]==0:
        plt.scatter(a[0][:,0][i],a[0][:,1][i],color='red')
    if kmeans.labels_[i]==1:
        plt.scatter(a[0][:,0][i],a[0][:,1][i],color='blue')

plt.plot(kmeans.cluster_centers_[0][0],kmeans.cluster_centers_[0][1],
            marker='^',markeredgewidth=4,markeredgecolor='grey')
plt.plot(kmeans.cluster_centers_[1][0],kmeans.cluster_centers_[1][1],
            marker='^',markeredgewidth=4,markeredgecolor='grey')

kmeans=KMeans(n_clusters=5,random_state=42,max_iter=300)
kmeans.fit(a[0])
plt.figure()
for i in range(100):
    if kmeans.labels_[i]==0:
        plt.scatter(a[0][:,0][i],a[0][:,1][i],color='red')
    if kmeans.labels_[i]==1:
        plt.scatter(a[0][:,0][i],a[0][:,1][i],color='blue')
    if kmeans.labels_[i]==2:
        plt.scatter(a[0][:,0][i],a[0][:,1][i],color='green')
    if kmeans.labels_[i]==3:
        plt.scatter(a[0][:,0][i],a[0][:,1][i],color='black')
    if kmeans.labels_[i]==4:
        plt.scatter(a[0][:,0][i],a[0][:,1][i],color='yellow')
        


