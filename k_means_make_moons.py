import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
plt.ion()

a=make_moons(n_samples=200,random_state=25, noise=0.05)

kmeans=KMeans(n_clusters=2,random_state=2,max_iter=300)
kmeans.fit(a[0])

for i in range(200):
    if kmeans.labels_[i]==0:
        plt.scatter(a[0][:,0][i],a[0][:,1][i],color='red')
    if kmeans.labels_[i]==1:
        plt.scatter(a[0][:,0][i],a[0][:,1][i],color='blue')

plt.plot(kmeans.cluster_centers_[0][0],kmeans.cluster_centers_[0][1],
            marker='^',markeredgewidth=4,markeredgecolor='grey')
plt.plot(kmeans.cluster_centers_[1][0],kmeans.cluster_centers_[1][1],
            marker='^',markeredgewidth=4,markeredgecolor='grey')

plt.figure()

kmeans = KMeans(n_clusters=10, random_state=3)
kmeans.fit(a[0])
print(kmeans.transform(a[0]))
print(kmeans.transform(a[0]).shape)

for i in range(200):
    if kmeans.labels_[i]==0:
        plt.scatter(a[0][:,0][i],a[0][:,1][i],color='red')
    if kmeans.labels_[i]==1:
        plt.scatter(a[0][:,0][i],a[0][:,1][i],color='blue')
    if kmeans.labels_[i]==2:
        plt.scatter(a[0][:,0][i],a[0][:,1][i],color='black')
    if kmeans.labels_[i]==3:
        plt.scatter(a[0][:,0][i],a[0][:,1][i],color='green')
    if kmeans.labels_[i]==4:
        plt.scatter(a[0][:,0][i],a[0][:,1][i],color='grey')
    if kmeans.labels_[i]==5:
        plt.scatter(a[0][:,0][i],a[0][:,1][i],color='pink')
    if kmeans.labels_[i]==6:
        plt.scatter(a[0][:,0][i],a[0][:,1][i],color='cyan')
    if kmeans.labels_[i]==7:
        plt.scatter(a[0][:,0][i],a[0][:,1][i],color='orange')
    if kmeans.labels_[i]==8:
        plt.scatter(a[0][:,0][i],a[0][:,1][i],color='violet')
    if kmeans.labels_[i]==9:
        plt.scatter(a[0][:,0][i],a[0][:,1][i],color='indigo')

plt.plot(kmeans.cluster_centers_[0][0],kmeans.cluster_centers_[0][1],
            marker='^',markeredgewidth=4,markeredgecolor='grey')
plt.plot(kmeans.cluster_centers_[1][0],kmeans.cluster_centers_[1][1],
            marker='^',markeredgewidth=4,markeredgecolor='grey')
plt.plot(kmeans.cluster_centers_[2][0],kmeans.cluster_centers_[2][1],
            marker='^',markeredgewidth=4,markeredgecolor='grey')
plt.plot(kmeans.cluster_centers_[3][0],kmeans.cluster_centers_[3][1],
            marker='^',markeredgewidth=4,markeredgecolor='grey')
plt.plot(kmeans.cluster_centers_[4][0],kmeans.cluster_centers_[4][1],
            marker='^',markeredgewidth=4,markeredgecolor='grey')
plt.plot(kmeans.cluster_centers_[5][0],kmeans.cluster_centers_[5][1],
            marker='^',markeredgewidth=4,markeredgecolor='grey')
plt.plot(kmeans.cluster_centers_[6][0],kmeans.cluster_centers_[6][1],
            marker='^',markeredgewidth=4,markeredgecolor='grey')
plt.plot(kmeans.cluster_centers_[7][0],kmeans.cluster_centers_[7][1],
            marker='^',markeredgewidth=4,markeredgecolor='grey')
plt.plot(kmeans.cluster_centers_[8][0],kmeans.cluster_centers_[8][1],
            marker='^',markeredgewidth=4,markeredgecolor='grey')
plt.plot(kmeans.cluster_centers_[9][0],kmeans.cluster_centers_[9][1],
            marker='^',markeredgewidth=4,markeredgecolor='grey')


