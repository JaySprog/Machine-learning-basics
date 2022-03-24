import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score
plt.ion()

a=make_moons(n_samples=200,random_state=13, noise=0.05)
plt.scatter(a[0][:,0],a[0][:,1])
plt.figure()

dbscan=DBSCAN(eps=0.3,min_samples=10)
dbscan.fit(a[0])
ari=adjusted_rand_score(a[1], dbscan.labels_)
sc=silhouette_score(a[0],dbscan.labels_)
for i in range(200):
    if dbscan.labels_[i]==0:
        plt.scatter(a[0][:,0][i],a[0][:,1][i],color='red')
    if dbscan.labels_[i]==1:
        plt.scatter(a[0][:,0][i],a[0][:,1][i],color='blue')
    if dbscan.labels_[i]==-1:
        plt.scatter(a[0][:,0][i],a[0][:,1][i],color='grey')
str_ari=['DBSCAN:ARI_score',str(ari),'silhouette_score:',str(sc)] 
plt.title(str_ari)



plt.figure()
kmeans=KMeans(n_clusters=2,random_state=2,max_iter=300)
kmeans.fit(a[0])
ari=adjusted_rand_score(a[1], kmeans.labels_)
sc=silhouette_score(a[0],kmeans.labels_)
for i in range(200):
    if kmeans.labels_[i]==0:
        plt.scatter(a[0][:,0][i],a[0][:,1][i],color='red')
    if kmeans.labels_[i]==1:
        plt.scatter(a[0][:,0][i],a[0][:,1][i],color='blue')

str_ari=['KMeans:ARI_score',str(ari),'silhouette_score:',str(sc)] 
plt.title(str_ari)



plt.figure()
ac=AgglomerativeClustering(n_clusters=2,linkage='ward')
ac.fit(a[0])
ari=adjusted_rand_score(a[1], ac.labels_)
sc=silhouette_score(a[0],ac.labels_)
for i in range(200):
    if ac.labels_[i]==0:
        plt.scatter(a[0][:,0][i],a[0][:,1][i],color='red')
    if ac.labels_[i]==1:
        plt.scatter(a[0][:,0][i],a[0][:,1][i],color='blue')
   

str_ari=['AgglomerativeClustering:ARI_score',str(ari),
         'silhouette_score:',str(sc)] 
plt.title(str_ari)
