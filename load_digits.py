import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from PIL import Image
plt.ion()

a=load_digits()

pca = PCA(n_components=2,random_state=11)
pca.fit(a.data)
data_pca = pca.transform(a.data)
print(data_pca[:,1][13])
colors = ['red', 'blue','green','orange','black','pink',
          'yellow','grey','indigo','cyan']

for i in range(a.target.shape[0]):
    plt.text(data_pca[:,0][i],data_pca[:,1][i],str(a.target[i]),
             color = colors[a.target[i]])

plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.xlim(data_pca[:, 0].min(), data_pca[:, 0].max())
plt.ylim(data_pca[:, 1].min(), data_pca[:, 1].max())



tsne=TSNE(n_components=2,random_state=34)
data_tsne = tsne.fit_transform(a.data)

plt.figure()

colors = ['red', 'blue','green','orange','black','pink',
          'yellow','grey','indigo','cyan']

for i in range(a.target.shape[0]):
    plt.text(data_tsne[:,0][i],data_tsne[:,1][i],str(a.target[i]),
             color = colors[a.target[i]])

plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.xlim(data_tsne[:, 0].min(), data_tsne[:, 0].max())
plt.ylim(data_tsne[:, 1].min(), data_tsne[:, 1].max())
