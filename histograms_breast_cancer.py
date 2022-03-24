import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
plt.ion()

a=load_breast_cancer()
malignant = a.data[a.target == 0]
benign = a.data[a.target == 1]

fig, axes = plt.subplots(15, 2, figsize=(10, 20))
ax = axes.ravel()
for i in range(30):
    ax[i].hist(malignant[:,i], bins=50,color='lightgreen',alpha=0.7,edgecolor="black")
    ax[i].hist(benign[:,i],bins=50,color='steelblue',alpha=0.5,edgecolor="black")


    
    
    


