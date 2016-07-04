import matplotlib
matplotlib.use('Agg')
import numpy as np
import gzip, cPickle
from tsne import bh_sne
import matplotlib.pyplot as plt

#load data
X = np.load('tsne_features.npy').astype(dtype=np.float64)
Y = np.load('tsne_labels.npy')

#sanity check
unique, counts = np.unique(Y, return_counts=True)
print unique
#array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print counts
#array([ 997, 1113,  988,  983,  984,  921,  979, 1072,  963, 1000])
#or something like this (all counts should be approximately equal)

#t-SNE compute features
X_2d = bh_sne(X)

#plot clusters
fig, ax = plt.subplots()
matplotlib.rcParams['figure.figsize'] = 20, 20
ax.scatter(X_2d[:, 0], X_2d[:, 1], c=Y)
matplotlib.pyplot.savefig('tsne.png')