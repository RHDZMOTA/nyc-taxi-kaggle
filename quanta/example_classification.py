#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Example: Unsupervised learning

@author: Rodrigo Hernández-Mota
Contact info: rhdzmota@mxquants.com
"""

from Unsupervised import Kmeans
from scipy.io import loadmat
import matplotlib.pyplot as plt
from dataHandler import Dataset
import pandas as pd


# load data
m = loadmat('datos2.mat')
m = m['datos2']
data = pd.DataFrame(m.T)

# dataset
dataset = Dataset(input_data=data, output_data=data, normalize=None)
samples, _ = dataset.nextBatch()

# k-means
kmeans = Kmeans(n_clusters=6)
kmeans.train(dataset, epochs=5000)


kmeans.centroids
kmeans.c
kmeans.centroids[0]


plt.plot(data[0], data[1], '.b')
[plt.plot(kmeans.c[i][0], kmeans.c[i][1], 'o') for i in kmeans.c]
plt.show()
