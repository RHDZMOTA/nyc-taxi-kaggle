#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unsupervised Learning - Classification.

import os
os.chdir("")
@author: Rodrigo Hernández-Mota
Contact info: rhdzmota@mxquants.com
"""

import numpy as np


class Kmeans(object):
    """Kmeans algorithm."""

    def __init__(self, n_clusters):
        """Initialization."""
        self.n_clusters = n_clusters

    def _initialCentroids(self, dataset, n_clusters):
        """Initialize centroids as random elements of sample data."""
        samples = dataset.norm_input_data if dataset.normalized else \
            dataset.input_data
        n_samples = np.shape(samples)[0]
        index = np.random.choice(np.arange(n_samples), n_clusters)
        temp = {}
        for i in range(n_clusters):
            temp[i] = np.asarray(samples[index[i]])
        return temp

    def _assign2Nearest(self, samples, centroids):
        """Asign vector to nearest centroid."""
        def getClusterNumber(x, centroids):
            min_distance = float("inf")
            min_cluster = None
            for i in centroids:
                distance = np.linalg.norm(x-centroids[i])
                if distance < min_distance:
                    min_distance = distance
                    min_cluster = i
            return min_cluster

        nearest, _c = {}, 0
        for centroid in list(map(lambda x: getClusterNumber(x, centroids),
                                 samples)):
            if centroid not in nearest:
                nearest[centroid] = []
            nearest[centroid].append(_c)
            _c += 1
        return nearest

    def _updateCentroids(self, samples, nearest):
        """Update centroids value."""
        centroids = {}
        for k in nearest:
            centroids[k] = samples[nearest[k]].mean(0)
        for k in self.centroids[-1]:
            if k not in centroids:
                centroids[k] = self.centroids[-1][k]
        return centroids

    def train(self, dataset, epochs=500):
        """Train k-means."""
        self.centroids = [self._initialCentroids(dataset, self.n_clusters)]
        for epoch in range(epochs):
            samples, _ = dataset.nextBatch()
            nearest = self._assign2Nearest(samples, self.centroids[-1])
            temp = self._updateCentroids(samples, nearest)
            self.centroids.append(temp)
        self.c = self.centroids[-1]
        self.nearest = nearest

    def eval(self, x):
        """Evaluate."""
        return self._assign2Nearest(x, self.c)
