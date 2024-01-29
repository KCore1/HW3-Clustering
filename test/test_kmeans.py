# Write your k-means unit tests here
import numpy as np
import pytest
from cluster import KMeans, make_clusters, plot_clusters, plot_multipanel

def test_kmeans_invalid_inputs():
    mat, labels = make_clusters()

def test_kmeans_default():
    mat, labels = make_clusters()

    # Create KMeans instance and fit the data
    kmeans = KMeans(k=2)
    kmeans.fit(mat)

    