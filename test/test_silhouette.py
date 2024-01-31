# write your silhouette score unit tests here
import numpy as np
import pytest
from cluster import KMeans, Silhouette, make_clusters
from sklearn.metrics import silhouette_score

def test_silhouette_errors():
    # Test not numpy arrays
    with pytest.raises(TypeError):
        s = Silhouette()
        s.score([1, 2, 3], [0, 1, 2])

    # Test X is not a 2D matrix
    with pytest.raises(ValueError):
        s = Silhouette()
        s.score(np.array([1, 2, 3]), np.array([0, 1, 2]))

    # Test y is not a 1D array
    with pytest.raises(ValueError):
        s = Silhouette()
        s.score(np.array([[1, 2], [3, 4]]), np.array([[0, 1], [2, 3]]))

    # Test X and y have different number of observations
    with pytest.raises(ValueError):
        s = Silhouette()
        s.score(np.array([[1, 2], [3, 4]]), np.array([0, 1, 2]))

    # Test number of labels is not valid for silhouette calculation
    with pytest.raises(ValueError):
        s = Silhouette()
        s.score(np.array([[1, 2], [3, 4]]), np.array([0, 0]))

def test_silhouette_vs_sklearn():
    s = Silhouette()
    kmeans = KMeans(k=3)

    # Test 2 clusters
    mat, labels = make_clusters(n=100, k=2)
    kmeans.fit(mat)
    y = kmeans.predict(mat)
    assert isinstance(s.score(mat, y), np.ndarray)
    assert len(s.score(mat, y)) == 100
    assert np.isclose(np.mean(s.score(mat, y)), silhouette_score(mat, y), atol=0.05)

    # Test 3 clusters
    mat, labels = make_clusters(n=1000, k=3)
    kmeans.fit(mat)
    y = kmeans.predict(mat)
    assert isinstance(s.score(mat, y), np.ndarray)
    assert len(s.score(mat, y)) == 1000
    assert np.isclose(np.mean(s.score(mat, y)), silhouette_score(mat, y), atol=0.05)

    # Test 4 clusters
    mat, labels = make_clusters(n=1000, k=4)
    kmeans.fit(mat)
    y = kmeans.predict(mat)
    assert isinstance(s.score(mat, y), np.ndarray)
    assert len(s.score(mat, y)) == 1000
    assert np.isclose(np.mean(s.score(mat, y)), silhouette_score(mat, y), atol=0.05)