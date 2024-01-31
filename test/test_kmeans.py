# Write your k-means unit tests here
import numpy as np
import pytest
import timeit
from cluster import KMeans, make_clusters, plot_clusters, plot_multipanel, Silhouette
from sklearn.metrics.cluster import contingency_matrix


def test_kmeans_invalid_inputs():
    mat, labels = make_clusters()

    # Test TypeError for k
    with pytest.raises(TypeError):
        kmeans = KMeans(k="2")
        kmeans.fit(mat)

    # Test TypeError for tol
    with pytest.raises(TypeError):
        kmeans = KMeans(k=2, tol="0.001")
        kmeans.fit(mat)

    # Test TypeError for max_iter
    with pytest.raises(TypeError):
        kmeans = KMeans(k=2, tol=0.001, max_iter="100")
        kmeans.fit(mat)

    # Test ValueError for k
    with pytest.raises(ValueError):
        kmeans = KMeans(k=0)
        kmeans.fit(mat)

    # Test ValueError for tol
    with pytest.raises(ValueError):
        kmeans = KMeans(k=2, tol=-0.01)
        kmeans.fit(mat)

    # Test ValueError for max_iter
    with pytest.raises(ValueError):
        kmeans = KMeans(k=2, tol=0.001, max_iter=0)
        kmeans.fit(mat)


def test_kmeans_fit_invalid_inputs():
    mat, labels = make_clusters()

    # Test TypeError for mat
    with pytest.raises(TypeError):
        kmeans = KMeans(k=2)
        kmeans.fit("mat")


def test_kmeans_predict_invalid_inputs():
    mat, labels = make_clusters()

    # Test TypeError for mat
    with pytest.raises(TypeError):
        kmeans = KMeans(k=2)
        kmeans.fit(mat)
        kmeans.predict("mat")

    # Test not fitted
    with pytest.raises(ValueError):
        kmeans = KMeans(k=2)
        kmeans.predict(mat)

    # Test dimensions for mat
    with pytest.raises(ValueError):
        kmeans = KMeans(k=2)
        kmeans.fit(mat)
        kmeans.predict(np.array([1, 2, 3]))

    # Test feature dimensions are correct
    with pytest.raises(ValueError):
        kmeans = KMeans(k=2)
        kmeans.fit(mat)
        kmeans.predict(np.array([[1, 2, 3]]))

    # Test if valid distance can be calculated
    with pytest.raises(ValueError):
        kmeans = KMeans(k=2)
        kmeans.fit(mat)
        kmeans.predict(np.array([["1", "2"], ["3", "4"]]))


def test_kmeans_get_centroids():
    mat, labels = make_clusters()

    # Create KMeans instance and fit the data
    kmeans = KMeans(k=2)
    kmeans.fit(mat)

    # Test the centroids
    assert kmeans.get_centroids().shape == (2, 2)

    # Test not fitted
    with pytest.raises(ValueError):
        kmeans = KMeans(k=2)
        kmeans.get_centroids()


def test_kmeans_default():
    mat, labels = make_clusters(1000, 3, 4, (-100, 100), 10)
    predict_mat, predict_labels = make_clusters(200, 3, 4, (-100, 100), 10)

    # Create KMeans instance and fit the data
    kmeans = KMeans(k=4)
    kmeans.fit(mat)

    # Test the labels
    assert len(kmeans.predict(mat)) == len(labels)
    assert len(kmeans.predict(predict_mat)) == len(predict_labels)
    # Hacky way to test label correctness for the 4 clusters
    matching_clusters = [max(row) for row in contingency_matrix(labels, kmeans.predict(mat))]
    assert matching_clusters[0] / np.sum(labels == 0) > 0.7
    assert matching_clusters[1] / np.sum(labels == 1) > 0.7
    assert matching_clusters[2] / np.sum(labels == 2) > 0.7
    assert matching_clusters[3] / np.sum(labels == 3) > 0.7

    # Test that tolerance and max_iter are faster when larger and lower, respectively
    # NOTE: This test is not deterministic and may fail if the machine load is dynamic
    kmeans = KMeans(k=4, tol=1e-9, max_iter=100)
    kmeans.fit(mat)
    fast_kmeans = KMeans(k=4, tol=1e-2, max_iter=100)
    fast_kmeans.fit(mat)
    assert timeit.timeit(lambda: kmeans.fit(mat), number=1) > timeit.timeit(
        lambda: fast_kmeans.fit(mat), number=1
    )
    slow_kmeans = KMeans(k=4, tol=1e-2, max_iter=1000)
    slow_kmeans.fit(mat)
    assert timeit.timeit(lambda: slow_kmeans.fit(mat), number=1) > timeit.timeit(
        lambda: fast_kmeans.fit(mat), number=1
    )


if __name__ == "__main__":
    # Plotting the clusters
    mat, labels = make_clusters()
    scorer = Silhouette()

    plot_clusters(mat, labels)

    # Plotting the clusters with different k values
    kmeans = KMeans(k=2)
    kmeans.fit(mat)
    pred = kmeans.predict(mat)
    plot_multipanel(mat, labels, pred, scorer.score(mat, pred))

    kmeans = KMeans(k=3)
    kmeans.fit(mat)
    pred = kmeans.predict(mat)
    plot_multipanel(mat, labels, pred, scorer.score(mat, pred))

    kmeans = KMeans(k=4)
    kmeans.fit(mat)
    pred = kmeans.predict(mat)
    plot_multipanel(mat, labels, pred, scorer.score(mat, pred))
    
