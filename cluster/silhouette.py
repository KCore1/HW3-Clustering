import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        # Error checking
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be numpy arrays.")

        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X must be a 2D matrix and y must be a 1D array.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of observations.")

        unique_labels = np.unique(y)
        n_clusters = len(unique_labels)
        n_observations = X.shape[0]

        # Verify Silhouette can be calculated
        if not (2 <= n_clusters and n_clusters <= n_observations - 1):
            raise ValueError("Number of labels is not valid for silhouette calculation.")

        # Initialize numpy array for SPEED
        silhouette_scores = np.zeros(n_observations)

        for i in range(n_observations):
            label = y[i]

            # a is the distance to all other observations in the same cluster
            a = np.mean(
                cdist(X[i], X[y == label])
            )

            # b is the distance to all observations in the nearest cluster
            b = np.min(
                [
                    np.mean(cdist(X[i], X[y == other_label]))
                    for other_label in unique_labels
                    if other_label != label
                ]
            )

            # Calculate the silhouette score for the current observation
            silhouette_scores[i] = (b - a) / max(a, b)

        return silhouette_scores
