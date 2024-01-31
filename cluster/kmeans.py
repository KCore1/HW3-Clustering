import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        # Error checking
        if not isinstance(k, int):
            raise TypeError("k must be an integer.")
        if not isinstance(tol, float):
            raise TypeError("tol must be a float.")
        if not isinstance(max_iter, int):
            raise TypeError("max_iter must be an integer.")
        if k < 1:
            raise ValueError("k must be a positive integer.")
        if tol <= 0:
            raise ValueError("tol must be a positive non-zero float.")
        if max_iter < 1:
            raise ValueError("max_iter must be a positive non-zero integer.")
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        # Error checking
        if not isinstance(mat, np.ndarray):
            raise TypeError("mat must be a numpy array.")

        # TODO: Implement kmeans++?
        self.centroids = {i: mat[np.random.randint(0, len(mat))] for i in range(self.k)}

        for _ in range(self.max_iter):
            # Compute all squared distances between data points and centroids
            distances = cdist(mat, list(self.centroids.values()), "sqeuclidean")
            # Assign each data point to the nearest centroid
            closest_centroids = np.argmin(distances, axis=1)

            # Recompute clusters
            self.clusters = {i: [] for i in range(self.k)}
            for i, centroid in enumerate(closest_centroids):
                self.clusters[centroid].append(mat[i])

            # Keep a copy of the old centroids for checking tolerance
            old_centroids = dict(self.centroids)

            # Recalculate centroids
            for cluster in self.clusters:
                self.centroids[cluster] = np.mean(self.clusters[cluster], axis=0)

            # Check tolerance
            stable = True
            for centroid in self.centroids:
                original_centroid = old_centroids[centroid]
                current_centroid = self.centroids[centroid]
                # Checking tolerance this way may quit early for movements
                # that are significant but smaller than the value of tol. Make
                # sure tol is set appropriate to the scale of the data.
                if np.linalg.norm(current_centroid - original_centroid) > self.tol:
                    stable = False

            # Break out of the loop if the centroids have stabilized
            if stable:
                break

        # Get centroids for each data point
        centroids_array = np.array(
            [
                self.centroids[cluster]
                for cluster in np.argmin(
                    cdist(mat, list(self.centroids.values())), axis=1
                )
            ]
        )

        # Compute squared distances between data points and centroids
        squared_distances = np.sum((cdist(mat, centroids_array, "sqeuclidean")), axis=1)

        # Store mean squared error
        self.error = np.mean(squared_distances)

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        # Error checking
        if not hasattr(self, "centroids") or not hasattr(self, "clusters"):
            raise ValueError(
                "Model must be fit before predictions can be made. Call the fit() method on the appropriate data first."
            )

        if not isinstance(mat, np.ndarray):
            raise TypeError("mat must be a numpy array.")

        if mat.ndim != 2:
            raise ValueError(
                "mat must be a 2D matrix where the rows are observations and columns are features."
            )

        if mat.shape[1] != next(iter(self.centroids.values())).shape[0]:
            raise ValueError(
                "Feature dimension of input does not match the dimension of the fitted data."
            )

        try:
            # Compute distances from each point in mat to each centroid
            distances = cdist(mat, list(self.centroids.values()), "sqeuclidean")
        except ValueError:
            raise ValueError(
                "Distances could not be computed. The feature type that was fitted may not have a \
                              defined distance operation to the given data."
            )

        # Assign each data point to the nearest centroid
        closest_centroids = np.argmin(distances, axis=1)

        return closest_centroids

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        if not hasattr(self, "error"):
            raise ValueError(
                "Model must be fit before predictions can be made. Call the fit() method on the appropriate data first."
            )

        return self.error

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        if not hasattr(self, "centroids"):
            raise ValueError(
                "Model must be fit before predictions can be made. Call the fit() method on the appropriate data first."
            )

        centroids_array = np.array(list(self.centroids.values()))
        return centroids_array
