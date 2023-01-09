import numpy as np
import scipy
from sklearn.cluster import KMeans
from aopssop.preprocessing.posnd import data_structure as Data_Structure
from aopssop.preprocessing.posnd.__verbose__ import PrintLog


class ClusterFilling:
    # A class that fill Nones in features based on K-means clusters.
    # Processing preferences should be defined in Data.
    # Typical usage example:
    # ClusterFilling.fill(data)

    """
    Class for the empty values filling process based on clustering

    Typical usage example:
    ClusterFilling.fill(data)

    :param X: array of data to cluster
    :param n_clusters: number of clusters to form
    :param max_iter: Maximum number of EM iterations to perform
    :param column_types: list of feature types

    :return X_hat: Copy of X with the missing values filled in
    """

    @staticmethod
    def __fill_with_centroids__(
            X: np.ndarray,
            n_clusters: np.ndarray,
            max_iter=10
    ) -> np.ndarray:
        """
        K-Means clustering with filling of missing values by centroids

        :param X: array of data to cluster
        :param n_clusters: number of clusters to form
        :param max_iter: Maximum number of EM iterations to perform

        :return X_hat: Copy of X with the missing values filled in
        """

        # Initialize missing values to their column means
        missing = ~np.isfinite(X)
        mu = np.nanmean(X, 0, keepdims=1)
        X_hat = np.where(missing, mu, X)

        for i in range(max_iter):
            if i > 0:
                # initialize KMeans with the previous set of centroids. this is much
                # faster and makes it easier to check convergence (since labels
                # won't be permuted on every iteration), but might be more prone to
                # getting stuck in local minima.
                cls = KMeans(n_clusters, init=prev_centroids)
            else:
                # do multiple random initializations in parallel
                cls = KMeans(n_clusters)

            # perform clustering on the filled-in data
            labels = cls.fit_predict(X_hat)
            centroids = cls.cluster_centers_

            # fill in the missing values based on their cluster centroids
            X_hat[missing] = centroids[labels][missing]

            # when the labels have stopped changing then we have converged
            if i > 0 and np.all(labels == prev_labels):
                break

            prev_labels = labels
            prev_centroids = cls.cluster_centers_

        return X_hat


    @staticmethod
    def __fill_with_stats__(
            X: np.ndarray,
            n_clusters: int,
            max_iter: int,
            column_types: list
    ) -> np.ndarray:

        """
        K-Means clustering with filling of missing values by mean and mode in dependence on feature type

        :param X: array of data to cluster
        :param n_clusters: number of clusters to form
        :param max_iter: Maximum number of EM iterations to perform
        :param column_types: list of feature types

        :return X_hat: Copy of X with the missing values filled in
        """

        # Initialize missing values to their column means
        missing = ~np.isfinite(X)
        mu = np.nanmean(X, 0, keepdims=1)
        X_hat = np.where(missing, mu, X)

        for i in range(max_iter):
            if i > 0:
                # initialize KMeans with the previous set of centroids. this is much
                # faster and makes it easier to check convergence (since labels
                # won't be permuted on every iteration), but might be more prone to
                # getting stuck in local minima.
                cls = KMeans(n_clusters, init=prev_centroids, n_init=1)
            else:
                # do multiple random initializations in parallel
                cls = KMeans(n_clusters, n_init=1)

            # perform clustering on the filled-in data
            labels = cls.fit_predict(X_hat)

            # make clusters map
            clusters = {}
            for cl in range(n_clusters):
                for row in np.where(labels == cl)[0]:
                    clusters[row] = cl

            # find null indexes
            null_indexes = np.argwhere(missing)
            PrintLog.instance().status(("iteration ", i, ", fill ", len(null_indexes), " NaNs"))
            for null_index in null_indexes:
                # find
                missing_value_cluster_number = clusters[null_index[0]]
                missing_value_column = X[:, null_index[1]]
                missing_value_column_cluster = missing_value_column[np.where(labels == missing_value_cluster_number)[0]]
                missing_value_column_cluster = missing_value_column_cluster[~np.isnan(missing_value_column_cluster)]
                fill_value = scipy.mean(missing_value_column_cluster) if column_types[null_index[1]] == "num" else scipy.stats.mode(missing_value_column_cluster, keepdims=True)[0][0]

                X_hat[null_index[0]][null_index[1]] = fill_value

            # when the labels have stopped changing then we have converged
            if i > 0 and np.all(labels == prev_labels):
                PrintLog.instance().status(("=> conversged on iteration ", i))
                break
            prev_labels = labels
            prev_centroids = cls.cluster_centers_
        return X_hat

    @staticmethod
    def fill(data: Data_Structure.Data) -> None:
        """
        K-Means clustering with filling of missing values for data
        """

        PrintLog.instance().info((
            "START CLUSTER FILLING (fill_method=", data.fill_method,
            ", n_clusters=", data.n_clusters,
            ", max_iter=", data.cluster_max_iter, ")"
        ))
        if data.fill_method == "centroids":
            data.features_matrix = ClusterFilling.__fill_with_centroids__(
                X=data.features_matrix,
                n_clusters=data.n_clusters,
                max_iter=data.cluster_max_iter
            )
        else:
            data.features_matrix = ClusterFilling.__fill_with_stats__(
                X=data.features_matrix,
                n_clusters=data.n_clusters,
                max_iter=data.cluster_max_iter,
                column_types=data.features_types
            )
        PrintLog.instance().info("DONE CLUSTER FILLING")
