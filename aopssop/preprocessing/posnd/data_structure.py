import numpy as np


class Data:
    # Example of class for storage of data for analysis
    """
    Class to storage dara for preprocessing

    :features_matrix: 2D array (only numeric), 1st D are objects and 2nd D are features (np.array).
    :features_types: 1D array (only=["num", "cat", None]), with len = 2nd D features_matrix (np.array).
    :features_names: 1D array (only str), with len = features_matrix (np.array).
    :thresholds_min_number_of_predicted_labels: 1D int array, with len = len of 2D features (array).
    :labels_matrix: 2D array (only numeric), 1st D are objects and 2nd D are labels, with 1st D = 1st D of features_matrix (np.array).
    :labels_types: 1D array (only=["num", "cat", None]), with len = 2nd D labels_matrix (np.array).
    :labels_names: 1D array (only str), with len = labels_matrix (np.array).
    :n_jobs: setting for the multiprocess parallel processing (int).
    :feature_names_substrings: str arrays of categorical and numerical features substrings (dict).
    :types_priority: key-value structure to define priorities of data types checking methods(dict).
    :fill_method: setting for the selection of the clustering method (str). Possible = ["mean_mode", "centroids"]. Default = "mean_mode".
    :n_clusters: setting for the selection of the number of clusters(int). Default = 10.
    :cluster_max_iter: setting for the selection of the number of clustering iterations (int). Default = 10.
    :thresholds_correlation_with_label: key-value structure for the informativity analysis settings (dict). Key "num_num" - 1D float array (range = [0,1]) with len = 2nd D labels_matrix. Key "cat_cat" - 1D float array (range = [0,1]) with len = 2nd D labels_matrix. Key "num_cat" - 1D float array (range = [0,1]) with len = 2nd D labels_matrix.
    :thresholds_min_number_of_predicted_labels: 1D int array, with len = len of 2D features (array).
    :thresholds_multicolinear: key-value structure for the multicolinearity analysis settings (dict). Key "num_num" - float (range = [0,1]). Key "cat_cat" - float (range = [0,1]). Key "num_cat" - float (range = [0,1]).
    """

    # MULTICOLINEAR SETTINGS
    # dict with following keys
    # {
    #     "num_num": float (range = [0,1]),
    #     "cat_cat": float (range = [0,1]),
    #     "num_cat": float (range = [0,1])
    # }

    def delete_features_by_idx(self, idx_to_delete) -> None:
        # Method for removing features by their idx in feature matrix

        self.features_matrix = np.delete(self.features_matrix, idx_to_delete, axis=1)
        self.features_types = np.delete(self.features_types, idx_to_delete, axis=0)
        self.features_names = np.delete(self.features_names, idx_to_delete, axis=0)
        self.thresholds_min_number_of_predicted_labels = np.delete(self.thresholds_min_number_of_predicted_labels, idx_to_delete, axis=0)

    def transpose(self):
        # Method for feature/label matrix transpose
        self.features_matrix = np.array(list(zip(*self.features_matrix)))
        self.labels_matrix = np.array(list(zip(*self.labels_matrix)))

    def __init__(self):
        # EXAMPLE OF DATA FIELDS AND THEIR REQUIREMENT

        # INITIAL DATA SETTINGS
        # 2D ndarray (only numeric), 1st D are objects and 2nd D are features
        self.features_matrix = None
        # 2D ndarray (only numeric), 1st D are objects and 2nd D are labels, with 1st D = 1st D of features_matrix
        self.labels_matrix = None

        # 1D ndarray (only=["num", "cat", None]), with len = 2nd D features_matrix
        self.features_types = None
        # 1D ndarray (only=["num", "cat", None]), with len = 2nd D labels_matrix
        self.labels_types = None

        # 1D ndarray (only str), with len = features_matrix
        self.features_names = None
        # 1D ndarray (only str), with len = labels_matrix
        self.labels_names = None

        # MULTIPROCESSING SETTINGS
        self.n_jobs = None

        # DATA TYPES SETTINGS
        # dict
        # {
        #     "num": array of str,
        #     "cat": array of str
        # }
        self.feature_names_substrings = None

        # int
        self.feature_max_cat = None

        # dict with following keys
        # {
        #     "manual": float,
        #     "substring": float,
        #     "unique": float,
        #     "float": float
        # }
        self.types_priority = None

        # CLUSTERISATION SETTINGS
        # can be ["mean_mode", "centroids"]
        self.fill_method = "mean_mode"

        # int
        self.n_clusters = 10

        # int
        self.cluster_max_iter = 5

        # INFORMATIVITY SETTINGS
        # dict with following keys
        # {
        #     "num_num": 1D float array (range = [0,1]) with len = 2nd D labels_matrix,
        #     "cat_cat": 1D float array (range = [0,1]) with len = 2nd D labels_matrix,
        #     "num_cat": 1D float array (range = [0,1]) with len = 2nd D labels_matrix
        # }
        self.thresholds_correlation_with_label = None

        # 1D array int array, with len = len of 2D features
        self.thresholds_min_number_of_predicted_labels = None

        # MULTICOLINEAR SETTINGS
        # dict with following keys
        # {
        #     "num_num": float (range = [0,1]),
        #     "cat_cat": float (range = [0,1]),
        #     "num_cat": float (range = [0,1])
        # }
        self.thresholds_multicolinear = None
