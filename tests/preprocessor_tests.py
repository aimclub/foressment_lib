import unittest
import numpy as np

from foressment_ai import \
    CheckDataTypes, ClusterFilling, Multicolinear, Informativity, Data

data = Data()
data.features_matrix = np.array(list(zip(*[
    [1, 1, 1, 1, np.nan, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [4, 4, 4, 4, 4, 4, 4, 4, 3, np.nan, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, np.nan, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, np.nan, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2],
    [4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, np.nan, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
])))
data.features_types = ["num", "num", "cat", "cat", "num"]
data.features_names = ["name", "age", "gender", "last_name", "price"]
data.labels_matrix = np.array(list(zip(*[
    [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2],
    [4, 4, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
])))
data.labels_types = ["num", "cat", "num"]
data.labels_names = ["salary", "color", "id"]

# MULTIPROC OPTION
data.n_jobs = 2

# DATA TYPES SETTINGS
data.feature_names_substrings = {
    "num": ["ge", "price"],
    "cat": ["name", "id", "color"]
}
data.feature_max_cat = 5
data.types_priority = {
    "manual": 0.8,
    "substring": 0.5,
    "unique": 1,
    "float": 0.3
}

# CLUSTERING SETTINGS
# can be = [mean_mode, centroids]
data.fill_method = "mean_mode"
data.n_clusters = 3
data.cluster_max_iter = 5

# INFORMATIVITY SETTINGS
data.thresholds_correlation_with_label = {
    "num_num": [0.3, 0.3, 0.3],
    "cat_cat": [0.3, 0.3, 0.3],
    "num_cat": [0.3, 0.3, 0.3]
}
data.thresholds_min_number_of_predicted_labels = [1, 1, 2, 1, 3]

# MULTICOLINEAR SETTINGS
data.thresholds_multicolinear = {
    "num_num": 0.3,
    "cat_cat": 0.3,
    "num_cat": 0.3
}


class PreprocessorTestCase(unittest.TestCase):
    def test_check_data_types(self):
        expected_features_types = ['cat', 'num', 'cat', 'cat', 'num']
        CheckDataTypes.CheckDataTypes.correct_types(data)
        self.assertEqual(data.features_types, expected_features_types)

    def test_cluster_filling(self):
        expected_features_matrix = np.array(list(zip(*[
            [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2],
            [4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        ])))
        ClusterFilling.ClusterFilling.fill(data)
        self.assertTrue(np.array_equal(data.features_matrix, expected_features_matrix))

    def test_informativity(self):
        expected_features_matrix = np.array(list(zip(*[
            [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2]
        ])))
        Informativity.Informativity.calculate_informativity(data)
        self.assertTrue(np.array_equal(data.features_matrix, expected_features_matrix))

    def test_multicolinearity(self):
        expected_features_matrix = np.array(list(zip(*[
            [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2]
        ])))
        Multicolinear.MultiCollinear.remove_uninformative_features(data)
        self.assertTrue(np.array_equal(data.features_matrix, expected_features_matrix))