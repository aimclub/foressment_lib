import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np

from aopssop import \
    CheckDataTypes, ClusterFilling, Multicolinear, Informativity, __Verbose__, Data


def posnd_example_titanic():
    """
    Example of the POSND algorithm work on the titanic dataset.
    This dataset is suitable, because it contains categorical and numerical features, while some values of features are empty.
    Moreover, the dataset is small, which helps to receive results fast.

    The following features were used: "PassengerId", "Pclass", "Age", "SibSp", "Parch".
    As well as the following labels: "Survived", "Fare".

    As a result, no non-informative features were found.
    As well as it was suggested to remove "Age" and "SibSp" features in accordance with multicolinear analysis.
    """
    # TEST BASED ON titanic.csv DATA

    # INITIAL DATA SETTINGS
    data = Data()
    titanic_path = '../datasets/titanic.csv'
    titanic = pd.read_csv(titanic_path)
    data.features_names = ["PassengerId", "Pclass", "Age", "SibSp", "Parch"]
    data.labels_names = ["Survived", "Fare"]
    data.features_matrix = np.array(titanic[data.features_names])
    data.labels_matrix = np.array(titanic[data.labels_names])
    data.features_types = ["cat", "cat", "num", None, None]
    data.labels_types = ["cat", None]

    # MULTIPROC OPTION
    data.n_jobs = 2

    # DATA TYPES SETTINGS
    data.feature_names_substrings = {
        "num": ["age", "id"],
        "cat": ["surv", "tick", "cabin"]
    }
    data.feature_max_cat = 10
    data.types_priority = {
        "manual": 0.5,
        "substring": 1,
        "unique": 1,
        "float": 0.3
    }

    # CLUSTERING SETTINGS
    # can be = [mean_mode, centroids]
    data.fill_method = "mean_mode"
    data.n_clusters = 10
    data.cluster_max_iter = 5

    # INFORMATIVITY SETTINGS
    data.thresholds_correlation_with_label = {
        "num_num": [0.2] * len(data.labels_names),
        "cat_cat": [0.1] * len(data.labels_names),
        "num_cat": [0.2] * len(data.labels_names)
    }
    data.thresholds_min_number_of_predicted_labels = [1] * len(data.features_names)

    # MULTICOLINEAR SETTINGS
    data.thresholds_multicolinear = {
        "num_num": 0.9,
        "cat_cat": 0.7,
        "num_cat": 0.8
    }

    # SET TRUE TO GET OUTPUT
    __Verbose__.PrintLog.instance().set_print_mode(True)
    __Verbose__.PrintLog.instance().set_severity_level("status")

    # RUN
    CheckDataTypes.CheckDataTypes.correct_types(data)
    ClusterFilling.ClusterFilling.fill(data)
    Informativity.Informativity.calculate_informativity(data)
    Multicolinear.MultiCollinear.remove_uninformative_features(data)


def posnd_example_basic():
    # TEST BASED ON SYNTHETIC DATA
    """
    Basic example of the POSND algorithm work.
    In this example, POSND algorithm is applied to the generated data.
    All data reduction steps are printed in console.

    The following features were used: "name", "age", "gender", "last_name" and "price".
    As well as the following labels: "salary", "color", "id".

    As a result, it was suggested to remove "gender" and "price" features due to their noninformativeness.
    As well as to remove "age" feature in accordance with multicolinear analysis.
    """

    # INITIAL DATA SETTINGS
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

    # SET True TO GET OUTPUT
    __Verbose__.PrintLog.instance().set_print_mode(True)
    __Verbose__.PrintLog.instance().set_severity_level("status")

    # RUN
    CheckDataTypes.CheckDataTypes.correct_types(data)
    ClusterFilling.ClusterFilling.fill(data)
    Informativity.Informativity.calculate_informativity(data)
    Multicolinear.MultiCollinear.remove_uninformative_features(data)


if __name__ == '__main__':
    print('===================')
    print('BASIC EXAMPLE')
    print('===================')
    posnd_example_basic()
    print('===================')

    # print('===================')
    # print('TITANIC EXAMPLE')
    # print('===================')
    # posnd_example_titanic()
    # print('===================\n')
