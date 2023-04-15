import sys

import numpy as np
import pandas as pd
import scipy

from foressment_ai.preprocessing.preprocessor import data_structure as Data_Structure
from foressment_ai.preprocessing.preprocessor.__verbose__ import PrintLog


class Informativity:
    # A class that remove uninformative features by correlation of feature with label.
    # Processing preferences should be defined in Data.
    # Typical usage example:
    # Informativity.calculate_informativity(data)

    """
    Class for the deletion of non-informative features based on the informativity analysis

    Typical usage example:
    Informativity.calculate_informativity(data)

    :param crosstab: cross table for calculation of features informativity
    :type crosstab: numpy.array

    :param stat: test statistics
    :type stat: numpy.float

    :param obs: number of observations
    :type obs: numpy.int

    :param mini: minimum value between the columns and the rows of the cross table
    :type mini: int

    :param informativity_matrix: feature-label informativity matrix
    :type informativity_matrix: list

    :return: stat / (obs * mini)
    :rtype: numpy.float
    """

    @staticmethod
    def calculate_informativity(data: Data_Structure.Data):
        """
        Method that removes non-informative features

        :param use_spearman: if True than Spearman correlation is used (recommended if type of data in unknown)
        :type use_spearman: bool

        :param use_cramerV: if True than Cramer correlation is used (recommended for categorical data)
        :type use_cramerV: bool

        :param use_pearson: if True than Pearson correlation is used (recommend for numerical data)
        :type use_pearson: bool

        :param informativity: if True than decision is made in accordance with thresholds
        :type informativity: numpy.bool
        """

        def __cramers_V__(var1, var2):
            # Method that calculate informativity between categorical values

            # Cross table building
            crosstab = np.array(pd.crosstab(var1, var2, rownames=None, colnames=None))

            # Keeping of the test statistic of the Chi2 test
            stat = scipy.stats.chi2_contingency(crosstab)[0]

            # Number of observations
            obs = np.sum(crosstab)

            # Take the minimum value between the columns and the rows of the cross table
            mini = min(crosstab.shape) - 1

            return stat / (obs * mini)

        PrintLog.instance().info("START INFORMATIVITY ANALYSIS")

        # Transpose features and labels to get columns
        # - don't forget to transpose back after analysis
        data.transpose()

        # Create empty feature-label informativity matrix
        informativity_matrix = [[0] * len(data.labels_matrix)]*len(data.features_matrix)

        # Calculate correlation between feature and label
        for column_idx, column in enumerate(data.features_matrix):
            for label_idx, label in enumerate(data.labels_matrix):

                # Select informativity type
                # use Spearman corr. if we dont know data types or they are categorial & numerical
                # use CramerV corr. if data types are both categorial
                # use Pearson corr. if data types are both numerical
                # we make decision by defined threshold for that feature

                use_spearman = (data.features_types[column_idx] is None or data.labels_types is None) or \
                              (data.features_types[column_idx] != data.labels_types[label_idx])
                use_cramerV = not use_spearman \
                              and data.features_types[column_idx] == data.labels_types[label_idx] == "cat"
                use_pearson = not use_spearman \
                              and data.features_types[column_idx] == data.labels_types[label_idx] == "num"
                informativity = None
                if use_spearman:
                    informativity = 1 - scipy.stats.spearmanr(column, label)[1] \
                                    > data.thresholds_correlation_with_label["num_cat"][label_idx]
                if use_cramerV:
                    informativity = __cramers_V__(column, label) \
                                    > data.thresholds_correlation_with_label["cat_cat"][label_idx]
                if use_pearson:
                    informativity = 1 - scipy.stats.pearsonr(column, label)[1] \
                                    > data.thresholds_correlation_with_label["num_num"][label_idx]

                # Raise exceprion if there is unknown data type
                if informativity is None:
                    raise Exception("wrong label types=", data.labels_types, "or features types=", data.features_types)

                # Fill informativity matrix
                informativity_matrix[column_idx][label_idx] = informativity

        # Remember uninformative features
        uninformative_features_idx = set()
        for idx, feature_name in enumerate(data.features_names):
            # Count how many labels can predict that feature
            can_predict_labels = np.size(informativity_matrix[idx]) - np.count_nonzero(informativity_matrix[idx])
            # Make decision either this amount of labels is enough
            result = can_predict_labels >= data.thresholds_min_number_of_predicted_labels[idx]
            # Add to delete list if not
            if not result:
                PrintLog.instance().status((
                    feature_name, " is uninformative, can predict ",
                    can_predict_labels, ", while required ",
                    data.thresholds_min_number_of_predicted_labels[idx]
                ))
                uninformative_features_idx.add(idx)

        # Transpose data back to normal state and delete uninformative features
        PrintLog.instance().info(
            (
                "REMOVE UNINFORMATIVE FEATURES: [" ,
                ", ".join([data.features_names[i] for i in uninformative_features_idx]),
                "]"
            ) if len(uninformative_features_idx)>0
            else "NO UNINFORMATIVE FEATURES"
        )
        data.transpose()
        data.delete_features_by_idx(list(uninformative_features_idx))

        if len(data.features_matrix[0]) == 0:
            PrintLog.instance().warn("ALL FEATURES WERE DELETED - CHECK THE SETTINGS")
            sys.exit(0)
