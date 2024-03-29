import sys

import numpy as np
import scipy
import pandas as pd

import foressment_ai.preprocessing.preprocessor.data_structure as Data_Structure
from foressment_ai.preprocessing.preprocessor.__verbose__ import PrintLog


class MultiCollinear:
    # A class that remove uninformative features by multicorrelation analysis.
    # Processing preferences should be defined in Data.
    # Typical usage example:
    # MultiCollinear.remove_uninformative_features(data)

    """
    Class for the deletion of non-informative features based on the multicorrelation analysis

    Typical usage example:
    MultiCollinear.remove_uninformative_features(data).

    :param crosstab: cross table for calculation of features informativity
    :type crosstab: numpy.array

    :param stat: test statistics
    :type stat: numpy.float

    :param obs: number of observations
    :type obs: numpy.int

    :param mini: minimum value between the columns and the rows of the cross table
    :type mini: int

    :param multicolinear_features: data structure to save multicolinear features
    :type multicolinear_features: set

    :return: stat / (obs * mini)
    :rtype: numpy.float
    """

    @staticmethod
    def remove_uninformative_features(data: Data_Structure.Data) -> None:
        """
        Method that remove uninformative features

        :param use_spearman: if True than Spearman correlation is used (recommended if type of data is unknown)
        :type use_spearman: bool

        :param use_cramerV: if True than Cramer correlation is used (recommended for categorical data)
        :type use_cramerV: bool

        :param use_pearson: if True than Pearson correlation is used (recommend for numerical data)
        :type use_pearson: bool

        :param multicolinear: if True than decision is made in accordance with thresholds
        :type multicolinear: numpy.bool
        """

        def __cramers_V__(var1, var2):
            # Method that calculates informativity between categorical values

            # Cross table building
            crosstab = np.array(pd.crosstab(var1, var2, rownames=None, colnames=None))

            # Keeping of the test statistic of the Chi2 test
            stat = scipy.stats.chi2_contingency(crosstab)[0]

            # Number of observations
            obs = np.sum(crosstab)

            # Take the minimum value between the columns and the rows of the cross table
            mini = min(crosstab.shape) - 1

            return stat / (obs * mini)

        PrintLog.instance().info("START MULTICOLINEARITY ANALYSIS")

        # Transpose features and labels to get columns
        # - don't forget to transpose back after analysis
        data.transpose()

        # Remember milticolinear features
        multicolinear_features = set()

        # Find multilocinear features
        # select 1st feature that is not in delete list
        for first_idx, first_vector in enumerate(data.features_matrix):
            if first_idx not in multicolinear_features:
                # Select 2nd feature, that is later that 1st feature and that is not in delete list
                for second_idx, second_vector in enumerate(data.features_matrix):
                    if second_idx > first_idx and second_idx not in multicolinear_features:
                        # Select correlation type
                        # use Spearman corr. if we dont know data types or they are categorial & numerical
                        # use CramerV corr. if data types are both categorial
                        # use Pearson corr. if data types are both numerical

                        use_spearman = (
                                               data.features_types[first_idx] is None or
                                               data.features_types[second_idx] is None
                                       ) or \
                                       (
                                               data.features_types[first_idx] != data.features_types[second_idx]
                                       )
                        use_cramerV = not use_spearman \
                                      and data.features_types[first_idx] == data.features_types[second_idx] == "cat"
                        use_pearson = not use_spearman \
                                      and data.features_types[first_idx] == data.features_types[second_idx] == "num"

                        multicolinear = None

                        if use_spearman:
                            multicolinear = 1-scipy.stats.spearmanr(first_vector, second_vector)[1] \
                                            > data.thresholds_multicolinear["num_cat"]
                        if use_cramerV:
                            multicolinear = __cramers_V__(first_vector, second_vector) \
                                            > data.thresholds_multicolinear["cat_cat"]
                        if use_pearson:
                            multicolinear = 1-scipy.stats.pearsonr(first_vector, second_vector)[1] \
                                            > data.thresholds_multicolinear["num_num"]

                        # Raise exceprion if there is unknown feature type
                        if multicolinear is None:
                            raise Exception("wrong features types=", data.features_types)

                        # Make decision by multicolinear threshold and add milticolinear feature to delete list
                        if multicolinear:
                            PrintLog.instance().status((
                                "delete feature=", data.features_names[second_idx],
                                " as it correlates with feature=", data.features_names[first_idx]
                            ))
                            multicolinear_features.add(second_idx)

        # Transpose data back to normal state and delete uninformative features
        PrintLog.instance().info(
            (
                "REMOVE MILTICOLINEAR FEATURES: [",
                ", ".join([data.features_names[i] for i in multicolinear_features]),
                "]"
            ) if len(multicolinear_features) > 0
            else "NO MULTICOLINEAR FEATURES"
        )
        data.transpose()
        data.delete_features_by_idx(list(multicolinear_features))

        if len(data.features_matrix[0]) == 0:
            PrintLog.instance().warn("ALL FEATURES WERE DELETED - CHECK THE SETTINGS")
            sys.exit(0)
