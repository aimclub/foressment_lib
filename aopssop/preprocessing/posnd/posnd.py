# Алгоритм ПОСНД предназначен для работы с подаваемыми на вход наборами данных
# и включает в себя коррекцию типов данных, а также устранение неполноты и
# мультиколлинеарности данных о сложных объектах (СлО) или процессах.
from aopssop.data_classes import AopssopData


def posnd(
        features, labels,
        features_types : None, labels_types : None
):
    """
        Preprocessing of the given dataset.

        :param features: matrix of features of the datasets.
        :type features: np.array

        :param labels: matrix of labels of the datasets.
        :type labels: np.array

        :param feature_types: array of data types of features (optional).
        :type feature_types: np.array

        :param label_types: array of data types of labels (optional).
        :type label_types: np.array

        :return: preprocessed features and labels.
        :rtype: np.arrays
    """

    data = AopssopData
    # data.features_matrix = set_features_matrix(features)
    # и т.п.
    data_types_correctness_analysis(data)

    return data


def data_types_correctness_analysis(data):
    data += f'\n- data_types_correctness_analysis'

    return data


def data_incompleteness_elimination(data):
    data += f'\n- data_incompleteness_elimination'

    return data


def data_multicollinearity_elimination(data):
    data += '\n- data_multicollinearity_elimination'

    return data
