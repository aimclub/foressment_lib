# Алгоритм ПОСНД предназначен для работы с подаваемыми на вход наборами данных
# и включает в себя коррекцию типов данных, а также устранение неполноты и
# мультиколлинеарности данных о сложных объектах (СлО) или процессах.
from aopssop.data_classes import AopssopData


def posnd(
        features, labels,
        features_types : None, labels_types : None
):
    data = AopssopData
    data.features_matrix = set_features_matrix(features)

    return output_data


def data_types_correctness_analysis(data):
    data += f'\n- data_types_correctness_analysis'

    return data


def data_incompleteness_elimination(data):
    data += f'\n- data_incompleteness_elimination'

    return data


def data_multicollinearity_elimination(data):
    data += '\n- data_multicollinearity_elimination'

    return data
