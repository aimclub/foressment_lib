# Алгоритм ПОСНД предназначен для работы с подаваемыми на вход наборами данных
# и включает в себя коррекцию типов данных, а также устранение неполноты и
# мультиколлинеарности данных о сложных объектах (СлО) или процессах.

def posnd(data):
    data += f'\nposnd:'
    data = data_types_correctness_analysis(data)
    data = data_incompleteness_elimination(data)
    data = data_multicollinearity_elimination(data)

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
