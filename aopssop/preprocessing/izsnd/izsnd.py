# Алгоритм ИЗСНД предназначен для извлечения фрагментов знаний, имеющихся в данных о СлО,
# в виде ассоциативных правил (вида «ЕСЛИ <посылка>, ТО <следствие>»), содержащих
# в правой части (следствии) метку класса (англ. class association rules).
# В случае, если обрабатывается непрерывная измерительная информация, то в левой части
# (посылке) используются коэффициенты дискретного вейвлет-преобразования, производимого
# с использованием ортогональных базисных вейвлет-функций (Добеши, симлетов, койфлетов и др.).
import random


def izsnd(data):
    if data_type_checker(data):
        result = rvp(data)

    else:
        result = izdap(data)

    return result


def data_type_checker(data):

    return bool(random.getrandbits(1))


def rvp(data):
    result = data + '\n- rvp'

    return result


def izdap(data):
    result = data + '\n- izdap'

    return result
