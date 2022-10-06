# Алгоритм АОССОП предназначен для автономного оценивания текущего состояния сложных объектов
# и процессов, включающего наличие или отсутствие в текущий момент определенного вида
# функциональных неисправностей, дефектов и атакующих воздействий, свойственных целевой системе

import  subprocess
import sys

def aossop(data):
    data += '\naossop:'
    # just to push smth

    subprocess.Popen(['python3', 'classify_records.py', '-f', sys.argv[1], '-d', sys.argv[2]])

    return data
