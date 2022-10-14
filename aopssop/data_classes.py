from dataclasses import dataclass
import numpy as np


@dataclass
class AopssopData:
    # Матрица признаков
    # Массив размерности O×N,
    # где O – количество объектов,
    # N – количество признаков
    features_matrix: np.array([])

    # Вектор типов признаков
    # Массив размерности N
    features_types: np.array([])

    # Матрица лейблов
    # Массив размерности O×M,
    # где O – количество объектов,
    # где M – количество лейблов
    labels_matrix: np.array([])

    # Вектор имен лейблов
    # Массив размерности M
    labels_types: np.array([])

    # Вектор координат пустых значений
    # Массив размерности K,
    # где K – количество пустых значений
    empty_coordinates: np.array([])

    # База ассоциативных правил
    # Массив размерности Q, где Q – количество ассоциативных правил
    assoc_rules: np.array([])

    # База правил на вейвлетах
    # Массив размерности U, где U – количество правил на вейвлетах
    wave_rules: np.array([])

    # Результаты оценки состояния
    assessment_results: np.array([])

    # Результаты прогнозирования состояния
    forecasting_results: np.array([])


