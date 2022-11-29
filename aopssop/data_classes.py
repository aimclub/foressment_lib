from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class AopssopData:
    """
    Data class for AOPSSOP library

    :features_matrix: матрица признаков, массив размерности O×N, где O – количество объектов, N – количество признаков (np.array).
    :features_names: вектор названий фич, массив размерности N (np.array).
    :features_types: вектор типов признаков, массив размерности N (np.array).
    :labels_matrix: матрица лейблов, массив размерности O×M, где O – количество объектов, M – количество лейблов (np.array).
    :labels_names: вектор названий лейблов, массив размерности N (np.array).
    :labels_types: вектор типов лейблов, массив размерности M (np.array).
    :empty_coordinates: вектор координат пустых значений, массив размерности K, где K – количество пустых значений (np.array).
    :assoc_rules: множество ассоциативных правил, массив размерности Q, где Q – количество ассоциативных правил (array).
    :wave_rules: база правил на вейвлетах, массив размерности U, где U – количество правил на вейвлетах (np.array).
    :assessment_results: результаты оценки состояния (np.array).
    :assessment_quality: показатели качества оценки (dict).
    :forecasting_results: результаты прогнозирования состояния (np.array).
    :forecasting_quality: показатели качества прогнозирования (pd.DataFrame()).
    :correctness_threshold: порог корректности. Default = 0.70 (float).
    :clasterization_method: метод кластеризации. Default = 'K-means' (str).
    :incompletness_method: метод устранения неполноты. Default = 'avg' (str).
    :informative_threshold: порог информативности. Default = 0.70 (float).
    :forecasting_model_path: путь к модели прогнозирования. Default = 'models/default_forecasting_model' (str).
    :normalization_model_path: путь к модели нормализации. Default = 'models/default_normalization_model' (str).
    :time_window_length: размер временного окна при обучении. Default = 60 (int).
    :forecasting_time_window: размер временного окна при прогнозировании. Default = 1000 (int).
    """
    # Матрица признаков
    # Массив размерности O×N,
    # где O – количество объектов,
    # N – количество признаков
    # ПОСНД, АПССОП, АОССОП, ИЗДАП
    features_matrix: np.array([])

    # Вектор названий фич
    # Массив размерности N
    features_names: np.array([])

    # Вектор типов признаков
    # Массив размерности N
    features_types: np.array([])

    # Матрица лейблов
    # Массив размерности O×M,
    # где O – количество объектов,
    # где M – количество лейблов
    # АОССОП, ИЗДАП
    labels_matrix: np.array([])

    # Вектор названий лейблов
    # Массив размерности N
    labels_names: np.array([])

    # Вектор типов лейблов
    # Массив размерности M
    labels_types: np.array([])

    # Вектор координат пустых значений
    # Массив размерности K,
    # где K – количество пустых значений
    empty_coordinates: np.array([])

    # ИЗДАП: множество ассоциативных правил
    # Массив размерности Q, где Q – количество ассоциативных правил
    assoc_rules: []

    # База правил на вейвлетах
    # Массив размерности U, где U – количество правил на вейвлетах
    wave_rules: np.array([])

    # АОССОП: результаты оценки состояния
    assessment_results: np.array([])

    # АОССОП: показатели качества оценки
    assessment_quality: {}

    # АПССОП: результаты прогнозирования состояния
    forecasting_results: np.array([])

    # АПССОП: показатели качества прогнозирования
    forecasting_quality: pd.DataFrame()

    # -- ПОСНД --
    # порог корректности
    correctness_threshold: float = 0.70

    # метод кластеризации
    clasterization_method: str = "K-means"

    # метод устранения неполноты
    incompletness_method: str = "avg"

    # порог информативности
    informative_threshold: float = 0.70
    # -- ПОСНД --

    # -- АПССОП --
    # путь к модели прогнозирования
    forecasting_model_path: str = "models/default_forecasting_model"

    # путь к модели нормализации
    normalization_model_path: str = "models/default_normalization_model"

    # размер временного окна при обучении
    time_window_length: int = 60

    # размер временного окна при прогнозировании
    forecasting_time_window: int = 1000
    # -- АПССОП --







