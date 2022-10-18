from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class AopssopData:
    # Матрица признаков
    # Массив размерности O×N,
    # где O – количество объектов,
    # N – количество признаков
    # ПОСНД, АПССОП, АОССОП
    features_matrix: np.array([])

    # Вектор типов признаков
    # Массив размерности N
    features_types: np.array([])

    # Матрица лейблов
    # Массив размерности O×M,
    # где O – количество объектов,
    # где M – количество лейблов
    # АОССОП
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







