from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class AopssopData:
    """
    Data class for AOPSSOP library

    :features_matrix: matrix of features, array with size of O×N, where O – number of objects, N – number of features (np.array).
    :features_names: vector of features' names, array with size of N (np.array).
    :features_types: vector of features' types, array with size of N (np.array).
    :labels_matrix: matrix of labels, array with size of O×M, where O – number of objects, M – number of labels (np.array).
    :labels_names: vector of labels' names, array with size of N (np.array).
    :labels_types: vector of labels types, array with size of M (np.array).
    :empty_coordinates: vector of empty values coordinates, array with size of K, where K – number of empty values (np.array).
    :assoc_rules: associative rules, array with size of Q, where Q – number of associative rules (array).
    :wave_rules: wavelet rules, array with size of U, where U – number of wavelet rules (np.array).
    :assessment_results: results of the state assessment (np.array).
    :assessment_quality: quality parameters of the state assessment (dict).
    :forecasting_results: results of the state forecasting (np.array).
    :forecasting_quality: quality parameters of the state forecasting (pd.DataFrame()).
    :correctness_threshold: threshold for data type correctness checking algorithm. Default = 0.70 (float).
    :clasterization_method: method for the data clasterization during its preprocessing. Default = 'K-means' (str).
    :incompletness_method: method for the elimination of the data incompleteness during its preprocessing. Default = 'avg' (str).
    :informative_threshold: threshold of informativeness of the feature that is used in elimination of the multicolinearity of data during its preprocessing. Default = 0.70 (float).
    :forecasting_model_path: path to the forecasting model. Default = 'models/default_forecasting_model' (str).
    :normalization_model_path: path to the normalizatiob model. Default = 'models/default_normalization_model' (str).
    :time_window_length: size of the time window during training of the model. Default = 60 (int).
    :forecasting_time_window: size of the time window during state forecasting. Default = 1000 (int).
    """
    features_matrix: np.array([])
    features_names: np.array([])
    features_types: np.array([])
    labels_matrix: np.array([])
    labels_names: np.array([])
    labels_types: np.array([])
    empty_coordinates: np.array([])
    assoc_rules: []
    wave_rules: np.array([])
    assessment_results: np.array([])
    assessment_quality: {}
    forecasting_results: np.array([])
    forecasting_quality: pd.DataFrame()
    correctness_threshold: float = 0.70
    clasterization_method: str = "K-means"
    incompletness_method: str = "avg"
    informative_threshold: float = 0.70
    forecasting_model_path: str = "../models/"
    normalization_model_path: str = "../models/"
    time_window_length: int = 60
    forecasting_time_window: int = 1000







