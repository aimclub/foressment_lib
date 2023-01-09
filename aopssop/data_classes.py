from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class AopssopData:
    """
    Data class for AOPSSOP library

    :param features_matrix: matrix of features, array with size of O×N, where O – number of objects, N – number of features
    :type features_matrix: np.array

    :param features_names: vector of features' names, array with size of N
    :type features_names: np.array

    :param features_types: vector of features' types, array with size of N
    :type features_types: np.array

    :param labels_matrix: matrix of labels, array with size of O×M, where O – number of objects, M – number of labels
    :type labels_matrix: np.array

    :param labels_names: vector of labels' names, array with size of N
    :type labels_names: np.array

    :param labels_types: vector of labels types, array with size of M
    :type labels_types: np.array

    :param empty_coordinates: vector of empty values coordinates, array with size of K, where K – number of empty values
    :type empty_coordinates: np.array

    :param assoc_rules: associative rules, array with size of Q, where Q – number of associative rules
    :type assoc_rules: array

    :param wave_rules: wavelet rules, array with size of U, where U – number of wavelet rules
    :type wave_rules: np.array

    :param assessment_results: results of the state assessment
    :type assessment_results: np.array

    :param assessment_quality: quality parameters of the state assessment
    :type assessment_quality: dict

    :param forecasting_results: results of the state forecasting
    :type forecasting_results: np.array

    :param forecasting_quality: quality parameters of the state forecasting
    :type forecasting_quality: pandas.DataFrame()

    :param correctness_threshold: threshold for data type correctness checking algorithm (default = 0.70)
    :type correctness_threshold: float

    :param clasterization_method: method for the data clasterization during its preprocessing (default = 'K-means')
    :type clasterization_method: str

    :param incompletness_method: method for the elimination of the data incompleteness during its preprocessing (default = 'avg')
    :type incompletness_method: str

    :param informative_threshold: threshold of informativeness of the feature that is used in elimination of the multicolinearity of data during its preprocessing (default = 0.70)
    :type informative_threshold: float

    :param forecasting_model_path: path to the forecasting model (default = '../models/')
    :type forecasting_model_path: str

    :param normalization_model_path: path to the normalizatiob model (default = '../models/')
    :type normalization_model_path: str

    :param time_window_length: size of the time window during training of the model (default = 60)
    :type time_window_length: int

    :param forecasting_time_window: size of the time window during state forecasting (default = 1000)
    :type forecasting_time_window: int
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







