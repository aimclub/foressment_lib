import os
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import configparser
from foressment_ai import RulesExtractor

class DataLoaderAndPreprocessorDefault:
    """
    Class for loading and preprocessing data

    :param DATA_PATH: Path to th e directory with datasets
    :type DATA_PATH: string

    :param drop_features: Set of features to remove from the data
    :type drop_features: list

    :param categorical_features: A set of categorical features in the data
    :type categorical_features: list

    :param data: Data feature matrix
    :type data: pandas.DataFrame
    """
    def __init__(self, dataset_name, nrows=None, suf='', data_configs_path="../datasets/configs", label_format='binary'):
        """
        Initializing

        :param dataset_name: Data set name
        :type dataset_name: string

        :param mode: Boot mode, for developers
        :type mode: int
        """

        self.dataset_name = dataset_name

        self.__load_data__(data_configs_path, label_format, nrows)

        # if not os.path.exists('forecaster_models/'):
        #     os.makedirs('forecaster_models/')

        # self.forecasting_model_path = 'forecaster_models/forecasting_model_' + dataset_name + suf
        self.normalization_model_path = 'forecaster_models/scaler_' + dataset_name + suf + '.pkl'

    def __load_data__(self, data_configs_path, label_format, nrows):
        """
        Import data with configuration file.
        """
        config = configparser.ConfigParser()

        try:
            config.read(data_configs_path + '/' + self.dataset_name + '.ini')
        except:
            print('Unknown dataset name!')
            exit()

        dataset_path = config.get('Params', 'data_path')
        dataset_filename = config.get('Params', 'filename',  fallback='')
        sep = config.get('Params', 'sep',  fallback=',')
        decimal = config.get('Params', 'decimal', fallback='.')

        if dataset_filename:
            self.data = pd.read_csv(dataset_path + '/' + dataset_filename, sep=sep, decimal=decimal, nrows=nrows)
        else:
            for file in os.listdir(dataset_path):
                self.data = pd.concat([self.data, pd.read_csv(dataset_path  + '/' + file, sep=sep, decimal=decimal)],
                                      ignore_index=True)
                if nrows:
                    if self.data.shape[0] > nrows:
                        self.data = self.data.loc[:nrows]
                        break

        self.data.columns = [c.strip() for c in self.data.columns]

        self.timestamp_feature = config.get('Features', 'timestamp_feature', fallback=None)
        self.binaries_features = self.__config_get_list__(config, 'binaries_features')
        self.drop_features = self.__config_get_list__(config, 'drop_features')
        self.categorical_features = self.__config_get_list__(config, 'categorical_features')
        self.important_features = self.__config_get_list__(config, 'important_features')

        if self.timestamp_feature:
            self.timestamps = self.data[self.timestamp_feature].squeeze()

            time_format = config.get('Other', 'time_format', fallback='')
            timestamp_unit = config.get('Other', 'timestamp_unit', fallback='')

            if time_format:
                self.timestamps = self.str_data_to_time(self.timestamps,
                                                        time_format=time_format)
            if timestamp_unit:
                self.timestamps = self.float_data_to_time(self.timestamps, timestamp_unit)

            self.data = self.data.drop(self.timestamp_feature, axis=1)

        label_section = 'Labels.' + label_format
        label_config = config[label_section]

        self.label_feature = label_config.get('label', fallback=None)
        normal_label = label_config.get('normal_label', fallback='Normal')
        self.labels = self.data[self.label_feature]
        self.data = self.data.drop(self.label_feature, axis=1)

        if not is_numeric_dtype(self.labels):
            self.__encoding_labels__(normal_label)

        self.drop_features = [f for f in self.drop_features if f in self.data.columns]
        if len(self.drop_features) > 0:
            self.data = self.data.drop(self.drop_features, axis=1)

    @staticmethod
    def __config_get_list__(config, feature_name):
        try:
            return config.get('Features', feature_name).replace('\n', '').split(',')
        except:
            return []

    @staticmethod
    def str_data_to_time(data, time_format):
        '''
        :param data: timestamp column (pd.Series)
        :return:
        '''
        data = data.map(lambda x: x.strip() if not pd.isnull(x) else x)
        data = pd.to_datetime(data, format=time_format)
        return data

    @staticmethod
    def float_data_to_time(data, timestamp_unit):
        '''
        :param data: timestamp column (pd.Series)
        :return:
        '''
        data = pd.to_datetime(data, unit=timestamp_unit)
        return data

    def __encoding_labels__(self, normal_label):
        labels_classes = sorted(self.labels.unique().tolist())
        labels_dictionary = {}

        if normal_label in labels_classes:
            labels_dictionary[normal_label] = 0
            labels_classes.remove(normal_label)

        for i, label in enumerate(labels_classes):
            labels_dictionary[label] = i

        self.labels = self.labels.map(lambda x: labels_dictionary[x])

    def categorical_feature_encoding(self):
        """
        Data categorical feature encoding
        """
        if len(self.categorical_features) > 0:
            new_categorical_features = []
            for feature in self.categorical_features:
                try:
                    self.data[feature] = self.data[feature].astype(str)
                    categorical_feature_values = self.data[feature].unique().tolist()
                    if ('0.0' in categorical_feature_values) and ('0' in categorical_feature_values):
                        self.data[feature] = self.data[feature].map(lambda x: '0'if x == '0.0'else x)
                    encoded_feature_data = pd.get_dummies(self.data[feature])
                    for col in encoded_feature_data.columns:
                        encoded_feature_data = encoded_feature_data.rename(columns={col: feature + '_' + col})
                        new_categorical_features.append(feature + '_' + col)
                except:
                    encoded_feature_data = pd.DataFrame()

                if not encoded_feature_data.empty:
                    old_data_columns = self.data.columns.tolist()
                    feature_index = old_data_columns.index(feature)
                    new_data_columns = old_data_columns[:feature_index] + \
                                       encoded_feature_data.columns.tolist() + \
                                       old_data_columns[feature_index+1:]

                    self.data = pd.concat([self.data, encoded_feature_data], axis=1)
                    self.data = self.data[new_data_columns]
                else:
                    print('Too many values for categorical feature "' + feature +'". Delete feature from data')
                    self.data = self.data.drop(feature, axis=1)
            self.categorical_features = new_categorical_features
        self.data = self.data.fillna(0)

    def train_test_split(self, train_size=0.9):
        """
        Split data into training and test sets

        :param train_size: Share of the training sample (default = 0.9)
        :type train_size: float

        :return train: Feature matrix of training data
        :rtype train: pandas.DataFrame

        :return test: Feature matrix of test data
        :rtype test: pandas.DataFrame
        """
        if (train_size < 0) and (train_size > 1):
            print('The proportion of the training sample is not in the interval (0, 1)')
            return None, None

        train_ind = round(train_size*self.data.shape[0])

        train = self.data.iloc[:train_ind]
        test = self.data.iloc[train_ind:]
        return train, test


class DataLoaderAndPreprocessorExtractor:
    """
    Class for loading and preprocessing data.

    :param DATA_PATH: Path to th e directory with datasets
    :type DATA_PATH: string

    :param drop_features: Set of features to remove from the data
    :type drop_features: list

    :param categorical_features: A set of categorical features in the data
    :type categorical_features: list

    :param data: Data feature matrix
    :type data: pandas.DataFrame
    """

    def __init__(self, test_name):
        """
        Initializing

        :param dataset_name: Data set name
        :type dataset_name: str

        :param use_extractor: use preprocessing with Extractor module or not
        :type use_extractor: bool

        :param mode: Boot mode, for developers
        :type mode: int
        """

        self.DATA_PATH = "../datasets/"
        self.dataset_name = ''
        self.suf = test_name

        self.features_names = []
        self.drop_features = []
        self.categorical_features = []
        self.data = None

    def __call__(self, dataset_name, use_extractor=False, mode=1):

        self.dataset_name = dataset_name

        if not os.path.exists('../models/' + dataset_name):
            os.makedirs('../models/' + dataset_name)

        self.forecasting_model_path = '../models/' + dataset_name + '/forecaster_model_' + dataset_name + '_' + self.suf
        self.normalization_model_path = '../models/' + dataset_name + '/forecaster_scaler_' + dataset_name + '_' + self.suf + '.pkl'

        if dataset_name == 'smart_crane':
            self.data = pd.read_csv(self.DATA_PATH + 'IEEE_smart_crane.csv')

            if mode not in range(1, 9):
                print('Wrong cycle number')
                exit()
            self.data = self.data[self.data['Cycle'] == mode]

            self.drop_features = ['Date']
            self.drop_labels = ['Alarm', 'Cycle']
            self.delete_features()

            if use_extractor:
                self.extractor(class_column='Alarm')

            self.delete_labels()
            self.features_names = self.data.columns.values
            return self.data

        elif dataset_name == 'hai':
            self.data = pd.read_csv(self.DATA_PATH + 'HAI_test2.csv.zip')
            self.drop_features = ['timestamp']
            self.drop_labels = ['Attack']
            self.delete_features()

            if use_extractor:
                self.extractor(class_column='Attack')

            self.delete_labels()
            self.features_names = self.data.columns.values
            return self.data

        else:
            print('Unknown dataset name')
            exit()

    def extractor(self, class_column, positive_class_label=1):
        algo = RulesExtractor(0.1)
        algo.fit(self.data, class_column=class_column,
                 positive_class_label=positive_class_label)
        rules = algo.get_rules()
        self.data = algo.transform(self.data)

    def delete_features(self):
        if len(self.drop_features) > 0:
            self.data = self.data.drop(self.drop_features, axis=1)

    def delete_labels(self):
        if len(self.drop_labels) > 0:
            try:
                self.data = self.data.drop(self.drop_labels, axis=1)
            except:
                pass

    def categorical_features_encoding(self):
        """
        Data feature preprocessing
        """
        if len(self.categorical_features) > 0:
            for feature in self.categorical_features:
                if 'ip' in feature:
                    encoded_feature_data = pd.DataFrame([x.split('.')
                                                         if x != '0' else [0, 0, 0, 0]
                                                         for x in self.data[feature].tolist()])

                    for col in encoded_feature_data.columns:
                        encoded_feature_data = encoded_feature_data.rename(columns={col: feature + '_' + str(col)})
                else:
                    try:
                        encoded_feature_data = pd.get_dummies(self.data[feature])
                        for col in encoded_feature_data.columns:
                            encoded_feature_data = encoded_feature_data.rename(columns={col: feature + '_' + col})
                    except:
                        encoded_feature_data = pd.DataFrame()

                if not encoded_feature_data.empty:
                    old_data_columns = self.data.columns.tolist()
                    feature_index = old_data_columns.index(feature)
                    new_data_columns = old_data_columns[:feature_index] + \
                                       encoded_feature_data.columns.tolist() + \
                                       old_data_columns[feature_index + 1:]

                    self.data = pd.concat([self.data, encoded_feature_data], axis=1)
                    self.data = self.data[new_data_columns]
                else:
                    print('Too many values for categorical feature "' + feature + '". Delete feature from data')
                    self.data = self.data.drop(feature, axis=1)
        self.data = self.data.fillna(0)

