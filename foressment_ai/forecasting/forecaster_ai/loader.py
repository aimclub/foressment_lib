import os
import pandas as pd
import numpy as np
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
    def __init__(self, dataset_name, mode=1, suf=''):
        """
        Initializing

        :param dataset_name: Data set name
        :type dataset_name: srting

        :param mode: Boot mode, for developers
        :type mode: int
        """

        DATA_PATH = "../datasets/"

        self.features_names = []
        self.drop_features = []
        self.categorical_features = []
        self.data = None

        if not os.path.exists('forecaster_models/'):
            os.makedirs('forecaster_models/')

        self.forecasting_model_path = 'forecaster_models/forecasting_model_' + dataset_name + suf
        self.normalization_model_path = 'forecaster_models/scaler_' + dataset_name + suf + '.pkl'

        if dataset_name == 'hai':
            DATA_PATH = DATA_PATH + 'hai-22.04/train'

            if mode == 0:
                self.data = pd.read_csv(DATA_PATH + '/train1.csv', nrows=10000)

            else:
                self.data = pd.DataFrame()
                for file in os.listdir(DATA_PATH):
                    self.data = pd.concat([self.data, pd.read_csv(DATA_PATH + '/' + file)],
                                           ignore_index=True)

            self.drop_features = ['timestamp', 'Attack']
            self.preprocessing()
            self.features_names = self.data.columns.values

        elif dataset_name == 'smart_crane':
            self.data = pd.read_csv(DATA_PATH + 'IEEE_smart_crane.csv')

            if mode not in range(1, 9):
                print('Wrong cycle number')
                exit()
            self.data = self.data[self.data['Cycle'] == mode]

            self.drop_features = ['Date', 'Alarm', 'Cycle']
            self.preprocessing()
            self.features_names = self.data.columns.values

        elif dataset_name == 'test':
            self.data = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)),
                                     columns=list('ABCD'))
            self.features_names = self.data.columns.values

        else:
            print('Unknown dataset name')
            exit()

    def preprocessing(self):
        """
        Data feature preprocessing
        """
        if len(self.drop_features) > 0:
            self.data = self.data.drop(self.drop_features, axis=1)

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
                                       old_data_columns[feature_index+1:]

                    self.data = pd.concat([self.data, encoded_feature_data], axis=1)
                    self.data = self.data[new_data_columns]
                else:
                    print('Too many values for categorical feature "' + feature +'". Delete feature from data')
                    self.data = self.data.drop(feature, axis=1)
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

