import os
import pandas as pd
import numpy as np

class DataLoaderAndPreprocessor:
    """
    Class for loading and preprocessing data.

    :param DATA_PATH: Path to th e directory with datasets (string).
    :param drop_features: Set of features to remove from the data (list).
    :param categorical_features: A set of categorical features in the data (list).
    :param data: Data feature matrix (pd.DataFrame).
    """
    def __init__(self, dataset_name, mode=1, suf=''):
        """
        Initializing.

        :param dataset_name: Data set name (srting).
        :param mode: Boot mode, for developers (integer).
        """

        DATA_PATH = "../datasets/"

        self.features_names = []
        self.drop_features = []
        self.categorical_features = []
        self.data = None

        if not os.path.exists('apssop_models/'):
            os.makedirs('apssop_models/')

        self.forecasting_model_path = 'apssop_models/forecasting_model_' + dataset_name + suf
        self.normalization_model_path = 'apssop_models/scaler_' + dataset_name + suf + '.pkl'

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
        Data feature preprocessing.
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
        Split data into training and test sets.

        :param train_size: Share of the training sample (float). Default=0.9.
        :return: train: Feature matrix of training data (pd.DataFrame).
        :return test: Feature matrix of test data (pd.DataFrame).
        """
        if (train_size < 0) and (train_size > 1):
            print('The proportion of the training sample is not in the interval (0, 1)')
            return None, None

        train_ind = round(train_size*self.data.shape[0])

        train = self.data.iloc[:train_ind]
        test = self.data.iloc[train_ind:]
        return train, test