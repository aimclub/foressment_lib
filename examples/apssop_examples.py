from aopssop import DataScaler, AIForecaster, ForecastEstimator
from aopssop import AopssopData as PDATA
import pandas as pd
import numpy as np
import os
import sys
import time
import psutil

import math
import matplotlib.pyplot as plt

pid = os.getpid()
proc = psutil.Process(pid)
proc.as_dict()


class DataLoaderAndPreprocessor:
    """
    Class for loading and preprocessing data

    :param DATA_PATH: Path to th e directory with datasets (string).
    :param drop_features: Set of features to remove from the data (list).
    :param categorical_features: A set of categorical features in the data (list).
    :param data: Data feature matrix (pd.DataFrame).
    """
    def __init__(self, dataset_name, mode=1, suf=''):
        """
        Initializing

        :param dataset_name: Data set name (srting).
        :param mode:Boot mode, for developers (integer).
        """

        DATA_PATH = __file__.replace('examples/_examples.py', '') + 'datasets/'

        self.features_names = []
        self.drop_features = []
        self.categorical_features = []
        self.data = None

        self.forecasting_model_path = ''
        self.normalization_model_path = ''

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

            self.forecasting_model_path = 'models/forecasting_model_' + dataset_name + suf
            self.normalization_model_path = 'models/scaler_' + dataset_name + suf + '.pkl'


        # elif dataset_name == 'alarms':
        #     DATA_PATH = DATA_PATH + 'IEEE_alarms_log_data/raw'
        #     if mode == 'simple_test':
        #         self.data = pd.read_csv(DATA_PATH + '/alarms.csv', nrows=10000)
        #     else:
        #         self.data = pd.read_csv(DATA_PATH + '/alarms.csv')
        #     self.drop_features = ['timestamp']
        #     self.preprocessing()
        #     self.features_names = self.data.columns.values

        # elif dataset_name == 'edge-iiotset':
        #     DATA_PATH = DATA_PATH + 'Edge-IIoTset/'
        #     self.data = pd.read_csv(DATA_PATH + 'Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv',
        #                             low_memory=False)
        #
        #     self.data['mqtt.conack.flags'] = self.data['mqtt.conack.flags'].map(lambda x: 0 if x == '0x00000000' else x)
        #     self.data['mqtt.conack.flags'] = self.data['mqtt.conack.flags'].astype(float).round()
        #
        #     self.data['mqtt.protoname'] = self.data['mqtt.protoname'].map(lambda x: '0' if x == '0.0' else x)
        #     self.data['mqtt.topic'] = self.data['mqtt.topic'].map(lambda x: '0' if x == '0.0' else x)
        #
        #     self.drop_features = ['frame.time', 'Attack_label', 'Attack_type',
        #                          'tcp.options', 'tcp.payload',
        #                           'http.file_data', 'http.request.uri.query', 'http.request.method',
        #                           'http.referer', 'http.request.full_uri', 'http.request.version',
        #                           'tcp.srcport', 'dns.qry.name.len',
        #                           'mqtt.msg']
        #
        #     self.categorical_features = ['ip.src_host', 'ip.dst_host',
        #                                  'arp.dst.proto_ipv4', 'arp.src.proto_ipv4',
        #                                   'mqtt.protoname', 'mqtt.topic']
        #     self.preprocessing()
        #     self.features_names = self.data.columns.values

        elif dataset_name == 'smart_crane':
            self.data = pd.read_csv(DATA_PATH + 'IEEE_smart_crane.csv')
            if mode not in range(1,9):
                print('Wrong cycle number')
                exit()
            self.data = self.data[self.data['Cycle'] == mode]

            self.drop_features = ['Date', 'Alarm', 'Cycle']
            self.preprocessing()
            self.features_names = self.data.columns.values

            self.forecasting_model_path = 'models/forecasting_model_' + dataset_name + suf + '_c' + str(mode)
            self.normalization_model_path = 'models/scaler_' + dataset_name + suf + '_c' + str(mode) + '.pkl'

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
                    encoded_feature_data = pd.DataFrame([x.split('.') if x != '0' else [0, 0, 0, 0]
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


class Logger:
    def __init__(self, filename, rewrite=False):
        self.filename = filename
        self.log = None

        if rewrite:
            self.log = self.open('w')
        else:
            self.log = self.open('a')

        self.start_time = None
        self.text = ''

    def start_line(self, text=None, wait=True):
        self.start_time = time.time()
        if text:
            self.text = text

        if not wait:
            self.end_line()

    def end_line(self, text=None, show=True):
        if text:
            self.text = text

        if not self.start_time:
            duration = '??? sec.'
        else:
            duration = str(round(time.time() - self.start_time, 2)) + ' sec.'

        d = proc.as_dict(attrs=['cpu_percent', 'memory_info', 'memory_percent'])
        cpu_num = psutil.cpu_count()

        cpu_percent = 'CPU ' + str(round(d['cpu_percent']/cpu_num, 2)) + '%'
        memory = 'RAM ' + str(round(d['memory_info'].rss / 1024 ** 2, 2)) + 'Mb'

        linetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        text = '|'.join([linetime, self.text, duration, cpu_percent, memory]) + '\n'
        self.log.write(text)
        self.log.flush()
        os.fsync(self.log.fileno())

        if show:
            print(text[:-1])

    def open(self, how):
        return open(self.filename, how)

    def close(self):
        self.log.close()

    def parse_to_dataframe(self):
        pass


def test_appsop_example_forecast_all_test_seconds(dataset_name, suf='', mode=1):
    """
    Testing APPSOP module methods

    In this example, all samples in the test set are forecasted based on the last batch of the training set.
    Note: In such an experiment, the accumulation of prediction errors is possible.
    """

    PDATA.time_window_length = 10

    LOG = Logger('results/' + dataset_name + suf + '.log')
    LOG.start_line('Start with dataset "' + dataset_name + '"', wait=False)

    LOG.start_line()
    data = DataLoaderAndPreprocessor(dataset_name, mode=mode, suf=suf)
    PDATA.features_names = data.features_names
    LOG.end_line('Input data preprocessed. Shape ({0}, {1})'.format(data.data.shape[0], data.data.shape[1]))

    PDATA.forecasting_model_path = data.forecasting_model_path
    PDATA.normalization_model_path =  data.normalization_model_path

    LOG.start_line()
    train, test = data.train_test_split(train_size=0.1)
    LOG.end_line('Data is divided into train and test samples of length {0} and {1}'.format(len(train), len(test)))

    LOG.start_line()
    normalization_model = DataScaler(scaler_path=PDATA.normalization_model_path,
                                     open=True
                                     )
    normalization_model.fit(train)
    LOG.end_line('Data normalized')

    scaled_train = normalization_model.transform(train)
    scaled_test = normalization_model.transform(test)

    forecasting_model = AIForecaster(n_epochs=3,
                                     time_window_length=PDATA.time_window_length,
                                     n_features=len(PDATA.features_names),
                                     model_path=PDATA.forecasting_model_path,
                                     # open=True
                                     )

    LOG.start_line('Forecasting model:' + forecasting_model.model_path, wait=False)

    LOG.start_line()
    train_generator = forecasting_model.data_to_generator(scaled_train)
    LOG.end_line('Data generator created')

    LOG.start_line()
    loss = forecasting_model.train(train_generator)
    LOG.end_line('Training completed (loss:' + str(loss) + ')')

    PDATA.forecasting_time_window = len(scaled_test)
    last_batch = forecasting_model.get_batch(train_generator, -1)

    LOG.start_line()
    pred = forecasting_model.forecasting(last_batch,
                                         forecasting_data_length=PDATA.forecasting_time_window)
    PDATA.forecasting_results = normalization_model.inverse(pred)
    LOG.end_line('Forecasting completed')

    LOG.start_line()
    estimator = ForecastEstimator()
    PDATA.forecasting_quality = estimator.estimate(true=scaled_test, pred=pred,
                                                             feature_names=PDATA.features_names)

    estimator.save(file_name=dataset_name + suf + '_all_tests')
    LOG.end_line('Evaluation done')

    print(PDATA.forecasting_quality)
    print('Done')


def test_appsop_example_forecast_by_parts(dataset_name, suf = '', mode=1):
    """
    Testing APPSOP module methods

    In this example, test set samples are predicted one at a time based on incoming test set data.
    """

    PDATA.time_window_length = 10

    LOG = Logger('results/' + dataset_name + suf + '.log')
    LOG.start_line('Start with dataset "' + dataset_name + '"', wait=False)

    LOG.start_line()
    data = DataLoaderAndPreprocessor(dataset_name, mode=mode, suf=suf)
    PDATA.features_names = data.features_names
    LOG.end_line('Inpit data preprocessed. Shape ({0}, {1})'.format(data.data.shape[0], data.data.shape[1]))

    PDATA.forecasting_model_path = data.forecasting_model_path
    PDATA.normalization_model_path = data.normalization_model_path

    LOG.start_line()
    train, test = data.train_test_split(train_size=0.1)
    LOG.end_line('Data is divided into train and test samples of length {0} and {1}'.format(len(train), len(test)))

    LOG.start_line()
    normalization_model = DataScaler(scaler_path=PDATA.normalization_model_path,
                                     open=True
                                     )
    scaled_train = normalization_model.transform(train)
    scaled_test = normalization_model.transform(test)
    LOG.end_line('Data normalized')

    forecasting_model = AIForecaster(time_window_length=PDATA.time_window_length,
                                     n_features=len(PDATA.features_names),
                                     model_path=PDATA.forecasting_model_path,
                                     open=True
                                     )

    LOG.start_line('Forecasting model:' + forecasting_model.model_path, wait=False)

    LOG.start_line()
    train_generator = forecasting_model.data_to_generator(scaled_train)
    test_generator = forecasting_model.data_to_generator(scaled_test)
    LOG.end_line('Data generator created')

    PDATA.forecasting_time_window = len(scaled_test)
    current_batch = forecasting_model.get_batch(train_generator, -1)

    predictions = []
    for i in range(PDATA.forecasting_time_window):
        sys.stdout.write('\r\x1b[K' + 'Forecasting: {0}/{1}'.format(i, PDATA.forecasting_time_window - 1))
        sys.stdout.flush()
        LOG.start_line()

        current_pred = forecasting_model.forecasting(current_batch,
                                                     forecasting_data_length=1,
                                                     verbose=False)
        predictions.append(current_pred[0])
        new_event = scaled_test[i]
        current_batch = np.append(current_batch[:, 1:, :], [[new_event]], axis=1)

        LOG.end_line('Forecasting: {0}/{1}'.format(i, PDATA.forecasting_time_window - 1),
                     show=False)

    predictions = pd.DataFrame(predictions).values
    PDATA.forecasting_results = normalization_model.inverse(predictions)

    LOG.start_line('Forecasting complited', wait=False)
    estimator = ForecastEstimator()

    LOG.start_line()
    PDATA.forecasting_quality = estimator.estimate(true=scaled_test, pred=predictions,
                                                   feature_names=PDATA.features_names)

    estimator.save(file_name=dataset_name + suf + '_by_parts')
    LOG.end_line('Evaluation done')

    print(PDATA.forecasting_quality)

    print('Done')


if __name__ == '__main__':
    suf = '_ex1'

    dataset_name = 'smart_crane'
    for mode in range(1,9):
        test_appsop_example_forecast_all_test_seconds(dataset_name, suf, mode)
        test_appsop_example_forecast_by_parts(dataset_name, suf, mode)

    dataset_name = 'hai'
    test_appsop_example_forecast_all_test_seconds(dataset_name, suf)
    test_appsop_example_forecast_by_parts(dataset_name, suf)