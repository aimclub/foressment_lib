import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from aopssop import DataScaler, AIForecaster, ForecastEstimator
from aopssop import IzdapAlgo
from aopssop import AopssopData as PDATA

from threading import Thread
import psutil
import pandas as pd
import numpy as np

pid = os.getpid()
proc = psutil.Process(pid)
cpu_num = psutil.cpu_count()
proc.as_dict()

from examples.logger import Logger
LOG0 = Logger(proc, cpu_num)


class DataLoaderAndPreprocessor:
    """
    Class for loading and preprocessing data.

    :param DATA_PATH: Path to th e directory with datasets (string).
    :param drop_features: Set of features to remove from the data (list).
    :param categorical_features: A set of categorical features in the data (list).
    :param data: Data feature matrix (pd.DataFrame).
    """
    def __init__(self, test_name):
        """
        Initializing.

        :param dataset_name: Data set name (srting).
        :param use_izdap: use preprocessing with IZDAP module or not (boolean).
        :param mode: Boot mode, for developers (integer).
        """

        self.DATA_PATH = "../datasets/"
        self.dataset_name = ''
        self.suf = test_name

        self.features_names = []
        self.drop_features = []
        self.categorical_features = []
        self.data = None

    def __call__(self, dataset_name, use_izdap=False, mode=1):

        self.dataset_name = dataset_name

        if not os.path.exists('../models/' + dataset_name):
            os.makedirs('../models/' + dataset_name)

        self.forecasting_model_path = '../models/' + dataset_name + '/apssop_model_' + dataset_name + '_' + self.suf
        self.normalization_model_path = '../models/' + dataset_name + '/apssop_scaler_' + dataset_name + '_' + self.suf + '.pkl'

        if dataset_name == 'smart_crane':
            self.data = pd.read_csv(self.DATA_PATH + 'IEEE_smart_crane.csv')

            if mode not in range(1, 9):
                print('Wrong cycle number')
                exit()
            self.data = self.data[self.data['Cycle'] == mode]

            self.drop_features = ['Date']
            self.drop_labels = ['Alarm', 'Cycle']
            self.delete_features()

            if use_izdap:
                algo = IzdapAlgo(0.1)
                algo.fit(self.data, class_column='Alarm', positive_class_label=1)
                rules = algo.get_rules()
                self.data = algo.transform(self.data)

            self.delete_labels()
            self.features_names = self.data.columns.values
            return self.data

        else:
            print('Unknown dataset name')
            exit()

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
        Data feature preprocessing.
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
                                       old_data_columns[feature_index+1:]

                    self.data = pd.concat([self.data, encoded_feature_data], axis=1)
                    self.data = self.data[new_data_columns]
                else:
                    print('Too many values for categorical feature "' + feature +'". Delete feature from data')
                    self.data = self.data.drop(feature, axis=1)
        self.data = self.data.fillna(0)


def train_test_split(data, train_size=0.9):
        """
        Split data into training and test sets.

        :param train_size: Share of the training sample (float). Default=0.9.
        :return: train: Feature matrix of training data (pd.DataFrame).
        :return test: Feature matrix of test data (pd.DataFrame).
        """
        if (train_size < 0) and (train_size > 1):
            print('The proportion of the training sample is not in the interval (0, 1)')
            return None, None

        train_ind = round(train_size*data.shape[0])

        train = data.iloc[:train_ind]
        test = data.iloc[train_ind:]
        return train, test


def example_appsop_model_training(logname='example'):
    """
    Testing APPSOP module methods.
    In this example, forecasting model and  normalization model are trained on the dataset.
    """
    logfile = logname + '_training_res.log'
    result = {'test': logfile, 'data_size': PDATA.features_matrix.shape[0]}

    LOG0.create(logfile, rewrite=True)
    LOG0.run = True
    thread1 = Thread(target=LOG0.daemon_logger)
    thread1.start()

    LOG0.event_init(text='Start with dataset "')

    LOG0.event_init(event_name='preproc', text='Input data splitting')
    train, test = train_test_split(PDATA.features_matrix, train_size=0.9)
    LOG0.event_init(event_name='preproc',
                    text='Data is divided into train and test samples of length {0} and {1}'.format(len(train),
                                                                                                    len(test)))

    LOG0.event_init(event_name='norm', text='Normalization model training')
    normalization_model = DataScaler(scaler_path=PDATA.normalization_model_path)
    normalization_model.fit(train)
    LOG0.event_init(event_name='norm', text='Data normalization')
    scaled_train = normalization_model.transform(train)
    LOG0.event_init(event_name='norm', text='Data normalized')

    LOG0.event_init(event_name='prepare', text='Forecasting model creation:' + PDATA.forecasting_model_path)

    forecasting_model = AIForecaster(n_epochs=3,
                                     time_window_length=PDATA.time_window_length,
                                     n_features=len(PDATA.features_names),
                                     model_path=PDATA.forecasting_model_path,
                                     )

    LOG0.event_init(event_name='prepare', text='Data generator creation')
    train_generator = forecasting_model.data_to_generator(scaled_train)
    LOG0.event_init(event_name='prepare', text='Data generator created')

    LOG0.event_init(event_name='train', text='Model training')
    LOG0.show_off()
    result['mse'] = forecasting_model.train(train_generator)
    LOG0.show_on()
    LOG0.event_init(event_name='train', text='Training completed')

    LOG0.run = False
    thread1.join()
    LOG0.close()

    result.update(LOG0.get_resources(event_name='train'))

    print('Done')
    return result


def example_appsop_forecasting(logname='example', train_size=0.9,
                               apriori=True):
    """
    Testing APPSOP module methods
    In this example, test set samples are predicted one at a time based on incoming test set data.
    """

    if apriori:
        logname = logname + '_apriori'
    else:
        logname = logname + '_aposteriori'

    logfile = logname + '_forecasting_res.log'
    result = {'test': logfile, 'data_size': PDATA.features_matrix.shape[0]}

    LOG0.create(logfile, rewrite=True)
    LOG0.run = True
    thread1 = Thread(target=LOG0.daemon_logger)
    thread1.start()

    LOG0.event_init(text='Start with dataset "')
    LOG0.event_init(event_name='preproc', text='Input data splitting')
    train, test = train_test_split(PDATA.features_matrix, train_size=0.9)
    LOG0.event_init(event_name='preproc',text='Data is divided into train and test samples of length {0} and {1}'.format(len(train),
                                                                                                    len(test)))
    LOG0.event_init(event_name='norm', text='Normalization model open')
    normalization_model = DataScaler(scaler_path=PDATA.normalization_model_path,
                                     open=True)
    if not normalization_model:
        print('Wrong normalization model filename')
        exit()

    LOG0.event_init(event_name='norm', text='Data normalization')
    scaled_train = normalization_model.transform(train)
    scaled_test = normalization_model.transform(test)
    LOG0.event_init(event_name='norm', text='Data normalized')

    LOG0.event_init(event_name='prepare', text='Forecasting model creation:' + PDATA.forecasting_model_path)
    forecasting_model = AIForecaster(model_path=PDATA.forecasting_model_path,
                                     open=True)

    LOG0.event_init(event_name='prepare', text='Data generator creation')
    PDATA.time_window_length = forecasting_model.time_window_length
    train_generator = forecasting_model.data_to_generator(scaled_train)
    LOG0.event_init(event_name='prepare', text='Data generator created')

    predictions = []

    LOG0.event_init(event_name='prepare', text='Get batch for forecasting')
    PDATA.forecasting_time_window = len(scaled_test)
    current_batch = forecasting_model.get_batch(train_generator, -1)

    LOG0.event_init(event_name='forecast', text='Forecasting')
    if apriori:
        for i in range(PDATA.forecasting_time_window):
            current_pred = forecasting_model.forecasting(current_batch,
                                                         forecasting_data_length=1,
                                                         verbose=False)
            predictions.append(current_pred[0])
            new_event = scaled_test[i]
            current_batch = np.append(current_batch[:, 1:, :], [[new_event]], axis=1)

            LOG0.event_init(event_name='forecast', text='Forecasting: {0}/{1}'.format(i,
                                                                                      PDATA.forecasting_time_window - 1))
    else:
        LOG0.show_off()
        predictions = forecasting_model.forecasting(current_batch,
                                                    forecasting_data_length=PDATA.forecasting_time_window,
                                                    verbose=True)
        LOG0.show_on()

    predictions = pd.DataFrame(predictions).values
    PDATA.forecasting_results = normalization_model.inverse(predictions)

    LOG0.event_init(event_name='forecast', text='Forecasting complited')
    LOG0.event_init(event_name='eval', text='Evaluation')

    estimator = ForecastEstimator()
    PDATA.forecasting_quality = estimator.estimate(true=scaled_test,
                                                   pred=predictions,
                                                   feature_names=PDATA.features_names)

    result['mse'] = PDATA.forecasting_quality.loc['ALL_FEATURES', 'MSE']
    result['mae'] = PDATA.forecasting_quality.loc['ALL_FEATURES', 'MAE']
    result['1-mae'] = 1 - PDATA.forecasting_quality.loc['ALL_FEATURES', 'MAE']

    LOG0.event_init(event_name='eval', text='Evaluation done')
    print(PDATA.forecasting_quality)

    LOG0.run = False
    thread1.join()
    LOG0.close()

    result.update(LOG0.get_resources(event_name='forecast'))

    print('Done')
    return result


def example_join_test_smart_crane():
    PDATA.time_window_length = 10

    result = pd.DataFrame(columns=['test', 'data_size', 'mse', 'mae', '1-mae',
                                   'duration_sec', 'cpu%_min', 'cpu%_mean', 'cpu%_max',
                                   'ram_mb_min', 'ram_mb_mean', 'ram_mb_max'
                                   ])

    dataset_name = 'smart_crane'
    for mode in range(1, 9):
        for test_name, use_izdap in {'c' + str(mode) + '_appsop_izdap0': False,
                                     'c' + str(mode) + '_appsop_izdap1': True}.items():

            Loader = DataLoaderAndPreprocessor(test_name=test_name)
            PDATA.features_matrix = Loader(dataset_name, mode=mode, use_izdap=use_izdap)
            PDATA.features_names = Loader.features_names
            PDATA.forecasting_model_path = Loader.forecasting_model_path
            PDATA.normalization_model_path = Loader.normalization_model_path


            for ex in [str(i) + '_' for i in range(3)]:
                logname = ex + dataset_name + '_' + test_name
                if ex == '0_':
                    result = result.append(example_appsop_model_training(logname=logname), ignore_index=True)
                if ex == '1_':
                    result = result.append(example_appsop_forecasting(logname=logname), ignore_index=True)
                if ex == '2_':
                    result = result.append(example_appsop_forecasting(logname=logname, apriori=False), ignore_index=True)
    result = result.sort_values(by='test').reset_index(drop=True)
    result.to_csv('appsop_logs/join_test_smart_crane.csv', index=False, sep=';', decimal=',')


if __name__ == '__main__':
    example_join_test_smart_crane()



