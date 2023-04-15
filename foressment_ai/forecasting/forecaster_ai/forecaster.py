import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Input, Dropout
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import TimeseriesGenerator

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle

import pandas as pd
import numpy as np
import sys


class AIForecaster:
    """
    Class for forecasting the states of complex objects and processes

    :param epochs: Number of training epochs
    :type epochs: int

    :param batch_size: Training batch size
    :type batch_size: int

    :param early_stop: Object of the EarlyStopping class responsible for prematurely stopping training
    :type early_stop: keras.callback

    :param time_window_length: Time window size during training
    :type time_window_length: int

    :param n_features: Number of features
    :type n_features: int

    :param model: Neural network model
    :type model: keras.models.Sequential

    :param model_path: Path to the file with the forecasting model
    :type model_path: str
    """
    def __init__(self, time_window_length=0, n_features=0,
                 model_path='',
                 n_epochs=2, open=False):
        """
        Model initialization

        :param open: Parameter for load model
        :type open: bool
        """
        self.model_path = model_path

        if open:
            self.open_model()
        else:
            if (time_window_length != 0) and (n_features != 0):
                if type(time_window_length) == int:
                    if type(n_features) == int:
                        self.time_window_length = time_window_length
                        self.n_features = n_features
                    else:
                        print('Nuber of features length must be integer')
                        exit()
                else:
                    print('Time window length must be integer')
                    exit()

            else:
                print('Uncorrected zero values')
                exit()

            self.model = Sequential([
                Input(shape=(self.time_window_length, self.n_features)),
                # self.input = (time_window_length, n_features)
                LSTM(64, activation='relu', return_sequences=True,
                     kernel_regularizer=regularizers.l2(0.00)),
                Dropout(0.01),
                LSTM(32, return_sequences=True, activation='relu',
                     kernel_regularizer=regularizers.l2(0.00)),
                Dropout(0.01),
                LSTM(16, return_sequences=False, activation='relu',
                     kernel_regularizer=regularizers.l2(0.00)),
                Dropout(0.01),
                Dense(self.n_features, activation='sigmoid')
            ])
            self.model.compile(optimizer='adam', loss='mse')

        self.epochs = n_epochs
        self.batch_size = 128
        self.early_stop = EarlyStopping(monitor='loss', patience=1)

    def train(self, train_generator, validation_generator=None,
              save=True):
        """
        Training and validation of a neural network model on data

        :param train_generator: Temporary training data batch generator
        :type train_generator: keras.preprocessing.sequence.TimeseriesGenerator

        :param validation_generator: Temporary test data batch generator
        :type validation_generator: keras.preprocessing.sequence.TimeseriesGenerator

        :param save: Parameter fo saving model
        :type save: bool
        """
        generator_batch_size = train_generator[0][1].shape[1]
        if generator_batch_size != self.n_features:
            print('Incorrect data for training. Number of features must be = ' + str(self.n_features))
            exit()

        history = self.model.fit(train_generator, epochs=self.epochs,
                       validation_data=validation_generator,
                       callbacks=[self.early_stop],
                       batch_size=self.batch_size)
        if save:
            self.save_model()

        loss = round(history.history['loss'][-1], 4)
        return loss

    def forecasting(self, current_batch, forecasting_data_length, verbose=True):
        """
        Forecasting values within a given time window

        :param current_batch: Data array (batch) in the time window after which the forecasting is made
        :type current_batch: numpy.array

        :return: Array of forecast data
        :rtype: numpy.array
        """
        predictions = []

        for i in range(forecasting_data_length):
            if verbose:
                sys.stdout.write('\r\x1b[K' + 'Forecasting: {0}/{1}'.format(i, forecasting_data_length-1))
                sys.stdout.flush()
            current_pred = self.model.predict(current_batch,
                                              batch_size=self.batch_size)[0]
            predictions.append(current_pred)
            current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
        predictions = pd.DataFrame(predictions).values
        return predictions

    def save_model(self):
        """
        Save the forecasting model
        """
        self.model.save(self.model_path, save_format='h5')

    def open_model(self):
        """
        Open the forecasting model
        """
        if os.path.isfile(self.model_path):
            self.model = load_model(self.model_path)
            self.time_window_length = self.model.input.shape[1]
            self.n_features = self.model.input.shape[2]
            print(self.model.summary())

        else:
            print('File with foreasting model does not exist')
            self.model = None
            exit()

    def data_to_generator(self, data):
        """
        Convert the data to a temporary data batch generator

        :param data: Array of data to convert
        :type data: numpy.array

        :return: Temporary data batch generator
        :rtype: keras.preprocessing.sequence.TimeseriesGenerator
        """
        try:
            generator = TimeseriesGenerator(data, data,
                                            length=self.time_window_length,
                                            batch_size=1)
            return generator
        except:
            return None

    def get_batch(self, generator, current_batch_id):
        """
        Upload the last batch of temporary data

        :param generator: Temporary data batch generator
        :type generator: keras.preprocessing.sequence.TimeseriesGenerator

        :return: Last batch of temporary data
        :rtype: numpy.array
        """
        if current_batch_id == -1:
            config = generator.get_config()
            current_batch_id = config['end_index'] - self.time_window_length
        try:
            batch = generator[current_batch_id]
            batch = np.append(batch[0][:, 1:, :], [batch[1]], axis=1)
            return batch
        except:
            print('Wrong batch number')
            exit()


class DataScaler:
    """
    Class for data normalization (scaling)

    :param scaler: Data normalization (scaling) model
    :type scaler: sklearn.preprocessing

    :param scaler_path: Path to the normalization model file
    :type scaler_path: string
    """
    def __init__(self, scaler_path,
                 open=False):
        """
        Model initialization.

        :param open: Parameter for load model
        :type open: bool
        """
        self.scaler_path = scaler_path

        if open:
            self.open()
        else:
            self.scaler = MinMaxScaler()

    def fit(self, data, save=True):
        """
        Training the normalization model

        :param data: Training data array
        :type data: numpy.array

        :param open: Parameter for saving model
        :type open: bool
        """
        self.scaler.fit(data)
        if save:
            self.save()

    def save(self):
        """
        Save the normalization model
        """
        with open(self.scaler_path, 'wb') as file:
            pickle.dump(self.scaler, file)
            file.close()

    def open(self):
        """
        Open the normalization model
        """
        if os.path.isfile(self.scaler_path):
            with open(self.scaler_path, 'rb') as file:
                self.scaler = pickle.load(file)
                file.close()
        else:
            print('File with normalization model does not exist')
            self.scaler = None

    def transform(self, data):
        """
        Data normalization

        :param data: Data array
        :type data: numpy.array

        :return: Array of normalized data
        :rtype: numpy.array
        """
        return self.scaler.transform(data)

    def inverse(self, data):
        """
        Inverse data transformation

        :param data: Array of normalized data
        :type data: numpy.array

        :return: Array of inverted data
        :rtype: numpy.array
        """
        return self.scaler.inverse_transform(data)


class ForecastEstimator:
    """
    Class for evaluating the quality of the forecasting model

    :param quality: Matrix of forecasting quality metrics
    :type quality: pandas.DataFrame
    """
    def __init__(self):
        self.quality = pd.DataFrame()

    def estimate(self, true, pred, feature_names=[]):
        """
        Quality evaluation of the forecasting model

        :param data: Real data array
        :type data: numpy.array

        :param pred: Array of forecasted data
        :type pred: numpy.array

        :return: Matrix of forecasting quality metrics, MSE - mean squared error, MAE - mean absolute error
        :rtype: pandas.DataFrame
        """
        if len(true) != len(pred):
            print('The length of the samples is not equal')

        else:
            self.quality['MSE'] = mean_squared_error(true, pred, multioutput='raw_values')
            self.quality['MAE'] = mean_absolute_error(true, pred, multioutput='raw_values')

            if len(feature_names) == self.quality.shape[0]:
                self.quality.index = feature_names

            self.quality.loc['ALL_FEATURES', 'MSE'] = mean_squared_error(true, pred)
            self.quality.loc['ALL_FEATURES', 'MAE'] = mean_absolute_error(true, pred)

        return self.quality

    def save(self, file_name):
        """
        Save results to file

        :param file_name: name of the file to save
        :type file_name: str
        """
        if not os.path.exists('forecaster_results/'):
            os.makedirs('forecaster_results/')

        self.quality.to_csv('forecaster_results/' + file_name + '.csv',
                            index_label='feature')

