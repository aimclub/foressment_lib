import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from keras.models import load_model, Sequential
from keras.layers import Dense, Reshape, LSTM, Input, Dropout, SimpleRNN, GRU
from keras.callbacks import EarlyStopping
from keras import optimizers

import keras_tuner  # keras-tuner + grpcio (ver. 1.27.2)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import pandas as pd
import numpy as np
import sys
import json
import math
import matplotlib.pyplot as plt
from typing import Type
from tqdm import tqdm

from .checker import ParamChecker

checker = ParamChecker()


class ForecasterParameters:
    """
    Class to initialization parameters.
    """

    def __init__(self, n_features=1, look_back_length=10, horizon=1):
        """
        :param  n_features: Number of features
        :type n_features: int
        :param look_back_length: The width (number of time steps) of the input time windows
        :type look_back_length: int
        :param horizon: Output time window length
        :type horizon: int
        """

        self.param_names = ['n_features', 'look_back_length', 'horizon']

        self.n_features = n_features
        self.look_back_length = look_back_length
        self.horizon = horizon

    def __setattr__(self, name, val):
        if name == 'param_names':
            super().__setattr__(name, val)
        elif name in self.param_names:
            super().__setattr__(name, checker.check_param(val, name))
        else:
            raise AttributeError(name)

    def read_json(self, filename):
        with open(filename) as f:
            params = json.load(f)

        for k, v in params.items():
            self.__setattr__(k, v)

    def save_json(self, filename):
        """
        Save model parameters to file.
        :param filename: Name of file with parameters and their values
        :type filename: string

        """
        with open(filename, 'w') as outfile:
            class_dict = self.__dict__.copy()
            del class_dict['param_names']
            json_string = json.dumps(class_dict)
            outfile.write(json_string)

    def __str__(self):
        class_dict = self.__dict__.copy()
        del class_dict['param_names']
        return '\n'.join(['{0} = {1}'.format(k, v) for k, v in class_dict.items()])


class AIForecasterParameters(ForecasterParameters):
    def __init__(self, n_features=1, look_back_length=10, horizon=1,
                 units=None, block_type='LSTM', dropout=0,
                 hidden_activation='tanh', output_activation='linear',
                 loss='mse', optimizer_clipvalue=0.5,
                 params_from_file=''):
        """
        :param  n_features: Number of features
        :type n_features: int
        :param look_back_length: The width (number of time steps) of the input time windows
        :type look_back_length: int
        :param horizon: Output time window length
        :type horizon: int

        :param model_hps: Model hyperparameters
        :type model_params: dict

        :param model_hps['n_rec_layers']: Numbers of recurrent neural networl layers
        :type model_hps['n_rec_layers']: int

        :param model_hps['units']: List of number of units on each recurrent layer
        :type model_hps['units']: list

        :param model_hps['block_type']: Recurrent block type
        :type model_hps['block_type']: str

        :param model_hps['dropout']: Частота отсева слоев
        :type model_hps['dropout']: float

        :param model_hps['hidden_activation']: Activation function on hidden layers
        :type model_hps['hidden_activation']: str

        :param model_hps['output_activation']: Activation function on output layer
        :type model_params['output_activation']: str

        :param model_hps['optimizer']: Optimization function
        :type model_hps['optimizer']: str

        :param model_hps['loss']: Loss function
        :type model_hps['loss']: str
        """

        super().__init__(n_features, look_back_length, horizon)

        self.param_names = self.param_names + ['units', 'block_type', 'dropout',
                                               'hidden_activation', 'output_activation', 'optimizer', 'loss']
        if units is None:
            units = [512]

        self.units = units
        # self.n_rec_layers = len(units)
        self.block_type = block_type
        self.dropout = dropout
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.optimizer = optimizers.legacy.Adam(clipvalue=optimizer_clipvalue)
        self.loss = loss

    def n_rec_layers(self):
        return len(self.units.keys())

    def __setattr__(self, name, val):
        if name == 'param_names':
            super().__setattr__(name, val)
        elif name == 'units':
            assert ((type(val) == list) or (type(val) == dict)), "Type if units must be list or dict"
            if type(val) == list:
                super().__setattr__(name, self.units_to_dict(val))
            else:
                super().__setattr__(name, val)
        elif name in self.param_names:
            super().__setattr__(name, checker.check_param(val, name))
        else:
            raise AttributeError(name)

    def save_json(self, filename):
        """
        Save model parameters to file.
        :param filename: Name of file with parameters and their values
        :type filename: string

        """
        with open(filename, 'w') as outfile:
            class_dict = self.__dict__.copy()
            del class_dict['param_names']
            units = list(class_dict['units'].values())
            class_dict['units'] = units
            json_string = json.dumps(class_dict)
            outfile.write(json_string)

    @staticmethod
    def units_to_dict(units):
        units = [checker.check_param(u, 'units_of_layer') for u in units]
        # self.n_rec_layers = len(units)
        return {'units_{0}'.format(i): u for i, u in enumerate(units)}


class TSGenerator:
    """
    Class for timeseries generator.
    """

    def __init__(self, data, model_params: ForecasterParameters):
        """

        :param model_params:
        :type model_params: ForecasterParameters
        """
        self.model_params = model_params

        self._temporalize(data)

    def change_horizon(self, horizon):
        self.model_params.horizon = checker.check_param(horizon, 'horizon')

        data = self.get_data(flatten=True)
        targets = self.get_targets(window_id=-1, flatten=True)
        X = np.append(data, targets, axis=0)
        self._temporalize(X)

    def _temporalize(self, X):
        """
        Reformat data to time windows.
        :param X: Data
        :type X: np.array
        """
        X = checker.check_is_type_param(X, 'data for TSGenerator', np.ndarray)
        assert len(X.shape) in [1, 2], 'Data for TSGenerator must be 1D or 2D numpy array'
        if len(X.shape) == 1:
            X = np.reshape(X, (X.shape[0], 1))
        self.model_params.look_back_length = checker.check_in_range_param(self.model_params.look_back_length,
                                                                          'look_back_length', (0, X.shape[0]))
        self.model_params.horizon = checker.check_in_range_param(self.model_params.horizon,
                                                                 'horizon',
                                                                 (0, X.shape[0] - self.model_params.look_back_length + 1))

        self.data = np.empty(shape=(0, self.model_params.look_back_length, self.model_params.n_features))
        self.targets = np.empty(shape=(0, self.model_params.horizon, self.model_params.n_features))

        data_length = X.shape[0] - self.model_params.look_back_length - self.model_params.horizon + 1
        p, q = X.shape
        m, n = X.strides
        strided = np.lib.stride_tricks.as_strided

        self.data = strided(X,
                            shape=(data_length, self.model_params.look_back_length, q),
                            strides=(m, m, n))
        self.targets = strided(X[self.model_params.look_back_length:],
                               shape=(data_length, self.model_params.horizon, q),
                               strides=(m, m, n))

    def get_data(self, flatten=False, window_id=None, sample=None):
        return self._get_x(self.data, flatten, window_id, sample)

    def get_targets(self, flatten=False, window_id=None, sample=None):
        return self._get_x(self.targets, flatten, window_id, sample)

    def _get_x(self, x, flatten, window_id, sample):
        """
        Get time window by order id.
        :param x: Data
        :type x: np.ndarray
        :param flatten: Flat data or not
        :type  flatten: boolean
        :param window_id: Id of window
        :type  window_id: int
        :return: time window
        :rtype: np.ndarray
        """
        x_to_get = None
        if window_id is not None:
            window_id = checker.check_is_type_param(window_id, 'window_id', int)
            window_id = checker.check_in_range_param(window_id, 'window_id', (-1*(x.shape[0]-1), x.shape[0]-1))
            x_to_get = x[window_id]
            x_to_get = np.reshape(x_to_get, (1,) + x_to_get.shape)
        elif sample is not None:
            sample = checker.check_is_type_param(sample, 'sample', tuple)
            if sample[0]:
                sample_min = sample[0]
            else:
                sample_min = 0
            if sample[1]:
                sample_max = sample[1]
            else:
                sample_max = x.shape[0]
            x_to_get = x[sample_min:sample_max]
        if flatten:
            if x_to_get is not None:
                x_to_get = self._flatten(x_to_get)
            else:
                x_to_get = self._flatten(x)

        if x_to_get is not None:
            return x_to_get
        else:
            return x

    @staticmethod
    def _flatten(X):
        """
        Make data flat (3d array to 2d).
        :param X: Data
        :type X: np.ndarray
        :return: Flatten data
        :rtype: np.ndarray
        """
        flattened_X = np.empty((X.shape[0] + X.shape[1] - 1, X.shape[2]))
        last_id = X.shape[0] - 1
        for i in range(last_id):
            flattened_X[i] = X[i, 0, :]

        for i in range(last_id, X[last_id].shape[0]):
            flattened_X[i] = X[last_id, i, :]
        return flattened_X


class NaiveForecaster:
    def __init__(self, model_params: ForecasterParameters):
        """

        :param model_params:
        :type model_params: ForecasterParameters
        """
        self.model_params = model_params

    def _predict(self, data, verbose=0):
        predictions = np.empty(shape=(0, self.model_params.horizon, self.model_params.n_features))
        if verbose == 1:
            pbar = tqdm(desc='Forecasting', total=self.model_params.horizon * data.shape[0], file=sys.stdout)
        else:
            pbar = None
        for batch in data:
            current_pred = np.array([batch[-1] for i in range(self.model_params.horizon)])
            current_pred = np.reshape(current_pred, (1,) + current_pred.shape)
            predictions = np.concatenate((predictions, current_pred))
            if verbose == 1:
                pbar.update(1)
        if verbose == 1:
            pbar.close()
        return predictions

    def forecasting(self, data, forecasting_data_length=None, verbose=1):
        """
        Forecasting values within a given time window.
        :param data: Data for forecasting
        :type data: np.ndarray
        :param forecasting_data_length: Time window size for forecasting
        :type forecasting_data_length: int
        :param verbose: Show forecasting parameter
        :type verbose: int
        :return: Array of forecast data
        :rtype: np.ndarray
        """
        if not forecasting_data_length:
            forecasting_data_length = self.model_params.horizon

        forecasting_data_length = checker.check_is_type_param(forecasting_data_length, 'forecasting_data_length', int)
        forecasting_data_length = checker.check_in_range_param(forecasting_data_length, 'forecasting_data_length',
                                                               (0, None))

        assert len(data.shape) in [1, 2, 3], 'Data must be 1D, 2D or 3d numpy array'
        if len(data.shape) == 1:
            data = np.reshape(data, (1, data.shape[0], 1))
        if len(data.shape) == 2:
            data = np.reshape(data, (1, data.shape[0], data.shape[1]))
        assert data.shape[1] == self.model_params.look_back_length, ('Data length (or data.shape[1]) '
                                                                     'must be equal to look_back_length')

        if forecasting_data_length <= self.model_params.horizon:
            predictions = self._predict(data, verbose=1)
            for i in range(predictions.shape[0]):
                predictions[i] = predictions[i][:forecasting_data_length]
            return predictions
        else:
            predictions = np.empty(shape=(data.shape[0], forecasting_data_length, data.shape[2]))

            if verbose == 1:
                pbar = tqdm(desc='Forecasting', total=forecasting_data_length * data.shape[0], file=sys.stdout)
            else:
                pbar = None

            for i in range(data.shape[0]):
                batch = np.reshape(data[i], (1,) + data[i].shape)

                pred_to_batch = np.empty(shape=(0, data.shape[2]))
                while len(pred_to_batch) < forecasting_data_length:
                    current_pred = self._predict(batch)
                    current_pred = current_pred[0]
                    pred_to_batch = np.append(pred_to_batch, current_pred, axis=0)
                    batch = np.append(batch[:, self.model_params.horizon:, :], [current_pred], axis=1)
                    if verbose == 1:
                        pbar.update(self.model_params.horizon)
                pred_to_batch = pred_to_batch[:forecasting_data_length]
                pred_to_batch = np.reshape(pred_to_batch, (1,) + pred_to_batch.shape)
                predictions[i] = pred_to_batch
                # predictions = np.reshape(predictions, predictions.shape + (1,))
            if verbose == 1:
                pbar.close()
            return predictions


class AIForecaster(NaiveForecaster):
    """
    Class for forecasting the states of complex objects and processes
    """

    def __init__(self, model_params: AIForecasterParameters = None, model: Sequential = None,
                 from_file='', from_file_config='', from_config=None):
        # """
        # Model initialization
        #
        # :param _max_num_units: Maximum of layer units
        # :type _max_num_units: int
        # :param _default_units_step: Step between units of layers
        # :type _default_units_step: int
        #
        # """
        super().__init__(model_params)
        self.model_config = None
        self.model = model
        if model is not None:
            self._init_model()
        if from_file:
            self.load_from_file(from_file)
        if from_file_config:
            self.load_from_model_config(filename=from_file_config)
        if from_config:
            self.load_from_model_config(config=from_config)
        self.history = None

    def _init_model(self):
        self.model_config = self.model.get_config()
        self._set_model_params_from_config()
        self.default_filename = self.model_params.block_type.lower() + '_' + \
                                '_'.join(str(u) for u in self.model_params.units.values()) + \
                                '_d' + str(self.model_params.dropout).replace('.', '')

    def load_from_file(self, filename=''):
        """
        Open the forecasting model from file.
        :param filename: Name of model file
        :type filename: string
        """
        checker.check_file_is_exist(filename)
        self.model = load_model(filename)
        self._init_model()
        # print(self.model.summary())

    def load_from_model_config(self, filename='', config=None):
        """
        Create model by keras configuration.
        :param filename: Name of file with keras configuration.
        :type filename: string
        """
        if filename:
            with open(filename) as f:
                self.model_config = json.load(f)
        if config:
            self.model_config = config

        self.model = Sequential.from_config(self.model_config)
        self._init_model()
        # print(self.model.summary())
        self.model.compile(optimizer=self.model_params.optimizer, loss=self.model_params.loss)

    def _set_model_params_from_config(self):
        self.model_params = AIForecasterParameters(
            n_features=self.model_config['layers'][0]['config']['batch_input_shape'][2],
            look_back_length=self.model_config['layers'][0]['config']['batch_input_shape'][1],
            block_type=self.model_config['layers'][1]['class_name'],
            hidden_activation=self.model_config['layers'][1]['config']['activation']
            )

        n = 0
        units = []
        dropout = 0
        for layer in self.model_config['layers'][1:]:
            if layer['class_name'] == self.model_params.block_type:
                n = n + 1
                units.append(layer['config']['units'])
            if layer['class_name'] == 'Dropout':
                dropout = layer['config']['rate']
            if layer['class_name'] == 'Dense':
                self.model_params.output_activation = layer['config']['activation']
            if layer['class_name'] == 'Reshape':
                self.model_params.horizon = layer['config']['target_shape'][0]

        self.model_params.units = units
        self.model_params.dropout = dropout

    def save_model_config(self, filename):
        assert self.model, 'Model does not exist'

        self.model_config = self.model.get_config()

        with open(filename, 'w') as outfile:
            json_string = json.dumps(self.model_config)
            outfile.write(json_string)
            outfile.write('\n')

    def save_model(self, filename):
        """
        Save the forecasting model
        """
        self.model.save(filename)

    def build_model(self):
        """
        Build model by parameters.
        """
        self.model = Sequential()
        self.model.add(Input(shape=(self.model_params.look_back_length, self.model_params.n_features)))

        for n in range(self.model_params.n_rec_layers()):
            units = self.model_params.units['units_' + str(n)]
            activation = self.model_params.hidden_activation
            last_layer = False
            if n == (self.model_params.n_rec_layers() - 1):
                last_layer = True
            self._add_recurrent_layer(units, activation, last_layer)

            if self.model_params.dropout > 0:
                self.model.add(Dropout(self.model_params.dropout))

        output_activation = self.model_params.output_activation
        self.model.add(Dense(self.model_params.horizon * self.model_params.n_features,
                             activation=output_activation))
        # Shape => [batch, out_steps, features].
        self.model.add(Reshape([self.model_params.horizon, self.model_params.n_features]))

        self.model.compile(optimizer=self.model_params.optimizer, loss=self.model_params.loss)
        self.model_config = self.model.get_config()

    def _add_recurrent_layer(self, units, activation, last_layer=False):
        return_sequences = not last_layer
        # if n == (self.model_hps['n_rec_layers']-1):

        if self.model_params.block_type == 'SimpleRNN':
            self.model.add(SimpleRNN(units=units, activation=activation,
                                     return_sequences=return_sequences))
        if self.model_params.block_type == 'LSTM':
            self.model.add(LSTM(units=units, activation=activation,
                                return_sequences=return_sequences))
        if self.model_params.block_type == 'GRU':
            self.model.add(GRU(units=units, activation=activation,
                               return_sequences=return_sequences))

    def train(self, X, y, n_epochs=100, batch_size=128,
              verbose=1,
              validation_split=None):
        """
        Training and validation of a neural network model on data

        :param X:
        :type X: numpy.array

        :param y: targets
        :type X: numpy.ndarray

        :param batch_size: Training batch size
        :type batch_size: int

        :param epochs: Number of training epochs
        :type epochs: int

        :param validation_split:
        :type validation_split: float

        """
        early_stop = EarlyStopping(monitor='loss', patience=1)

        self.history = self.model.fit(X, y, epochs=n_epochs,
                                      callbacks=[early_stop],
                                      batch_size=batch_size,
                                      validation_split=validation_split,
                                      shuffle=False,
                                      verbose=verbose)

    def _predict(self, data, verbose=0):
        return self.model.predict(data, batch_size=128, verbose=verbose)

    def get_loss(self):
        return round(self.history.history['loss'][-1], 4)


class AIForecasterTuner:
    def __init__(self, model_params: AIForecasterParameters):
        self.hp_choices = None
        self.model_params = model_params

    def set_tuned_hps(self, block_type=None, units=None, n_rec_layers=None, dropout=None,
                      hidden_activation=None, output_activation=None):
        """
        Set parameters variables for tuning.
        :param tuned_hps: Parameters variables
        :param tuned_hps: dict
        """
        self.hp_choices = {}

        if block_type:
            assert checker.check_is_type_param(block_type, 'block_type', list)
            self.hp_choices['block_type'] = [checker.check_param(val, param_name='block_type') for val in block_type]

        if units:
            assert checker.check_is_type_param(units, 'units', list)
            if n_rec_layers:
                assert checker.check_is_type_param(n_rec_layers, 'n_rec_layers', list)
                assert (max(n_rec_layers) >= len(units)), \
                    'The number of layers should be no more than {0} (the number of transferred units). ' \
                    'Found {1}.'.format(len(units), max(n_rec_layers))

            self.hp_choices['units'] = {f'units_{i}': [checker.check_param(u, param_name='units_of_layer')
                                                       for u in
                                                       checker.check_is_type_param(units_of_layers, 'units_of_layers',
                                                                                   list)]
                                        for i, units_of_layers in enumerate(units)}

        if dropout:
            assert checker.check_is_type_param(dropout, 'dropout', list)
            self.hp_choices['dropout'] = [checker.check_param(val, param_name='dropout') for val in dropout]

        if n_rec_layers:
            assert checker.check_is_type_param(n_rec_layers, 'n_rec_layers', list)
            self.hp_choices['n_rec_layers'] = [checker.check_param(val, param_name='n_rec_layers') for val in
                                               n_rec_layers]

        if hidden_activation:
            assert checker.check_is_type_param(hidden_activation, 'hidden_activation', list)
            self.hp_choices['hidden_activation'] = [checker.check_param(val, param_name='hidden_activation')
                                                    for val in hidden_activation]

        if output_activation:
            assert checker.check_is_type_param(output_activation, 'output_activation', list)
            self.hp_choices['output_activation'] = [checker.check_param(val, param_name='output_activation')
                                                    for val in output_activation]

    def build_hypermodel(self, hp):
        """
        Build hypermodel takes an argument from which to sample hyperparameters.
        :param hp: Hyperparameter object of Keras Tuner (to define the search space for the hyperparameter values)
        :type hp: keras_tuner.HyperParameters
        """
        model = Sequential()
        model.add(Input(shape=(self.model_params.look_back_length,
                               self.model_params.n_features)))

        if 'n_rec_layers' in self.hp_choices:
            layers_range = range(hp.Choice('n_rec_layers', self.hp_choices['n_rec_layers']))
        else:
            layers_range = range(self.model_params.n_rec_layers())

        if 'block_type' in self.hp_choices:
            block_type = hp.Choice('block_type', self.hp_choices['block_type'])
        else:
            block_type = self.model_params.block_type

        for n in layers_range:
            if 'units' in self.hp_choices:
                units = hp.Choice(f'units_{n}', self.hp_choices['units'][f'units_{n}'])
            else:
                units = self.model_params.units[f'units_{n}']
            if 'hidden_activation' in self.hp_choices:
                activation = hp.Choice('hidden_activation', self.hp_choices['hidden_activation'])
            else:
                activation = self.model_params.hidden_activation
            last_layer = False
            if n == (layers_range.stop - 1):
                last_layer = True
            model = self._add_hidden_layer(model, block_type, units, activation, last_layer)

            if 'dropout' in self.hp_choices:
                dropout = hp.Choice('dropout', self.hp_choices['dropout'])
                model.add(Dropout(dropout))
            else:
                if self.model_params.dropout > 0:
                    model.add(Dropout(self.model_params.dropout))

        if 'output_activation' in self.hp_choices:
            output_activation = hp.Choice('output_activation', self.hp_choices['output_activation'])
        else:
            output_activation = self.model_params.output_activation

        model.add(Dense(self.model_params.horizon * self.model_params.n_features,
                        activation=output_activation))
        # Shape => [batch, out_steps, features].
        model.add(Reshape([self.model_params.horizon, self.model_params.n_features]))

        model.compile(optimizer=self.model_params.optimizer, loss=self.model_params.loss)
        return model

    def _add_hidden_layer(self, model, block_type, units, activation, last_layer=False):
        return_sequences = not last_layer
        # if n == (self.model_hps['n_rec_layers']-1):

        if block_type == 'SimpleRNN':
            model.add(SimpleRNN(units=units, activation=activation,
                                return_sequences=return_sequences))
        if block_type == 'LSTM':
            model.add(LSTM(units=units, activation=activation,
                           return_sequences=return_sequences))
        if block_type == 'GRU':
            model.add(GRU(units=units, activation=activation,
                          return_sequences=return_sequences))
        return model

    def _create_tuner_and_searh(self, x, y, tuner_type='RandomSearch',
                                max_trials=10, batch_size=128, epochs=10):
        """

        :param x:
        :param y:
        :param tuner_type:
        :param n_models:
        :param max_trials:
        :param batch_size:
        :param epochs:
        :return:
        """
        if not self.hp_choices:
            # Default tuned parameters.
            self.set_tuned_hps(
                units=[[int(u) for u in np.arange(checker.max_num_units,
                                                     checker.default_units_step,
                                                     -checker.default_units_step)]
                          for n in range(self.model_params.n_rec_layers())],
                hidden_activation=['tanh', 'relu'],
                output_activation=['linear', 'sigmoid'])

        tuner_type = checker.check_in_list_param(tuner_type, 'tuner_type',
                                                 ['RandomSearch', 'BayesianOptimization', 'Hyperband'])
        # Initialize tuner to run the model.
        tuner = None
        if tuner_type == 'RandomSearch':
            tuner = keras_tuner.RandomSearch(
                hypermodel=self.build_hypermodel,
                objective='loss',
                max_trials=max_trials,  # the number of different models to try
                project_name='ai_forecaster',
                overwrite=True
            )
        elif tuner_type == 'BayesianOptimization':
            tuner = keras_tuner.BayesianOptimization(
                hypermodel=self.build_hypermodel,
                objective='loss',
                max_trials=max_trials,
                project_name='ai_forecaster',
                overwrite=True
            )
        elif tuner_type == 'Hyperband':
            tuner = keras_tuner.Hyperband(
                hypermodel=self.build_hypermodel,
                objective='loss',
                project_name='ai_forecaster',
                overwrite=True
            )

        print(tuner.search_space_summary())
        # Run the search
        tuner.search(x, y, batch_size=batch_size, epochs=epochs,
                     callbacks=[EarlyStopping('loss', patience=1)])
        return tuner

    def find_best_models(self, x, y, tuner_type='RandomSearch', n_models=1,
                                  max_trials=10, batch_size=128, epochs=10):
        tuner = self._create_tuner_and_searh(x, y, tuner_type,
                                             max_trials=max_trials, batch_size=batch_size, epochs=epochs)

        print("Results summary")
        print("Showing %d best trials" % n_models)

        for trial in tuner.oracle.get_best_trials(n_models):
            print()
            print(f"Trial {trial.trial_id} summary")
            print("Hyperparameters:")
            hyperparameters = trial.hyperparameters.values

            if 'n_rec_layers' in hyperparameters:
                n_rec_layers = hyperparameters['n_rec_layers']
                units = [hp for hp in hyperparameters.keys() if 'units' in hp]
                if len(units) > n_rec_layers:
                    for i in range(n_rec_layers, len(units)):
                        del hyperparameters[f'units_{i}']

            for hp, value in trial.hyperparameters.values.items():
                print(f"{hp}:", value)
            if trial.score is not None:
                print(f"Score: {trial.score}")


        # best_hps = tuner.get_best_hyperparameters(n_models)
        best_tuner_models = tuner.get_best_models(n_models)
        best_models = [AIForecaster(from_config=tuner_model.get_config()) for tuner_model in best_tuner_models]
        return best_models


class ForecastEstimator:
    """
    Class for evaluating the quality of the forecasting model

    :param quality: Matrix of forecasting quality metrics
    :type quality: pandas.DataFrame
    """

    def __init__(self, feature_names=None):
        self.first_batch = None
        self.true = None
        self.pred = {}
        self.feature_names = feature_names
        self.quality = pd.DataFrame()

    def set_true_values(self, true):
        assert len(true.shape) in [1, 2, 3], 'True data must be 1D, 2D or 3d numpy array'
        if len(true.shape) == 1:
            true = np.reshape(true, (1, true.shape[0], 1))
        if len(true.shape) == 2:
            true = np.reshape(true, (1, true.shape[0], true.shape[1]))
        self.true = true

    def set_pred_values(self, pred, model_name='naive'):
        model_name = checker.check_is_type_param(model_name, model_name, str)
        assert len(pred.shape) in [1, 2, 3], 'Predicted data must be 1D, 2D or 3d numpy array'
        if len(pred.shape) == 1:
            pred = np.reshape(pred, (1, pred.shape[0], 1))
        if len(pred.shape) == 2:
            pred = np.reshape(pred, (1, pred.shape[0], pred.shape[1]))

        self.pred[model_name] = pred

    def set_first_batch(self, first_batch):
        self.first_batch = first_batch

    def estimate(self):
        """
        Quality evaluation of the forecasting model

        :param data: Real data array
        :type data: np.ndarray

        :param pred: Array of forecasted data
        :type pred: np.ndarray

        :return: Matrix of forecasting quality metrics, MSE - mean squared error, MAE - mean absolute error
        :rtype: pandas.DataFrame
        """
        assert self.true is not None, 'No true values for estimate'
        self.quality = pd.DataFrame()
        true_reshaped = self.true.reshape((self.true.shape[0] * self.true.shape[1], self.true.shape[2]))

        if self.feature_names is None:
            feature_names = [str(i) for i in range(self.true.shape[2])]
        else:
            feature_names = self.feature_names.copy()
        feature_names.append('ALL_FEATURES')

        for model_name, pred_vals in self.pred.items():
            assert (len(self.true) == len(pred_vals)), f'The length of' + model_name +\
                                                       ' result not equal to true values'
            pred_reshaped = pred_vals.reshape((self.true.shape[0] * self.true.shape[1], self.true.shape[2]))

            mse = mean_squared_error(true_reshaped, pred_reshaped, multioutput='raw_values', squared=True)
            mse = np.append(mse, mean_squared_error(true_reshaped, pred_reshaped, squared=True))
            self.quality[model_name + '_MSE'] = mse

            rmse = mean_squared_error(true_reshaped, pred_reshaped, multioutput='raw_values', squared=False)
            rmse = np.append(rmse, mean_squared_error(true_reshaped, pred_reshaped, squared=False))
            self.quality[model_name + '_RMSE'] = rmse

            mae = mean_absolute_error(true_reshaped, pred_reshaped, multioutput='raw_values')
            mae = np.append(mae, mean_absolute_error(true_reshaped, pred_reshaped))
            self.quality[model_name + '_MAE'] = mae

            r2 = r2_score(true_reshaped, pred_reshaped, multioutput='raw_values')
            r2 = np.append(r2, r2_score(true_reshaped, pred_reshaped))
            self.quality[model_name + '_R2'] = r2

        self.quality.index = feature_names

    def save_quality(self, filename):
        """
        Save estimation results to file

        :param filename: name of the file to save
        :type filename: str
        """
        if not os.path.exists('forecaster_results/'):
            os.makedirs('forecaster_results/')

        self.quality.to_csv('forecaster_results/' + filename + '.csv',
                            index_label='feature')

    def save_pred_result(self, dataset_name):
        """
        Save forecaster models results to file.
        :param filename: name of the file to save
        :return: str
        """
        if not os.path.exists('forecaster_results/'):
            os.makedirs('forecaster_results/')

        for model_name, pred_vals in self.pred.items():
            filename = 'forecaster_results/' + dataset_name + '_' + model_name + '.npy'
            np.save(filename, pred_vals)
            print('Save ' + filename)

    def draw_feature(self, i_feature, ax, data_size=1000):

        # plt.figure(figsize=(15, 8))
        def draw_windows(data, start_x=0, color='black', label='Data', alpha=1.0):
            timeline = []
            for p in range(data.shape[0]):
                y = data[p]
                timeline.append(y[0])
                x = range(p + start_x, p + y.shape[0] + start_x)
                if p == (data.shape[0] - 1):
                    ax.plot(x, y, marker='.', color=color, label=label, alpha=alpha)
                else:
                    ax.plot(x, y, marker='.', color=color, alpha=alpha)
            x = range(start_x, len(timeline) + start_x)
            ax.plot(x, timeline, color=color, alpha=alpha)

        target_start_point = 0
        connection_line = {}

        if self.first_batch is not None:
            draw_windows(self.first_batch[:, :, i_feature], color='blue', label='Inputs')
            target_start_point = self.first_batch.shape[1]
            connection_line = {'x': [target_start_point - 1, target_start_point],
                               'y': [self.first_batch[0][-1, i_feature], None]}

        if self.true is not None:
            if connection_line:
                connection_line['y'][1] = self.true[0][0, i_feature]
                plt.plot(connection_line['x'], connection_line['y'], marker='.', color='green')
            draw_windows(self.true[:data_size, :, i_feature], start_x=target_start_point, color='green', label='True')

        if self.pred:
            model_names = list(self.pred.keys())
            color_dicts = {}
            if 'naive' in model_names:
                model_names.remove('naive')
                color_dicts['naive'] = 'grey'

            color_map = plt.cm.get_cmap('plasma', len(model_names))
            for i, model_name in enumerate(model_names):
                color_dicts[model_name] = color_map(i)

            for model_name, pred_vals in self.pred.items():
                if connection_line:
                    connection_line['y'][1] = pred_vals[0][0, i_feature]
                    plt.plot(connection_line['x'], connection_line['y'], marker='.', color=color_dicts[model_name])
                draw_windows(pred_vals[:data_size, :, i_feature], start_x=target_start_point, color=color_dicts[model_name],
                             label=model_name.capitalize())

        ax.legend(fontsize=8)
        # plt.grid('both')
        # plt.show()

    def draw(self, size=1000, feature_names=None):
        if feature_names is None:
            if self.feature_names is None:
                feature_names = [str(i) for i in range(self.true.shape[2])]
            else:
                feature_names = self.feature_names.copy()

        n = len(feature_names)
        nrows = 1
        ncols = 1
        if n <= 3:
            nrows = n
        elif n % 3 == 0:
            ncols = 3
            nrows = int(n / 3)
        elif n % 2 == 0:
            ncols = 2
            nrows = int(n / 2)
        else:
            ncols = 2
            nrows = int(math.floor(n / 2))

        fig, axs = plt.subplots(nrows, ncols, sharex='col', figsize=(ncols*6, nrows*3))
        fig.tight_layout(pad=2.3)
        fig.supxlabel('Time')

        i_feature = 0
        color_map = plt.cm.get_cmap('plasma', n)

        for i in range(nrows):
            if ncols == 1:
                self.draw_feature(i_feature, axs[i], size)
                axs[i].set_ylabel(feature_names[i_feature])
                i_feature = i_feature + 1
            else:
                for j in range(ncols):
                    if i_feature > n:
                        break
                    self.draw_feature(i_feature, axs[i, j], size)
                    axs[i, j].set_ylabel(feature_names[i_feature])
                    i_feature = i_feature + 1
        plt.show()
