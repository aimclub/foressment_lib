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
from tqdm import tqdm
from typing import Union

from foressment_ai.forecasting.forecaster_ai.checker import ParamChecker

checker = ParamChecker()


class ForecasterParameters:
    """
    Main parameters for forecasting models.

    Args:
        n_features: Integer, number of features. Defaults to 1.
        look_back_length: Integer, width (number of time steps) of the input time windows. Defaults to 10.
        horizon: Integer, output time window length. Defaults to 1.

    Examples:
        model_params = ForecasterParameters()
        model_params.look_back_length = 100
        ```
        model_params = ForecasterParameters(4, 100, 10)
        ```
        model_params = ForecasterParameters()
        model_params.read_json('params.json')
    """
    __slots__ = ['n_features', 'look_back_length', 'horizon']

    def __init__(self,
                 n_features: int = 1,
                 look_back_length: int = 10,
                 horizon: int = 1):
        """
        Initialization.

        Args:
            n_features: Integer, number of features. Defaults to 1.
            look_back_length: Integer, width (number of time steps) of the input time windows. Defaults to 10.
            horizon: Integer, output time window length. Defaults to 1.
        """
        self.n_features = n_features
        self.look_back_length = look_back_length
        self.horizon = horizon

    def __setattr__(self, name, val):
        super().__setattr__(name, checker.check_param(val, name))

    def read_json(self, filename: str):
        with open(filename) as f:
            params = json.load(f)

        for k, v in params.items():
            self.__setattr__(k, v)

    def save_json(self, filename: str):
        """
        Save model parameters to file.
        Params:
            filename: String, name of file with parameters and their values
        """
        with open(filename, 'w') as outfile:
            class_dict = self.__dict__.copy()
            del class_dict['param_names']
            json_string = json.dumps(class_dict)
            outfile.write(json_string)


class DeepForecasterParameters(ForecasterParameters):
    """
    Configuration parameters and hyperparameters for deep forecasting models.

    Args:
        n_features: Integer, number of features. Defaults to 1.
        look_back_length: Integer, width (number of time steps) of the input time windows. Defaults to 10.
        horizon: Integer, output time window length. Defaults to 1.
        units: Dict, number of units on each recurrent layer. Defaults to {'units_0': 512}.
        block_type: String, recurrent block type. Defaults to 'LSTM'.
        dropout: Float, dropout rate. Defaults to 0.
        hidden_activation: String, activation function on hidden layers. Defaults to 'tanh'.
        output_activation: String, activation function on output layer. Defaults to 'linear'.
        loss: String, loss function. Defaults to 'mse'.
        optimizer: keras.optimizers.Adam, optimizer that implements the Adam algorithm.

    Note:
        Unit values are initialized by passing a list of values for each layer in order.

    Examples:
        model_params = DeepForecasterParameters()
        model_params.block_type = 'GRU'
        model_params = [512, 128]
        ```
        model_params = DeepForecasterParameters(4, 100, 10, block_type = 'GRU', units=[512, 128])
    """
    __slots__ = ['n_features', 'look_back_length', 'horizon', 'units', 'block_type', 'dropout',
                 'hidden_activation', 'output_activation', 'loss', 'optimizer']

    def __init__(self,
                 n_features: int = 1,
                 look_back_length: int = 10,
                 horizon: int = 1,
                 units: Union[list, dict] = None,
                 block_type: str = 'LSTM',
                 dropout: float = 0,
                 hidden_activation: str = 'tanh',
                 output_activation: str = 'linear',
                 loss: str = 'mse'):
        """
        Initialization.

        Args:
            n_features: Integer, number of features. Defaults to 1.
            look_back_length: Integer, width (number of time steps) of the input time windows. Defaults to 10.
            horizon: Integer, output time window length. Defaults to 1.
            units: Number of units on each recurrent layer. Defaults to {'units_0': 512}.
                There is teo type to set values:
                - as list of ordered integers as [512, 128],
                - as dict of layers as {'units_0': 512, 'units_1': 128}.
            block_type: String, recurrent block type. Defaults to 'LSTM'.
            dropout: Float, dropout rate. Defaults to 0.
            hidden_activation: String, activation function on hidden layers. Defaults to 'tanh'.
            output_activation: String, activation function on output layer. Defaults to 'linear'.
            loss: String, loss function. Defaults to 'mse'.
        """

        super().__init__(n_features, look_back_length, horizon)

        if units is None:
            self.units = [512]
        else:
            self.units = units
        self.block_type = block_type
        self.dropout = dropout
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.optimizer = optimizers.Adam(clipvalue=0.5)
        self.loss = loss

    def n_rec_layers(self) -> int:
        """
        Return number of hidden recurrent layers.
        """
        return len(self.units.keys())

    def __setattr__(self, name, val):
        if name != 'units':
            super().__setattr__(name, checker.check_param(val, name))
        else:
            assert ((type(val) == list) or (type(val) == dict)), "Type if units must be list or dict"
            if type(val) == list:
                super().__setattr__(name, self._units_to_dict(val))
            else:
                super().__setattr__(name, val)

    def save_json(self, filename: str):
        """
        Save model parameters to file.
        Args:
            filename: String, name of file with parameters and their values
        """
        with open(filename, 'w') as outfile:
            class_dict = self.__dict__.copy()
            del class_dict['param_names']
            units = list(class_dict['units'].values())
            class_dict['units'] = units
            json_string = json.dumps(class_dict)
            outfile.write(json_string)

    @staticmethod
    def _units_to_dict(units: list):
        """
        Reformat units to dictionary.

        Args:
            units: List of integers, number of units on each recurrent layer.

        Returns:
            Dict, number of units on each recurrent layer.
        """
        units = [checker.check_param(u, 'units_of_layer') for u in units]
        return {'units_{0}'.format(i): u for i, u in enumerate(units)}


class TSGenerator:
    """
    Time series generator.

    Args:
       model_params: ForecasterParameters or DeepForecasterParameters, parameters of forecasting models.
       data: Numpy array, input time windows for forecasting.
       targets: Numpy array, output time windows as target results of forecasting.

    Examples:
        x = np.random.rand(1000, 2)
        model_params = ForecasterParameters(2, 100, 1)
        ts = TSGenerator(x, model_params)
    """
    __slots__ = ['model_params', 'data', 'targets']

    def __init__(self, x: np.ndarray, model_params: Union[ForecasterParameters, DeepForecasterParameters]):
        """
        Create time series generator.

        Args:
            x: Numpy array, input data for generator.
            model_params: ForecasterParameters or DeepForecasterParameters, parameters of forecasting models.
        """
        self.model_params = model_params
        self.data = None
        self.targets = None
        self._temporalize(x)

    def change_horizon(self, horizon: int):
        """
        Change horizon of forecasting.

        Args:
            horizon: Integer, new forecasting horizon (time steps)
        """
        self.model_params.horizon = checker.check_param(horizon, 'horizon')

        data = self.get_data(flatten=True)
        targets = self.get_targets(window_id=-1, flatten=True)
        x = np.append(data, targets, axis=0)
        self._temporalize(x)

    def _temporalize(self, x: np.ndarray):
        """
        Reformat data to time windows.

        Args:
            x: Numpy array, input data for generator
        """
        x = checker.check_is_type_param(x, 'data for TSGenerator', np.ndarray)
        assert len(x.shape) in [1, 2], 'Data for TSGenerator must be 1D or 2D numpy array'
        if len(x.shape) == 1:
            x = np.reshape(x, (x.shape[0], 1))
        self.model_params.look_back_length = checker.check_in_range_param(self.model_params.look_back_length,
                                                                          'look_back_length',
                                                                          (0, x.shape[0]))
        self.model_params.horizon = checker.check_in_range_param(self.model_params.horizon,
                                                                 'horizon',
                                                                 (0,
                                                                  x.shape[0] - self.model_params.look_back_length + 1))

        self.data = np.empty(shape=(0, self.model_params.look_back_length, self.model_params.n_features))
        self.targets = np.empty(shape=(0, self.model_params.horizon, self.model_params.n_features))

        data_length = x.shape[0] - self.model_params.look_back_length - self.model_params.horizon + 1
        p, q = x.shape
        m, n = x.strides
        strided = np.lib.stride_tricks.as_strided

        self.data = strided(x,
                            shape=(data_length, self.model_params.look_back_length, q),
                            strides=(m, m, n))
        self.targets = strided(x[self.model_params.look_back_length:],
                               shape=(data_length, self.model_params.horizon, q),
                               strides=(m, m, n))

    def get_data(self,
                 flatten: bool = False,
                 window_id: int = None,
                 sample: int = None) -> np.ndarray:
        """
        Return input time windows data.

        Args:
            flatten: Boolean, convert sliding windows back to sequence or not. Defaults to `False`.
            window_id: Integer or `None`, number of returned single time window. Defaults to `None`.
            sample: Tuple ('start_id', 'end_id') or `None`, segment of returned time windows. Defaults to `None`.

        Returns:
            Numpy array, input time windows for forecasting.
=        """
        return self._get_x(self.data, flatten, window_id, sample)

    def get_targets(self,
                    flatten: bool = False,
                    window_id: int = None,
                    sample: int = None) -> np.ndarray:
        """
        Return output time windows data.

        Args:
            flatten: Boolean, convert sliding windows back to sequence or not. Defaults to `False`.
            window_id: Integer or `None`, number of returned single time window. Defaults to `None`.
            sample: Tuple ('start_id', 'end_id') or `None`, segment of returned time windows. Defaults to `None`.

        Returns:
            Numpy array, output time windows.
=        """
        return self._get_x(self.targets, flatten, window_id, sample)

    def _get_x(self, x, flatten: bool, window_id: int, sample: int) -> np.ndarray:
        """
        Return time windows data.

        Args:
            x: Numpy array, data or targets.
            flatten: Boolean, convert sliding windows back to sequence or not.
            window_id: Integer, number of returned single time window.
            sample: Tuple ('start_id', 'end_id'), segment of returned time windows.

        Returns:
            Numpy array, time windows.
=        """
        x_to_get = None
        if window_id is not None:
            window_id = checker.check_is_type_param(window_id, 'window_id', int)
            window_id = checker.check_in_range_param(window_id, 'window_id', (-1 * (x.shape[0] - 1), x.shape[0] - 1))
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
    def _flatten(x: np.ndarray):
        """
        Make returned data flat (3d array to 2d).

        Args:
            x: Numpy array, time series data.

        Returns:
            Numpy array, flattened time series data.
        """
        flattened_X = np.empty((x.shape[0] + x.shape[1] - 1, x.shape[2]))
        last_id = x.shape[0] - 1
        for i in range(last_id):
            flattened_X[i] = x[i, 0, :]

        for i in range(last_id, x[last_id].shape[0]):
            flattened_X[i] = x[last_id, i, :]
        return flattened_X


class NaiveForecaster:
    """
    Baseline forecasting model.

    Args:
        model_params: ForecasterParameters, parameters of forecasting model

    Examples:
        model_params = ForecasterParameters(4, 100, 10)
        model = NaiveForecaster(model_params)
    """
    __slots__ = ['model_params']

    def __init__(self, model_params: ForecasterParameters):
        self.model_params = model_params

    def _predict(self,
                 data: np.ndarray,
                 verbose: int = 0) -> np.ndarray:
        """
        Base forecasting function for model.

        Args:
            data: Numpy array, time series data for forecasting.
            verbose: 0 or 1, verbosity mode. 0 = silent, 1 = progress bar. Defaults to 0.

        Returns:
            Numpy array, forecasting result.
        """

        predictions = np.empty(shape=(0, self.model_params.horizon, self.model_params.n_features))
        if verbose == 1:
            pbar = tqdm(desc='Forecasting', total=data.shape[0], file=sys.stdout)
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

    def forecasting(self,
                    data: np.ndarray,
                    forecasting_data_length: int = None,
                    verbose: int = 1) -> np.ndarray:
        """
        Main forecasting functions.
        Args:
            data: Numpy array, data for forecasting
            forecasting_data_length: Integer or `None`, time window size for forecasting. Defaults to horizon value.
            verbose: 0 or 1, verbosity mode. 0 = silent, 1 = progress bar. Defaults to 1.

        Returns:
            Numpy array, forecasting result
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

        # Single-step prediction.
        if forecasting_data_length <= self.model_params.horizon:
            predictions = self._predict(data, verbose=1)
            if forecasting_data_length < self.model_params.horizon:
                for i in range(predictions.shape[0]):
                    predictions[i] = predictions[i][:forecasting_data_length]
            return predictions

        # Multi-step prediction.
        else:
            predictions = np.empty(shape=(data.shape[0], forecasting_data_length, data.shape[2]))

            if verbose == 1:
                pbar = tqdm(desc='Forecasting', total=data.shape[0], file=sys.stdout)
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
                    pbar.update(1)
                pred_to_batch = pred_to_batch[:forecasting_data_length]
                pred_to_batch = np.reshape(pred_to_batch, (1,) + pred_to_batch.shape)
                predictions[i] = pred_to_batch
            if verbose == 1:
                pbar.close()
            return predictions


class DeepForecaster(NaiveForecaster):
    """
    Deep forecasting models based on RNNs.

    Args:
        model_params: DeepForecasterParameters, parameters of forecasting model. Defaults to `None`.
        model: keras Sequential, groups of layers. Defaults to `None`.
        model_config: Dict, model configuration. Defaults to `None`.
        history: keras.callbacks.History, history of model training. Defaults to `None`.
        default_name: A string, model name as <`block_type`_`units`_`dropout`>. Defaults to ``.

    Examples:
        model_params = DeepForecasterParameters(4, 100, 10, block_type = 'GRU', units=[512, 128])
        model = DeepForecaster(model_params)
    """

    def __init__(self,
                 model_params: DeepForecasterParameters = None,
                 model: Sequential = None,
                 from_file: str = '',
                 from_file_config: str = '',
                 from_config: dict = None):
        """
        Initialization.

        Args:
            model_params: DeepForecasterParameters  or `None`, parameters of forecasting model. Defaults to `None`.
            model: keras Sequential  or `None`, groups of layers. Defaults to `None`.
            from_file: A string, name of keras model file. Defaults to ``.
            from_file_config: String, name of keras model configuration JSON file. Defaults to ``.
            from_config: Dict or `None`, model configuration in keras format. Defaults to ``.
        """
        super().__init__(model_params)
        self.model_config = None
        self.model = model
        self.default_name = ''
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
        self.default_name = self.model_params.block_type.lower() + '_' + \
                            '_'.join(str(u) for u in self.model_params.units.values()) + \
                            '_d' + str(self.model_params.dropout).replace('.', '')

    def load_from_file(self, filename: str = ''):
        """
        Open the forecasting model from file.
        Args:
            filename: A string, name of keras model file
        """
        checker.check_file_is_exist(filename)
        self.model = load_model(filename)
        self._init_model()

    def load_from_model_config(self, filename: str = '', config: dict = None):
        """
        Create model by keras configuration.

        Args:
            filename: A string, name of keras model configuration JSON file.
            config: Dict or `None`, model configuration in keras format.

        Returns:
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
        """
        Set model_params arguments from model_config.
        """
        self.model_params = DeepForecasterParameters(
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

    def save_model_config(self, filename: str):
        """
        Save model configuration in the file.

        Args:
            filename: String, name of keras model configuration JSON file.
        """
        assert self.model, 'Model does not exist'
        self.model_config = self.model.get_config()

        with open(filename, 'w') as outfile:
            json_string = json.dumps(self.model_config)
            outfile.write(json_string)
            outfile.write('\n')

    def save_model(self, filename: str):
        """
        Save model in keras format.

        Args:
            filename: String, name of keras model file.
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
            self._add_hidden_layer(units, activation, last_layer)

            if self.model_params.dropout > 0:
                self.model.add(Dropout(self.model_params.dropout))

        output_activation = self.model_params.output_activation
        self.model.add(Dense(self.model_params.horizon * self.model_params.n_features,
                             activation=output_activation))
        # Shape => [batch, out_steps, features].
        self.model.add(Reshape([self.model_params.horizon, self.model_params.n_features]))

        self.model.compile(optimizer=self.model_params.optimizer, loss=self.model_params.loss)
        self.model_config = self.model.get_config()

    def _add_hidden_layer(self, units: int, activation: str, last_layer: bool = False):
        """
        Add new hidden recurrent layer into the model.

        Args:
            units: Integer, number of units of layer.
            activation: String, hidden layer activation function.
            last_layer: Boolean, it's last layer or not. Defaults to `False`.
        """
        return_sequences = not last_layer

        if self.model_params.block_type == 'SimpleRNN':
            self.model.add(SimpleRNN(units=units, activation=activation,
                                     return_sequences=return_sequences))
        if self.model_params.block_type == 'LSTM':
            self.model.add(LSTM(units=units, activation=activation,
                                return_sequences=return_sequences))
        if self.model_params.block_type == 'GRU':
            self.model.add(GRU(units=units, activation=activation,
                               return_sequences=return_sequences))

    def train(self,
              x: np.ndarray,
              y: np.ndarray,
              n_epochs: int = 100,
              batch_size: int = 128,
              verbose: int = 1,
              validation_data: tuple = None,
              validation_split: float = None,
              early_stop_patience: int = 1):
        """
        Training and validation of a forecasting model on data.

        Args:
            x: Numpy array, training input data.
            y: Numpy array, target data.
            n_epochs: Integer, number of epochs to train the model. Defaults to 100.
            batch_size: Integer, number of samples per gradient update. Defaults to 128.
            verbose: 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. Defaults to 1.
            validation_data:  Tuple `(x_val, y_val)` of Numpy arrays  or `None`.
                Data on which to evaluate the loss at the end of each epoch. Defaults to `None`.
            validation_split:  Float between 0 and 1 or `None`, fraction of the training data
                to be used as validation data. Defaults to `None`.
            early_stop_patience: Integer, number of epochs with no improvement
                after which training will be stopped. Defaults to 1.
        """
        if (validation_data is not None) or (validation_split is not None):
            monitor = 'val_loss'
        else:
            monitor = 'loss'

        early_stop = EarlyStopping(monitor=monitor, patience=early_stop_patience)

        self.history = self.model.fit(x, y, epochs=n_epochs,
                                      callbacks=[early_stop],
                                      batch_size=batch_size,
                                      validation_data=validation_data,
                                      validation_split=validation_split,
                                      shuffle=False,
                                      verbose=verbose)

    def _predict(self, data: np.ndarray, verbose: int = 0) -> np.ndarray:
        """
        Base forecasting function for deep learning model.

        Args:
            data: Numpy array, time series data for forecasting.
            verbose: 0 or 1, verbosity mode. 0 = silent, 1 = progress bar. Defaults to 0.

        Returns:
            Numpy array, forecasting result
        """
        return self.model.predict(data, batch_size=10, verbose=verbose)


class DeepForecasterTuner:
    """
    Tuner for optimizing forecasting model hyperparameters.

    Args:
        hp_choices: Dictionary, predefined sets of possible hyperparameters values.
        model_params: DeepForecasterParameters, parameters of forecasting model.
        tuner: Tuner keras class or 'None', model for hyperparameters optimization.

    Examples:
        model_params = DeepForecasterParameters(4, 100, 10, block_type = 'GRU')
        tuner = DeepForecasterTuner()
    """

    def __init__(self, model_params: DeepForecasterParameters):
        self.hp_choices = {}
        self.model_params = model_params
        self.tuner = None

    def set_tuned_hps(self,
                      block_type: list = None,
                      units: list = None,
                      n_rec_layers: list = None,
                      dropout: list = None,
                      hidden_activation: list = None,
                      output_activation: list = None):
        """
        Set parameters variables for tuning.

        Args:
            block_type: List of strings or `None`, possible recurrent block types.
            units: List of a list of integers or `None`, possible numbers of units on each recurrent layer.
            n_rec_layers: List of integers or `None`, possible numbers of hidden layers.
            dropout: List of floats or `None`, possible dropout rate values.
            hidden_activation: List of strings or `None`, possible activation functions on hidden layers.
            output_activation: List of strings or `None`,possible activation functions on output layer.

        Examples:
            model_params = DeepForecasterParameters(4, 100, 10, block_type = 'GRU')
            tuner = DeepForecasterTuner()
            tuner.set_tuned_hps(n_rec_layers=[1,2], units=[[512, 256],[132, 80]])
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

    def build_hypermodel(self, hp: keras_tuner.HyperParameters) -> Sequential:
        """
        Build hypermodel takes an argument from which to sample hyperparameters.

        Args:
            hp: keras_tuner.HyperParameters, Hyperparameter object of Keras Tuner
            (to define the search space for the hyperparameter values).

        Returns:
            Keras `Sequential` model, forecasting hypermodel.
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

    @staticmethod
    def _add_hidden_layer(model: Sequential, block_type: str,
                          units: int, activation: str, last_layer: bool = False) -> Sequential:
        """
        Add new hidden recurrent layer into the model.

        Args:
            model: Keras `Sequential` model, forecasting hypermodel.
            block_type: String, recurrent block type.
            units: Integer, number of units of layer.
            activation: String, hidden layer activation function.
            last_layer: Boolean, it's last layer or not. Defaults to False.

        Returns:
            Keras `Sequential` model, forecasting hypermodel.
        """
        return_sequences = not last_layer

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

    def _create_tuner(self, tuner_type: str, max_trials: int, project_name: str, use_validation_data: bool):
        """
        Create tuner.

        Args:
            tuner_type: 'GridSearch', 'RandomSearch', 'BayesianOptimization', 'Hyperband', name of keras Tuner class.
            max_trials: Integer, the total number of trials (model configurations) to test at most.
            project_name: A string, the name to use as prefix for files saved by this `Tuner`.
            use_validation_data: Boolean, use validation data.
                on which to evaluate the loss at the end of each epoch or not.
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
                                                 ['GridSearch', 'RandomSearch', 'BayesianOptimization', 'Hyperband'])
        # Initialize tuner to run the model.
        if use_validation_data:
            objective = 'val_loss'
        else:
            objective = 'loss'
        if tuner_type == 'RandomSearch':
            self.tuner = keras_tuner.RandomSearch(
                hypermodel=self.build_hypermodel,
                objective=objective,
                max_trials=max_trials,  # the number of different models to try
                project_name=project_name,
                overwrite=True
            )
        elif tuner_type == 'BayesianOptimization':
            self.tuner = keras_tuner.BayesianOptimization(
                hypermodel=self.build_hypermodel,
                objective=objective,
                max_trials=max_trials,
                project_name=project_name,
                overwrite=True
            )
        elif tuner_type == 'Hyperband':
            self.tuner = keras_tuner.Hyperband(
                hypermodel=self.build_hypermodel,
                objective=objective,
                project_name=project_name,
                overwrite=True
            )
        elif tuner_type == 'GridSearch':
            self.tuner = keras_tuner.GridSearch(
                hypermodel=self.build_hypermodel,
                objective=objective,
                max_trials=max_trials,
                project_name=project_name,
                overwrite=True
            )

    def _search(self, x: np.ndarray, y: np.ndarray, validation_data: tuple, batch_size: int, epochs: int):
        """
        Performs a search for best hyperparameter configuations.

        Args:
            x: Numpy array, training input data.
            y: Numpy array, target data.
            validation_data: A tuple `(x_val, y_val)` of Numpy arrays  or `None`.
                Data on which to evaluate the loss at the end of each epoch.
            batch_size: Integer, number of samples per gradient update.
            epochs: Integer, number of epochs to train the model.
        """
        print(self.tuner.search_space_summary())
        # Run the search
        if validation_data is not None:
            self.tuner.search(x, y, validation_data=validation_data,
                              batch_size=batch_size, epochs=epochs,
                              shuffle=False,
                              callbacks=[EarlyStopping('val_loss', patience=1)])
        else:
            self.tuner.search(x, y, batch_size=batch_size, epochs=epochs,
                              shuffle=False,
                              callbacks=[EarlyStopping('loss', patience=1)])

    def find_best_models(self,
                         x: np.ndarray,
                         y: np.ndarray,
                         validation_data: tuple = None,
                         tuner_type: str = 'RandomSearch',
                         n_models: int = 1,
                         max_trials: int = 10,
                         batch_size: int = 128,
                         epochs: int = 10,
                         project_name: str = 'forecastate_tuner') -> list:
        """
        Returns the best forecasting model(s), as determined by the objective.

        Args:
            x: Numpy array, training input data.
            y: Numpy array, target data.
            validation_data: A tuple `(x_val, y_val)` of Numpy arrays or `None`.
                Data on which to evaluate the loss at the end of each epoch. Defaults to `None`.
            tuner_type: 'GridSearch', 'RandomSearch', 'BayesianOptimization', 'Hyperband', name of keras Tuner class.
                Defaults to 1.
            n_models: Integer, optional number of best models to return. Defaults to 1.
            max_trials: Integer, the total number of trials (model configurations) to test at most. Defaults to 10.
            batch_size: Integer, number of samples per gradient update. Defaults to 128.
            epochs: Integer, number of epochs to train the model. Defaults to 10.
            project_name: A string, the name to use as prefix for files saved by this `Tuner`.

        Returns:
            List of trained models sorted from the best to the worst.
        """
        self._create_tuner(tuner_type, max_trials, project_name, validation_data is not None)
        self._search(x, y, validation_data, batch_size, epochs)

        print("Results summary")
        print("Showing %d best trials" % n_models)

        for trial in self.tuner.oracle.get_best_trials(n_models):
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

        best_tuner_models = self.tuner.get_best_models(n_models)
        best_models = [DeepForecaster(from_config=tuner_model.get_config()) for tuner_model in best_tuner_models]
        return best_models


class ForecastEstimator:
    """
    Class for evaluating the quality of the forecasting model.

    Args:
        quality: Pandas `DataFrame`, matrix of forecasting quality metrics.
        features_names: List, name of features.
        true: Numpy array, data of true values.
        pred: Dict of Numpy arrays, data of predicted values for each forecasting models.
        first_batch: Numpy array, data of first time window in input data. Optional.

    Examples:
        estimator = ForecastEstimator()
    """

    def __init__(self, features_names: list = None):
        """
        Initialization.

        Args:
            features_names: List or `None`, name of features.
        """
        self.first_batch = None
        self.true = None
        self.pred = {}
        self.features_names = features_names
        self.quality = pd.DataFrame()

    def set_true_values(self, true: np.ndarray):
        """
        Set true output data.

        Args:
            true: Numpy array, data of true values.
        """
        assert len(true.shape) in [1, 2, 3], 'True data must be 1D, 2D or 3d numpy array'
        if len(true.shape) == 1:
            true = np.reshape(true, (1, true.shape[0], 1))
        if len(true.shape) == 2:
            true = np.reshape(true, (1, true.shape[0], true.shape[1]))
        self.true = true

    def set_pred_values(self, pred: np.ndarray, model_name: str = 'naive'):
        """
        Set forecasted output data.

        Args:
            pred: Numpy array, data of predicted values.
            model_name: A string, name of forecasting model. Defaults to 'naive'.
        """
        model_name = checker.check_is_type_param(model_name, model_name, str)
        assert len(pred.shape) in [1, 2, 3], 'Predicted data must be 1D, 2D or 3d numpy array'
        if len(pred.shape) == 1:
            pred = np.reshape(pred, (1, pred.shape[0], 1))
        if len(pred.shape) == 2:
            pred = np.reshape(pred, (1, pred.shape[0], pred.shape[1]))

        self.pred[model_name] = pred

    def set_first_batch(self, first_batch: np.ndarray):
        """
        Set first input time window data.

        Args:
            first_batch: Numpy array, data of first time window in input data.
        """
        self.first_batch = first_batch

    def estimate(self, how='features'):
        """
        Evaluation.

        Args:
            how: 'features' or 'timesteps', method of evaluation. Default to 'features'.
               If 'features', then metrics are calculated for each feature and for all data.
               If 'timesteps', then metrics are calculated for each time step.
        """
        assert how in ['features', 'timesteps'], 'Unknown "how" argument'

        assert self.true is not None, 'No true values for estimate'
        self.quality = pd.DataFrame()

        if how == 'features':
            if self.features_names is None:
                features_names = [str(i) for i in range(self.true.shape[2])]
            else:
                features_names = self.features_names.copy()
            features_names.append('ALL_FEATURES')

            true_reshaped = self.true.reshape((self.true.shape[0] * self.true.shape[1], self.true.shape[2]))

            for model_name, pred_vals in self.pred.items():
                assert (len(self.true) == len(pred_vals)), f'The length of' + model_name + \
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

            self.quality.index = features_names

        if how == 'timesteps':
            self.quality.index = range(self.true.shape[0])
            true_reshaped = self.true.reshape((self.true.shape[0], self.true.shape[1] * self.true.shape[2]))
            for model_name, pred_vals in self.pred.items():
                pred_reshaped = pred_vals.reshape((self.true.shape[0], self.true.shape[1] * self.true.shape[2]))
                mse = np.mean(np.square(np.subtract(pred_reshaped, true_reshaped)), axis=1)
                self.quality[model_name + '_MSE'] = mse
                mae = np.mean(np.abs(pred_reshaped - true_reshaped), axis=1)
                self.quality[model_name + '_MAE'] = mae

    def save_quality(self, filename: str):
        """
        Save evaluation results to file.

        Args:
            filename: A string, name of the file to save
        """
        if not os.path.exists('forecaster_results/'):
            os.makedirs('forecaster_results/')

        self.quality.to_csv('forecaster_results/' + filename + '.csv',
                            index_label='feature')

    def save_pred_result(self, dataset_name):
        """
        Save forecasting models results to file.
        Args:
            dataset_name: A string, name of the file to save
        """
        if not os.path.exists('forecaster_results/'):
            os.makedirs('forecaster_results/')

        for model_name, pred_vals in self.pred.items():
            filename = 'forecaster_results/' + dataset_name + '_' + model_name + '.npy'
            np.save(filename, pred_vals)
            print('Save ' + filename)

    def _draw_feature(self, i_feature: int, ax: object, data_size: int, draw_horizon: int):
        """
        Draw forecasting results for all models and one feature.
        Args:
            i_feature: Integer, index of feature in `feature_names`.
            ax: `matplotlib.axes.Axes` object
            data_size: Integer, number of drawing points in data.
            draw_horizon: Integer, number of drawing points for each time windows.
        """

        def draw_windows(data, start_x=0, color='indigo', label='Data', alpha=1.0):
            timeline = []
            for p in range(data.shape[0]):
                y = data[p]
                timeline.append(y[0])
                x = range(p + start_x, p + y.shape[0] + start_x)
                if p == (data.shape[0] - 1):
                    ax.plot(x, y, marker='.', color=color, label=label, alpha=alpha, markersize=2)
                else:
                    ax.plot(x, y, marker='.', color=color, alpha=alpha, markersize=2)
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
            draw_windows(self.true[:data_size, :draw_horizon, i_feature], start_x=target_start_point, color='green',
                         label='True')

        if self.pred:
            model_names = list(self.pred.keys())
            color_dicts = {}
            if 'naive' in model_names:
                model_names.remove('naive')
                color_dicts['naive'] = 'grey'

            color_map = plt.cm.get_cmap('twilight', len(model_names) + 2)
            for i, model_name in enumerate(model_names):
                color_dicts[model_name] = color_map(i)

            for model_name, pred_vals in self.pred.items():
                if connection_line:
                    connection_line['y'][1] = pred_vals[0][0, i_feature]
                    plt.plot(connection_line['x'], connection_line['y'], marker='.', color=color_dicts[model_name])
                draw_windows(pred_vals[:data_size, :draw_horizon, i_feature], start_x=target_start_point,
                             color=color_dicts[model_name],
                             label=model_name.capitalize())

        ax.legend(fontsize=8)

    def draw(self, size=1000, draw_horizon=None, features_names=None):
        """
        Draw forecasting results.
        Args:
            size: Integer, number of drawing points in data. Defaults to 1000.
            draw_horizon: Integer, number of drawing points for each time windows. Defaults to forecasting horizon.
            features_names: List or `None`, list of feature for drawing.
        """
        if features_names is None:
            if self.features_names is None:
                features_names = [str(i) for i in range(self.true.shape[2])]
            else:
                features_names = self.features_names.copy()

        if draw_horizon is None:
            draw_horizon = self.true.shape[1]

        n = len(features_names)
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

        fig, axs = plt.subplots(nrows, ncols, sharex='col', figsize=(ncols * 6, nrows * 3))
        fig.tight_layout(pad=2.3)
        fig.supxlabel('Time')

        i_feature = 0

        for i in range(nrows):
            if ncols == 1:
                if nrows == 1:
                    self._draw_feature(i_feature, axs, size, draw_horizon)
                    axs.set_ylabel(features_names[i_feature])
                else:
                    self._draw_feature(i_feature, axs[i], size, draw_horizon)
                    axs[i].set_ylabel(features_names[i_feature])
                i_feature = i_feature + 1
            else:
                for j in range(ncols):
                    if i_feature > n:
                        break
                    self._draw_feature(i_feature, axs[i, j], size, draw_horizon)
                    axs[i, j].set_ylabel(features_names[i_feature])
                    i_feature = i_feature + 1
        plt.show()
