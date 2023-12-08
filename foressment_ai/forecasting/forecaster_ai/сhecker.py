from inspect import getmembers, isfunction, signature
import keras.activations as activations
from keras import optimizers
import os
from typing import Type


class ParamChecker:
    def __init__(self, model_type='forecaster'):
        self.max_num_units = 512
        self.default_units_step = 64

        activations_names = [name[0] for name in getmembers(activations, isfunction)
                             if 'x' in signature(name[1]).parameters.keys()]

        optimization_names = ['adadelta', 'adagrad', 'adam', 'adamax', 'experimentaladadelta', 'experimentaladagrad',
                              'experimentaladam', 'experimentalsgd', 'nadam', 'rmsprop', 'sgd', 'ftrl',
                              'lossscaleoptimizer',
                              'lossscaleoptimizerv3', 'lossscaleoptimizerv1']

        if model_type == 'forecaster':
            self.params_requirements = {
                'look_back_length': {'is_type': int, 'in_range': (0, None)},
                'n_features': {'is_type': int, 'in_range': (0, None)},
                'horizon': {'is_type': int, 'in_range': (0, None)},
                'n_rec_layers': {'is_type': int, 'in_range': (0, 100)},
                'block_type': {'is_type': str, 'in_list': ['SimpleRNN', 'LSTM', 'GRU']},
                'units': {'is_type': dict},
                'units_of_layer': {'is_type': int, 'in_range': [1, self.max_num_units]},
                'unit_step': {'is_type': int, 'in_range': (0, self.max_num_units - 1)},
                'dropout': {'is_type': (float, int), 'in_range': [0, 1]},
                'hidden_activation': {'is_type': str, 'in_list': activations_names},
                'output_activation': {'is_type': str, 'in_list': activations_names},
                'optimizer': {'is_type': optimizers.legacy.Adam},
                'loss': {'is_type': str, 'in_list': ['mse', 'mae']}
            }

    def check_param(self, param, param_name):
        requirements = self.params_requirements[param_name]
        for req_type, req in requirements.items():
            if req_type == 'is_type':
                param = self.check_is_type_param(param, param_name, req)
            if req_type == 'in_list':
                param = self.check_in_list_param(param, param_name, req)
            if req_type == 'in_range':
                param = self.check_in_range_param(param, param_name, req)
        return param

    def check_in_range_param(self, param, param_name, range):
        min_val = range[0]
        max_val = range[1]

        if isinstance(range, tuple):
            if min_val is not None:
                assert (param > min_val), 'Value of the "{0}" argument must be more than {1} '.format(
                    param_name, min_val)
            if max_val is not None:
                assert (param <= max_val), 'Value of the "{0}" argument must be less than {1} '.format(
                    param_name, max_val)
            return param

        if isinstance(range, list):
            if min_val is not None:
                assert (param >= min_val), 'Value of the "{0}" argument must be more than {1} or equal'.format(
                    param_name, min_val)
            if max_val is not None:
                assert (param <= max_val), 'Value of the "{0}" argument must be less than {1} or equal'.format(
                    param_name, max_val)
            return param

    def check_in_list_param(self, param, param_name, list_of_values):
        assert (param in list_of_values), 'Value of the "{0}" argument must be in list {1}'.format(
            param_name, list_of_values)
        return param

    def check_is_type_param(self, param, param_name, stype):
        assert (isinstance(param, stype)), 'Type of the "{0}" argument must be {1}'.format(param_name, stype)
        return param

    @staticmethod
    def check_file_is_exist(filename):
        assert (os.path.isfile(filename)), 'File {0} does not exist'.format(filename)


