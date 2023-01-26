#!/usr/bin/python
# -*- coding: utf-8 -*-

from aopssop import AIForecaster, DataScaler, ForecastEstimator
from examples.apssop_examples import DataLoader
import unittest
import inspect
import os.path
import numpy as np


class TestAPSSOP(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestAPSSOP, self).__init__(*args, **kwargs)
        TestAPSSOP.n = 1

    def setUp(self):
        print('Test {} BEGIN'.format(TestAPSSOP.n))

    def tearDown(self):
        print('Test {} END'.format(TestAPSSOP.n))
        TestAPSSOP.n += 1

    def test_forecasting_model_is_not_none(self):
        print(inspect.stack()[0][3])
        message = 'The object of class AIForecaster could not be created'
        self.assertIsNotNone(AIForecaster(10, 10, 'models/test_model'), message)

    def test_forecasting_model_train(self):
        print(inspect.stack()[0][3])
        message = 'Error while calling train'
        data = DataLoader('test').data
        forecasting_model = AIForecaster(10, data.shape[1],
                                          'models/test_model.h5')
        generator = forecasting_model.data_to_generator(data.values)
        try:
            forecasting_model.train(generator, save=False)
        except:
            self.assertTrue(False, message)

    def test_forecasting_model_predict(self):
        print(inspect.stack()[0][3])
        message = 'Error while calling forecasting'
        data = DataLoader('test').data
        forecasting_model = AIForecaster(10, data.shape[1],
                                          'models/test_model.h5')
        generator = forecasting_model.data_to_generator(data.values)
        forecasting_model.train(generator, save=False)
        try:
            forecasting_model.forecasting(forecasting_model.get_batch(generator),
                                          forecasting_data_length=10)
            print()
        except:
            self.assertTrue(False, message)

    def test_forecasting_model_save(self):
        print(inspect.stack()[0][3])
        AIForecaster(10, 10, 'models/test_model.h5').save_model()
        message = 'The forecasting model could not be saved into the file'
        self.assertTrue(os.path.isfile('models/test_model.h5'), message)

    def test_forecasting_model_open(self):
        print(inspect.stack()[0][3])
        message = 'File with forecasting model does not exist'
        self.assertIsNotNone(AIForecaster(10, 10, 'models/test_model.h5', open=True), message)

    def test_generator_create(self):
        data = range(0, 1000)
        message = 'The object of class TimeseriesGenerator could not be created'
        self.assertIsNotNone(AIForecaster(10, 10, 'models/test_model').data_to_generator(data), message)
        print()

    # def test_data_load(self):
    #     message = 'The dataset name is not correct'
    #     for dataset_name in ['hai', 'alarms', 'edge-iiotset', 'test']:
    #         self.assertIsNotNone(DataLoaderAndPreprocessor(dataset_name).data, message)

    def test_dataframe_split(self):
        message = 'Data split failed'
        for i in [0.1, 0.25, 0.5, 0.8, 0.99]:
            self.assertIsNot(DataLoader('test').train_test_split(i), (None, None), message)

    def test_normalization_model_is_not_none(self):
        print(inspect.stack()[0][3])
        message = 'The object of class DataScaler could not be created'
        self.assertIsNotNone(DataScaler(scaler_path='models/scaler.pkl'), message)

    def test_normalization_model_train(self):
        print(inspect.stack()[0][3])
        message = 'Error while calling fit normalization model'
        data = DataLoader('test').data
        try:
            DataScaler(scaler_path='models/scaler.pkl').fit(data, save=False)
        except:
            self.assertTrue(False, message)

    def test_normalization_model_transform(self):
        print(inspect.stack()[0][3])
        message = 'Error while calling transform with normalization model'
        data = DataLoader('test').data
        scaler = DataScaler(scaler_path='models/scaler.pkl')
        scaler.fit(data, save=False)
        try:
            scaler.transform(data)
        except:
            self.assertTrue(False, message)

    def test_normalization_model_inverse(self):
        print(inspect.stack()[0][3])
        message = 'Error while inverse data with normalization model'
        data = DataLoader('test').data
        scaler = DataScaler(scaler_path='models/scaler.pkl')
        scaler.fit(data, save=False)
        scaled_data = scaler.transform(data)
        inverted_data = np.round(scaler.inverse(scaled_data), 0)
        self.assertTrue(np.array_equal(data.values.astype(float), inverted_data), message)

    def test_normalization_model_save(self):
        print(inspect.stack()[0][3])
        message = 'The normalization model could not be saved into the file'
        DataScaler(scaler_path='models/scaler.pkl').save()
        self.assertTrue(os.path.isfile('models/scaler.pkl'), message)

    def test_normalization_model_open(self):
        print(inspect.stack()[0][3])
        message = 'File with normalization model does not exist'
        self.assertIsNotNone(DataScaler('models/scaler.pkl', open=True), message)

    def test_estimator_mae_results(self):
        message = 'Forecasting quality evaluation failed in MAE'
        true = [0, 0, 1, 0, 2, 1, 1, 0, 0, 1]
        pred = [0, 1, 1, 0, 0, 0, 1, 0, 1, 1]
        result = ForecastEstimator().estimate(true, pred)
        self.assertEqual(result.loc['ALL_FEATURES', 'MAE'], 0.5, message)

    def test_estimator_mse_results(self):
        message = 'Forecasting quality evaluation failed in MSE'
        true = [0, 0, 1, 0, 2, 1, 1, 0, 0, 1]
        pred = [0, 1, 1, 0, 0, 0, 1, 0, 1, 1]
        result = ForecastEstimator().estimate(true, pred)
        self.assertEqual(result.loc['ALL_FEATURES', 'MSE'], 0.7, message)

    def test_estimator(self):
        message = 'Forecasting quality evaluation failed'
        true = [0, 0, 1, 0, 2, 1, 1, 0, 0, 1]
        pred = [0, 1, 1, 0, 0, 0]
        result = ForecastEstimator().estimate(true, pred)
        self.assertTrue(result.empty, message)


if __name__ == '__main__':
    unittest.main()
