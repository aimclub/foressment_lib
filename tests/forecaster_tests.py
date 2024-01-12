#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import unittest
import inspect
import numpy as np
import foressment_ai as foras


class TestForecasterParameters(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestForecasterParameters, self).__init__(*args, **kwargs)
        TestForecasterParameters.n = 1

    def setUp(self):
        print('Test {} BEGIN'.format(TestForecasterParameters.n))

    def tearDown(self):
        print('Test {} END'.format(TestForecasterParameters.n))
        TestForecasterParameters.n += 1

    def test_is_not_none(self):
        print(inspect.stack()[0][3])
        message = 'The object of class ForecasterParameters can\'t not be created'
        self.assertIsNotNone(foras.ForecasterParameters(), message)

    def test_set_incorrect_params(self):
        print(inspect.stack()[0][3])
        message = 'The object of class ForecasterParameters sets incorrect parameters'
        params = foras.ForecasterParameters()
        incorrect_value = -100
        for param_name in params.__slots__:
            with self.subTest():
                with self.assertRaises(AssertionError, msg=message):
                    params.__setattr__(name=param_name, val=incorrect_value)

    def test_set_incorrect_argument(self):
        print(inspect.stack()[0][3])
        message = 'The object of class ForecasterParameters sets incorrect argument'
        params = foras.ForecasterParameters()
        with self.assertRaises(AttributeError, msg=message):
            params.block_type = 'GRU'


class TestDeepForecasterParameters(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestDeepForecasterParameters, self).__init__(*args, **kwargs)
        TestDeepForecasterParameters.n = 1

    def setUp(self):
        print('Test {} BEGIN'.format(TestDeepForecasterParameters.n))

    def tearDown(self):
        print('Test {} END'.format(TestDeepForecasterParameters.n))
        TestDeepForecasterParameters.n += 1

    def test_is_not_none(self):
        print(inspect.stack()[0][3])
        message = 'The object of class ForecasterParameters can\'t not be created'
        self.assertIsNotNone(foras.DeepForecasterParameters(), message)

    def test_set_incorrect_params(self):
        print(inspect.stack()[0][3])
        message = 'The object of class ForecasterParameters sets incorrect parameters'
        params = foras.DeepForecasterParameters()
        incorrect_value = -100
        for param_name in params.__slots__:
            with self.subTest():
                with self.assertRaises(AssertionError, msg=message):
                    params.__setattr__(name=param_name, val=incorrect_value)


class TestTSGenerator(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestTSGenerator, self).__init__(*args, **kwargs)
        TestTSGenerator.n = 1

    def setUp(self):
        print('Test {} BEGIN'.format(TestTSGenerator.n))

    def tearDown(self):
        print('Test {} END'.format(TestTSGenerator.n))
        TestTSGenerator.n += 1

    def test_is_not_none(self):
        print(inspect.stack()[0][3])
        message = 'The object of class TSGenerator can\'t not be created'
        params = foras.ForecasterParameters()
        data = np.random.randint(1, 10, 10000)
        self.assertIsNotNone(foras.TSGenerator(data, params), message)

    def test_set_incorrect_data(self):
        print(inspect.stack()[0][3])
        message = 'The object of class TSGenerator can\'t not be created'
        params = foras.ForecasterParameters()
        data = np.random.randint(1, 10, size=(10000, 1, 1))
        with self.assertRaises(AssertionError, msg=message):
            foras.TSGenerator(data, params)

    def test_set_insufficient_data(self):
        print(inspect.stack()[0][3])
        message = 'The object of class TSGenerator can\'t not be created'
        params = foras.ForecasterParameters()
        data = np.random.randint(1, 10, params.look_back_length - 1)
        with self.assertRaises(AssertionError, msg=message):
            foras.TSGenerator(data, params)

    def test_get_data_and_targets(self):
        print(inspect.stack()[0][3])
        params = foras.ForecasterParameters()
        data = np.random.randint(1, 10, 10000)
        tg = foras.TSGenerator(data, params)

        with self.subTest(0):
            self.assertIsNotNone(tg.get_data(), msg='Unable to retrieve data')
            with self.assertRaises(AssertionError, msg='Unable to retrieve data'):
                tg.get_data(window_id=100000)

        with self.subTest(1):
            self.assertIsNotNone(tg.get_targets(), msg='Unable to retrieve targets')
            with self.assertRaises(AssertionError, msg='Unable to retrieve targets'):
                tg.get_targets(window_id=100000)


class TestNaiveForecaster(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestNaiveForecaster, self).__init__(*args, **kwargs)
        TestNaiveForecaster.n = 1

    def setUp(self):
        print('Test {} BEGIN'.format(TestNaiveForecaster.n))

    def tearDown(self):
        print('Test {} END'.format(TestNaiveForecaster.n))
        TestNaiveForecaster.n += 1

    def test_is_not_none(self):
        print(inspect.stack()[0][3])
        message = 'The object of class NaiveForecaster can\'t not be created'
        params = foras.ForecasterParameters()
        self.assertIsNotNone(foras.NaiveForecaster(params), message)

    def test_forecasting(self):
        print(inspect.stack()[0][3])
        message = 'Forecasting failed'
        params = foras.ForecasterParameters()
        nf = foras.NaiveForecaster(params)
        data = np.random.randint(1, 10, size=(10, 10, 1))
        with self.subTest(0):
            self.assertIsNotNone(nf.forecasting(data), message)
        with self.subTest(1):
            self.assertIsNotNone(nf.forecasting(data, 10), message)

    def test_forecasting_incorrect_data(self):
        print(inspect.stack()[0][3])
        message = 'Forecasting failed'
        params = foras.ForecasterParameters()
        nf = foras.NaiveForecaster(params)
        data = np.random.randint(1, 10, size=(100, 1, 1, 1))
        with self.assertRaises(AssertionError, msg=message):
            nf.forecasting(data)

    def test_forecasting_incorrect_horizon(self):
        print(inspect.stack()[0][3])
        message = 'Forecasting failed'
        params = foras.ForecasterParameters()
        nf = foras.NaiveForecaster(params)
        data = np.random.randint(1, 10, size=100)
        with self.assertRaises(AssertionError, msg=message):
            nf.forecasting(data, forecasting_data_length=-10)


class TestDeepForecaster(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestDeepForecaster, self).__init__(*args, **kwargs)
        TestDeepForecaster.n = 1

    def setUp(self):
        print('Test {} BEGIN'.format(TestDeepForecaster.n))

    def tearDown(self):
        print('Test {} END'.format(TestDeepForecaster.n))
        TestDeepForecaster.n += 1

    def test_is_not_none(self):
        print(inspect.stack()[0][3])
        message = 'The object of class DeepForecaster can\'t not be created'
        params = foras.DeepForecasterParameters()
        self.assertIsNotNone(foras.DeepForecaster(params), message)

    def test_forecasting_model(self):
        print(inspect.stack()[0][3])
        params = foras.DeepForecasterParameters()
        ts = foras.TSGenerator(np.random.randint(1, 10, 10000), params)
        x = ts.get_data()
        y = ts.get_data()
        with self.subTest(0):
            aif = foras.DeepForecaster(params)
            aif.build_model()
            self.assertIsNotNone(aif.model, 'The model of class DeepForecaster can\'t not be created')
        with self.subTest(1):
            aif.train(x, y, n_epochs=3, batch_size=256)
            self.assertIsNotNone(aif.history, 'Model training failed')
        with self.subTest(2):
            new_data = np.random.randint(1, 10, size=(10, 10, 1))
            self.assertIsNotNone(aif.forecasting(new_data), 'Forecasting failed')
        with self.subTest(3):
            self.assertIsNotNone(aif.forecasting(new_data, 10), 'Forecasting failed')


class TestDeepForecasterTuner(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestDeepForecasterTuner, self).__init__(*args, **kwargs)
        TestDeepForecasterTuner.n = 1

    def setUp(self):
        print('Test {} BEGIN'.format(TestDeepForecasterTuner.n))

    def tearDown(self):
        print('Test {} END'.format(TestDeepForecasterTuner.n))
        TestDeepForecasterTuner.n += 1

    def test_is_not_none(self):
        print(inspect.stack()[0][3])
        message = 'The object of class DeepForecaster can\'t not be created'
        params = foras.DeepForecasterParameters()
        self.assertIsNotNone(foras.DeepForecasterTuner(params), message)

    def test_set_tuned_hps(self):
        params = foras.DeepForecasterParameters()
        tuner = foras.DeepForecasterTuner(params)
        tuner.set_tuned_hps(units=[[512, 400, 316], [256, 160, 80], [64, 32, 16]],
                            n_rec_layers=[1, 2, 3],
                            dropout=[0.0, 0.01, 0.5],
                            hidden_activation=['tanh', 'relu'],
                            output_activation=['linear', 'sigmoid'])
        self.assertIsNotNone(tuner.hp_choices)

    def test_find_best_models(self):
        params = foras.DeepForecasterParameters()
        ts = foras.TSGenerator(np.random.randint(1, 10, 1000), params)
        x = ts.get_data()
        y = ts.get_data()

        tuner = foras.DeepForecasterTuner(params)
        tuner.find_best_models(x, y, epochs=3, n_models=1, max_trials=3)


class TestForecastEstimator(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestForecastEstimator, self).__init__(*args, **kwargs)
        TestForecastEstimator.n = 1

    def setUp(self):
        print('Test {} BEGIN'.format(TestForecastEstimator.n))

    def tearDown(self):
        print('Test {} END'.format(TestForecastEstimator.n))
        TestForecastEstimator.n += 1

    def test_is_not_none(self):
        print(inspect.stack()[0][3])
        message = 'The object of class ForecastEstimator can\'t not be created'
        self.assertIsNotNone(foras.ForecastEstimator(), message)

    def test_estimate(self):
        message = 'Forecasting quality evaluation failed'
        est = foras.ForecastEstimator()
        est.set_true_values(np.array([0, 0, 1, 0, 2, 1, 1, 0, 0, 1]))
        est.set_pred_values(np.array([0, 1, 1, 0, 0, 0, 1, 0, 0, 1]))
        est.estimate()
        self.assertFalse(est.quality.empty, message)


if __name__ == '__main__':
    unittest.main()
