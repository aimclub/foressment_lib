#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest
import foressment_ai as foras

import inspect
import os
import os.path

import numpy as np
import pandas as pd
import tempfile
import shutil
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

import matplotlib
matplotlib.use('Agg')

class TestAssessor(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestAssessor, self).__init__(*args, **kwargs)
        TestAssessor.n = 1
        self.xor_ds = {'features': np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), 'labels': np.array([[0], [1], [1], [0]])}

    def setUp(self):
        print('Test {} BEGIN'.format(TestAssessor.n))

    def tearDown(self):
        print('Test {} END'.format(TestAssessor.n))
        TestAssessor.n += 1

    def test_sc_is_not_none(self):
        print(inspect.stack()[0][3])
        self.assertFalse(foras.SAIClassifier('neural_network', 10, 1) is None,
                         'The object of class SAIClassifier could not be created')

    def test_sc_type_is_correct(self):
        print(inspect.stack()[0][3])
        classifier_types = ['decision_tree', 'naive_bayes', 'logistic_regression', 'neural_network']
        for c in classifier_types:
            classifier = foras.SAIClassifier(c, 10, 1, plot=False)
            self.assertTrue(classifier.cls_type in classifier_types, 'The classifier type is not correct')

    def test_sc_fit(self):
        print(inspect.stack()[0][3])
        classifier_types = ['decision_tree', 'naive_bayes', 'logistic_regression', 'neural_network']
        in_size = np.shape(self.xor_ds['features'])[1]
        out_size = np.shape(self.xor_ds['labels'])[1]
        for c in classifier_types:
            classifier = foras.SAIClassifier(c, in_size, out_size, plot=False)
            try:
                classifier.fit(self.xor_ds['features'], self.xor_ds['labels'])
            except:
                self.assertTrue(False, 'Error while calling fit')

    def test_sc_predict(self):
        print(inspect.stack()[0][3])
        classifier_types = ['decision_tree', 'naive_bayes', 'logistic_regression', 'neural_network']
        in_size = np.shape(self.xor_ds['features'])[1]
        out_size = np.shape(self.xor_ds['labels'])[1]
        for c in classifier_types:
            classifier = foras.SAIClassifier(c, in_size, out_size, plot=False)
            classifier.fit(self.xor_ds['features'], self.xor_ds['labels'])
            try:
                classifier.predict(self.xor_ds['features'])
            except:
                self.assertTrue(False, 'Error while calling predict')

    def test_sc_save(self):
        print(inspect.stack()[0][3])
        classifier_types = ['decision_tree', 'naive_bayes', 'logistic_regression', 'neural_network']
        in_size = np.shape(self.xor_ds['features'])[1]
        out_size = np.shape(self.xor_ds['labels'])[1]
        for c in classifier_types:
            classifier = foras.SAIClassifier(c, in_size, out_size, plot=False)
            classifier.fit(self.xor_ds['features'], self.xor_ds['labels'])
            f = './' + c + '.bin'
            if os.path.isfile(f):
                os.remove(f)
            classifier.save(f)
            self.assertTrue(os.path.isfile(f) or os.path.isfile(f + '.index'),
                            'The classifier could not be saved into the file')

    def test_sc_load(self):
        print(inspect.stack()[0][3])
        classifier_types = ['decision_tree', 'naive_bayes', 'logistic_regression', 'neural_network']
        in_size = np.shape(self.xor_ds['features'])[1]
        out_size = np.shape(self.xor_ds['labels'])[1]
        for c in classifier_types:
            classifier = foras.SAIClassifier(c, in_size, out_size, plot=False)
            f = './' + c + '.bin'
            classifier.save(f)
            try:
                classifier.load(f)
            except:
                self.assertTrue(False, 'Error while calling load')

    def test_fd_is_not_none(self):
        print(inspect.stack()[0][3])
        f = '../hai/hai-20.07/test1.csv.gz'
        if os.path.isfile(f):
            self.assertFalse(foras.FormatDetector(f) is None, 'The object of class FormatDetector could not be created')


    def test_fd_delimiter_is_correct(self):
        print(inspect.stack()[0][3])
        f = '../hai/hai-20.07/test1.csv.gz'
        if os.path.isfile(f):
            fd = foras.FormatDetector(f)
            self.assertTrue(fd.d in [';', ','], 'The delimiter is not correct')

    def test_dl_is_not_none(self):
        print(inspect.stack()[0][3])
        f = '../hai/hai-20.07/test1.csv.gz'

        class DataLoaderExample(foras.DataLoader):
            def load(self, file):
                pass

        if os.path.isfile(f):
            self.assertFalse(DataLoaderExample(f, 0, 0) is None,
                             'The object of class DataLoaderExample could not be created')

    def test_ce_is_not_none(self):
        print(inspect.stack()[0][3])
        classifier_types = ['decision_tree', 'naive_bayes', 'logistic_regression', 'neural_network']
        in_size = np.shape(self.xor_ds['features'])[1]
        out_size = np.shape(self.xor_ds['labels'])[1]
        classifiers = [foras.SAIClassifier(c, in_size, out_size, plot=False) for c in classifier_types]
        for classifier in classifiers:
            self.assertFalse(foras.ClsEstimator(self.xor_ds['features'], self.xor_ds['labels'], self.xor_ds['labels'],
                                          [classifier]) is None,
                             'The object of class ClsEstimator could not be created')

    def test_ce_estimate(self):
        print(inspect.stack()[0][3])
        classifier_types = ['decision_tree', 'naive_bayes', 'logistic_regression', 'neural_network']
        in_size = np.shape(self.xor_ds['features'])[1]
        out_size = np.shape(self.xor_ds['labels'])[1]
        classifiers = [foras.SAIClassifier(c, in_size, out_size, plot=False) for c in classifier_types]
        for classifier in classifiers:
            classifier.fit(self.xor_ds['features'], self.xor_ds['labels'])
        try:
            foras.ClsEstimator(self.xor_ds['features'], self.xor_ds['labels'], self.xor_ds['labels'], classifiers).estimate()
        except:
            self.assertTrue(False, 'Error while calling estimate')


def get_gan_data(param="clf"):
    if param == "re":
        X_train, _ = make_classification(n_samples=800, n_features=15, random_state=42)
        X_val, _ = make_classification(n_samples=200, n_features=15, random_state=42)
        return X_train,X_val
    elif param == "clf":
        X, y = make_classification(n_samples=1000, n_features=15, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
        return X_train, X_val, y_train, y_val

class GANTestCase(unittest.TestCase):
    def __init__(self, methodName: str = 'runTest'):
        super().__init__(methodName=methodName)
        X_train, X_val, y_train, y_val = get_gan_data("clf")
        # X_train, X_val = X_train.reshape(X_train.shape[0], X_train.shape[1], 1), X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        self.input_shape = X_train.shape[1:]
        self.X_train, self.X_val, self.y_train, self.y_val = X_train, X_val, y_train, y_val
        self.gan = foras.GAN(self.input_shape)

    def test_build_generator(self):
        self.assertIsNotNone(self.gan.build_generator())

    def test_build_discriminator(self):
        self.assertIsNotNone(self.gan.build_discriminator())

    def test_build_gan(self):
        self.assertIsNotNone(self.gan.build_gan())

    def test_train(self):
        history = self.gan.train(self.X_train, self.y_train, self.X_val, self.y_val, epochs=5, batch_size=32)
        self.assertIsNotNone(history)
        print(isinstance(history, dict))
        print("items in history: ", history.keys())

    @unittest.mock.patch('builtins.print')
    def test_print_classification_report(self, mock_print):
        X_test, y_test = make_classification(n_samples=100, n_features=15, random_state=42)
        model = foras.GAN(input_shape=self.input_shape)
        model.train(self.X_train, self.y_train, self.X_val, self.y_val, epochs=5, batch_size=32)
        model.print_classification_report(X_test, y_test)
        self.assertIsNone(model.print_classification_report(X_test, y_test, target_names=['class1', 'class2']))
        # Assert that the print function is called
        mock_print.assert_called()

    @unittest.mock.patch('matplotlib.pyplot.show')
    def test_draw_plot(self, mock_show):
        model = foras.GAN(input_shape=self.input_shape)
        model.train(self.X_train, self.y_train, self.X_val, self.y_val, epochs=5, batch_size=32)
        self.assertIsNone(model.draw_plot("accuracy"))
        mock_show.assert_called()


def get_fs_data():
    data = pd.DataFrame({'Alarm': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         'BridgeSpeedFeedback': [0, 0, -32, -1.26, 0, 0, 0, 0, 0, -10.08, -10.08],
                         'HoistMotorTorque': [0, 0, 75, 6, 0, 0, 0, 0, 0, -178, -178],
                         'BridgePosition': [16.575, 16.575, 16.676, 22.647, 16.51, 16.51, 16.523, 16.565, 16.414,
                                            17.079, 17.079],
                         'BridgeRopeAngle': [0.00273705, 0.00273705, 0.0691784, 0.00889362, 0.00868183, 0.00868183,
                                             0.00849418, 0.00621962, 0.0133168, -0.0340688, -0.0340688],
                         'BridgeMotorTorque': [0, 0, -1, 22, 0, 0, 0, 0, 0, -41, -41],
                         'HoistSpeedFeedback': [0, 0, -67, -13.9, 0, 0, 0, 0, 0, 0, 0],
                         'LoadTare': [-0.04, -0.04, -0.05, 0.06, -0.05, -0.05, -0.04, -0.04, -0.04, 0.98, 0.98],
                         'HoistPosition': [0.188, 0.188, 0.7829999999999999, 0.9359999999999999, 0.72, 0.72, 1.147,
                                           1.122, 0.805, 1.56, 1.56],
                         'TrolleyMotorTorque': [0, 0, 0, -3, 0, 0, 0, 0, 0, 0, 0],
                         'TrolleySpeedFeedback': [0, 0, 0, 15.53, 0, 0, 0, 0, 0, 0, 0],
                         'TrolleyPosition': [1.963, 1.963, 1.939, 9.19, 2.0380000000000003, 2.0380000000000003,
                                             2.082, 1.995, 2.045, 2.082, 2.082],
                         'TrolleyRopeAngle': [0.00429468, 0.00429468, -0.0100434, -0.00693946, 0.00392088,
                                              0.00353599, 0.00287688, 0.00390476, 0.00953259, -0.0273092,
                                              -0.0273092],
                         'Cycle': [1, 1, 2, 3, 4, 4, 5, 6, 7, 8, 8]
                         })

    X = data.drop(columns=['Cycle'])
    y = data['Cycle']

    return X, y


class TestFeatureSelector(unittest.TestCase):
    def test_elastic_net_selection(self):
        X, y = get_fs_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        fs = foras.FeatureSelector(method='elastic_net', params={'alpha': 0.1})
        selected_features = fs.fit_transform(X_train, y_train)

        # Assert that the selected features are of the expected shape
        self.assertEqual(selected_features.shape[1], len(fs.selected_features))

    def test_elastic_invalid(self):
        X, y = get_fs_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        with self.assertRaises(ValueError):
            fs = foras.FeatureSelector(method='elastic_net', params={'alpha': 2})
            fs.fit_transform(X_train, y_train)

    def test_pca_selection(self):
        X, y = get_fs_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        fs = foras.FeatureSelector(method='pca', params={'components': 5})
        transformed_features = fs.fit_transform(X_train)

        # Assert that the transformed features have the expected shape
        self.assertEqual(transformed_features.shape[1], fs.components)

    def test_pca_invalid(self):
        X, y = get_fs_data()
        # X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        with self.assertRaises(ValueError):
            fs = foras.FeatureSelector(method='pca', params={'components': 1.5})
            fs.fit_transform(X_train, y_train)


class TestDeepCNN(unittest.TestCase):
    def setUp(self):
        self.current_directory = os.path.dirname(os.path.realpath(__file__))
        self.temp_dir = tempfile.TemporaryDirectory(dir=self.current_directory)
        self.temp_model_path = os.path.join(self.temp_dir.name, "model.h5")
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=5, n_clusters_per_class=4, n_informative=5, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        self.input_shape = X_train.shape[1:]
        self.classes = len(set(y_train))
        y_train = to_categorical(y_train, self.classes)
        y_val = to_categorical(y_val, self.classes)
        self.X_train, self.X_val, self.y_train, self.y_val = X_train, X_val, y_train, y_val
        self.model = foras.DeepCNN(input_shape=self.input_shape, blocks=1, units=64, classes=self.classes)
            
    def test_model_build(self):
        model = foras.DeepCNN(input_shape=self.input_shape, blocks=1, units=64, classes=self.classes)
        self.assertIsNotNone(model.model)

    def test_model_fit(self):
        history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val), epochs=10, batch_size=128, verbose=0)
        self.assertIsInstance(history.history, dict)
        self.assertGreater(len(history.history), 0)

    def test_model_predict(self):
        model = foras.DeepCNN(input_shape=self.input_shape, blocks=2, units=64, classes=self.classes)
        model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val), epochs=10, batch_size=128, verbose=0)
        y_pred = model.predict(self.X_val)
        self.assertEqual(y_pred.shape[1], self.classes)

    def test_classification_report(self):
        model = foras.DeepCNN(input_shape=self.input_shape, blocks=2, units=128, classes=self.classes)
        model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val), epochs=10, batch_size=128, verbose=0)
        model.print_classification_report(self.X_val, self.y_val)

    def test_plot(self):
        model = foras.DeepCNN(input_shape=self.input_shape, blocks=3, units=64, classes=self.classes)
        model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val), epochs=10, batch_size=128, verbose=0)
        model.draw_plot(plot_type="accuracy")

    def test_save_load_model(self):
        self.assertIsNone(self.model.save_model(self.temp_model_path))

    #def test_load_model(self):
        loaded_model = foras.DeepCNN.load_model(self.temp_model_path)
        self.assertIsInstance(loaded_model, foras.DeepCNN)
        
    def tearDown(self):
        self.temp_dir.cleanup()
        cache_dir = os.path.join(os.getcwd(), '__pycache__')
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)


def get_hv_data(param="clf"):
    if param == "re":
        X_train, _ = make_classification(n_samples=800, n_features=15, random_state=42)
        X_val, _ = make_classification(n_samples=200, n_features=15, random_state=42)
        return X_train, X_val
    elif param == "clf":
        X, y = make_classification(n_samples=1000, n_features=15, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
        return X_train, X_val, y_train, y_val


class HybridVariationTestCase(unittest.TestCase):
    def __init__(self, methodName: str = 'runTest'):
        super().__init__(methodName=methodName)
        self.current_directory = os.path.dirname(os.path.realpath(__file__))
        self.temp_dir = tempfile.TemporaryDirectory(dir=self.current_directory)
        self.temp_model_path = os.path.join(self.temp_dir.name, "model.h5")
        X_train, X_val, y_train, y_val = get_hv_data("clf")
        X_train, X_val = X_train.reshape(X_train.shape[0], X_train.shape[1], 1), X_val.reshape(X_val.shape[0],                                                                                 X_val.shape[1], 1)
        self.input_shape = X_train.shape[1:]
        self.classes = len(set(y_train))
        self.X_train, self.X_val, self.y_train, self.y_val = X_train, X_val, y_train, y_val
        self.model = foras.hybrid_variation(input_shape=self.input_shape)

    def test_build_model(self):
        self.assertIsNotNone(
            self.model.build_model(input_shape=(100, 1), units=64, classes=2, block="Xception", loop_number=1))
        self.assertIsNotNone(
            self.model.build_model(input_shape=(100, 1), units=128, classes=4, block="Xception", loop_number=3))
        self.assertIsNotNone(
            self.model.build_model(input_shape=(50, 1), units=64, classes=2, block="residual", loop_number=2))
        self.assertIsNotNone(
            self.model.build_model(input_shape=(50, 1), units=128, classes=4, block="residual", loop_number=4))

    def test_fit(self):
        history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val), epochs=5,
                                 batch_size=32, verbose=0)
        self.assertIsNotNone(history)
        assert len(history.history['loss']) == 5

    def test_test(self):
        X_test, y_test = make_classification(n_samples=100, n_features=15, random_state=42)
        loss = self.model.test(X_test, y_test)
        self.assertIsNotNone(loss)
        print(loss)
        assert len(loss) == 5  # should have 5 items

    def test_predict(self):
        X_test, y_test = make_classification(n_samples=100, n_features=15, random_state=42)
        model = foras.hybrid_variation(input_shape=self.input_shape, units=64, classes=2)
        model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val), epochs=5, batch_size=32,
                  verbose=0)
        model.print_classification_report(X_test, y_test)

    def test_print_classification_report(self):
        X_test, y_test = make_classification(n_samples=100, n_features=15, random_state=42)
        model = foras.hybrid_variation(input_shape=self.input_shape, units=64, classes=2)
        model.fit(self.X_train, self.y_train, validation_data=(self.X_val,self.y_val), epochs=5, batch_size=32, verbose=0)
        model.print_classification_report(X_test, y_test)

    def test_draw_plot(self):
        model = foras.hybrid_variation(input_shape=self.input_shape, units=64, classes=2)
        model.fit(self.X_train, self.y_train, validation_data=(self.X_val,self.y_val), epochs=5, batch_size=32, verbose=0)
        self.assertIsNone(model.draw_plot("accuracy"))
        self.assertIsNone(model.draw_plot("loss"))
        self.assertIsNone(model.draw_plot("auc"))
        self.assertIsNone(model.draw_plot("invalid"))

    def test_save_and_load_model(self):
        self.assertIsNone(self.model.save_model(self.temp_model_path))
    #def test_load_model(self):
        loaded_model = foras.hybrid_variation.load_model(self.temp_model_path)
        self.assertIsInstance(loaded_model, foras.hybrid_variation)
    
    def tearDown(self):
        self.temp_dir.cleanup()
        cache_dir = os.path.join(os.getcwd(), '__pycache__')
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)

def get_cnnh_data(param="clf"):
    if param == "re":
        X_train, _ = make_classification(n_samples=800, n_features=15, random_state=42)
        X_val, _ = make_classification(n_samples=200, n_features=15, random_state=42)
        return X_train,X_val
    elif param == "clf":
        X, y = make_classification(n_samples=1000, n_features=15, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
        return X_train, X_val, y_train, y_val


class TestCNNGRU(unittest.TestCase):
    def __init__(self, methodName: str = 'runTest'):
        super().__init__(methodName=methodName)
        self.current_directory = os.path.dirname(os.path.realpath(__file__))
        self.temp_dir = tempfile.TemporaryDirectory(dir=self.current_directory)
        self.temp_model_path = os.path.join(self.temp_dir.name, "model.h5")
        X_train, X_val, y_train, y_val = get_cnnh_data("clf")
        X_train, X_val = X_train.reshape(X_train.shape[0], X_train.shape[1], 1), X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        self.input_shape = X_train.shape[1:]
        self.classes = len(set(y_train))
        self.X_train, self.X_val, self.y_train, self.y_val = X_train, X_val, y_train, y_val
        self.model = foras.Hybrid_CNN_GRU(input_shape=self.input_shape, units=64, classes=2)

    def test_build_model(self):
        assert self.model.build_model((100, 1), 64, 2).input_shape == (None, 100, 1)
        assert self.model.build_model((200, 1), 128, 3).input_shape == (None, 200, 1)
        assert self.model.build_model((300, 1), 256, 4).input_shape == (None, 300, 1)

    def test_fit(self):
        history = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_val,self.y_val), epochs=5, batch_size=32, verbose=0)
        assert len(history.history['loss']) == 5

    def test_test(self):
        X_test, y_test = make_classification(n_samples=100, n_features=15, random_state=42)
        loss = self.model.test(X_test, y_test)
        self.assertIsNotNone(loss)
        print(loss)
        assert len(loss) == 5 # should have 5 items

    def test_print_classification_report(self):
        X_test, y_test = make_classification(n_samples=100, n_features=15, random_state=42)
        model = foras.Hybrid_CNN_GRU(input_shape=self.input_shape, units=64, classes=2)
        model.fit(self.X_train, self.y_train, validation_data=(self.X_val,self.y_val), epochs=5, batch_size=32, verbose=0)
        model.print_classification_report(X_test, y_test)

    def test_draw_plot(self):
        model = foras.Hybrid_CNN_GRU(input_shape=self.input_shape, units=64, classes=2)
        model.fit(self.X_train, self.y_train, validation_data=(self.X_val,self.y_val), epochs=5, batch_size=32, verbose=0)
        model.draw_plot("accuracy")
        model.draw_plot("loss")
        model.draw_plot("auc")
        model.draw_plot("invalid")  # Should print "Invalid plot_type. Choose 'accuracy', 'loss', or 'auc'."

    def test_save_load_model(self):
        self.model.save_model("model_save_test.h5")

    #def test_load_model(self):
        loaded_model = foras.Hybrid_CNN_GRU.load_model(filepath="model_save_test.h5")
        assert isinstance(loaded_model, foras.Hybrid_CNN_GRU)
        print("==========test loading model with false info=========")
        loaded_model = foras.Hybrid_CNN_GRU.load_model(filepath="model_save_test.h5",units=128,classes=5)
        assert isinstance(loaded_model, foras.Hybrid_CNN_GRU)

    def tearDown(self):
        self.temp_dir.cleanup()
        cache_dir = os.path.join(os.getcwd(), '__pycache__')
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)

def get_ae_data(param="clf"):
    if param == "re":
        X_train, _ = make_classification(n_samples=800, n_features=20, random_state=42)
        X_val, _ = make_classification(n_samples=200, n_features=20, random_state=42)
        return X_train,X_val
    elif param == "clf":
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
        return X_train, X_val, y_train, y_val


class TestAutoEncoder(unittest.TestCase):

    def setUp(self):
        self.current_directory = os.path.dirname(os.path.realpath(__file__))
        self.temp_dir = tempfile.TemporaryDirectory(dir=self.current_directory)
        self.temp_model_path = os.path.join(self.temp_dir.name, "model.h5")
        X_train, X_val, y_train, y_val = get_ae_data("clf")
        #X_train, X_val = X_train.reshape(X_train.shape[0], X_train.shape[1], 1), X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        self.input_shape = (X_train.shape[1],1)
        self.classes = len(set(y_train))
        self.autoencoder_clf = foras.AutoEncoder(input_shape=self.input_shape, model_type='cnn', classifier=True, num_categories=2)
        self.X_train, self.X_val, self.y_train, self.y_val = X_train, X_val, y_train, y_val
        #set up data for reconstruction
        X_train_, X_val_ = get_ae_data("re")
        self.autoencoder = foras.AutoEncoder(input_shape=X_train_.shape[1:])
        self.X_train_, self.X_val_ = X_train_, X_val_

    def test_define_model(self):
        # Test that the define_model() method returns a valid autoencoder model
        self.assertIsNotNone(self.autoencoder.model)
        self.assertIsNotNone(self.autoencoder_clf.model)

    def test_fit(self):
        epochs = 5
        # the recnstruction mode of AE
        self.autoencoder.fit(self.X_train_, self.X_train_, validation_data = (self.X_val_, self.X_val_), epochs=epochs)
        # the classification mode of AE
        self.autoencoder_clf.fit(self.X_train, self.y_train, validation_data = (self.X_val, self.y_val), epochs=epochs)

    def test_test(self):
        X_test, y_test = make_classification(n_samples=100, n_features=20, random_state=42)

        re = self.autoencoder
        re.fit(self.X_train_, self.X_train_, validation_data = (self.X_val_, self.X_val_), epochs=2)
        print(re.history.history)
        loss = re.test(X_test,X_test)
        self.assertIsNotNone(loss)
        print("reconstruction loss exits.")

        clf = self.autoencoder_clf
        clf.fit(self.X_train, self.y_train, validation_data = (self.X_val, self.y_val), epochs=2)

        loss = clf.test(X_test,y_test)
        self.assertIsNotNone(loss)
        print("reconstruction loss exits.")

    def test_predict(self):
        # Testing the predict() method with dummy data
        X_test, y_test = make_classification(n_samples=100, n_features=20, random_state=42)
        re = self.autoencoder
        re.fit(self.X_train_, self.X_train_, validation_data = (self.X_val_, self.X_val_), epochs=2)
        decoded_X = re.predict(X_test)
        self.assertIsNotNone(decoded_X)
        self.assertEqual(decoded_X.shape, X_test.shape)

        clf = self.autoencoder_clf
        clf.fit(self.X_train, self.y_train, validation_data = (self.X_val, self.y_val), epochs=2)
        pred_y = clf.predict(X_test)
        self.assertIsNotNone(pred_y)
        
    def test_draw_plot(self):
        re = self.autoencoder
        re.fit(self.X_train_, self.X_train_, validation_data = (self.X_val_, self.X_val_), epochs=2)
        re.draw_mse_plot(plot_type="loss")

        clf = self.autoencoder_clf
        clf.fit(self.X_train, self.y_train, validation_data = (self.X_val, self.y_val), epochs=2)
        clf.draw_clf_plot(plot_type="auc")
        
    def tearDown(self):
        self.temp_dir.cleanup()
        cache_dir = os.path.join(os.getcwd(), '__pycache__')
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)


if __name__ == '__main__':
    unittest.main()
