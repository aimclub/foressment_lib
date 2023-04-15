#!/usr/bin/python
# -*- coding: utf-8 -*-

from foressment_ai import SAIClassifier, FormatDetector, DataLoader, ClsEstimator
import unittest
import inspect
import os.path
import os
import numpy as np
import tempfile


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
        self.assertFalse(SAIClassifier('neural_network', 10, 1) is None,
                         'The object of class SAIClassifier could not be created')

    def test_sc_type_is_correct(self):
        print(inspect.stack()[0][3])
        classifier_types = ['decision_tree', 'naive_bayes', 'logistic_regression', 'neural_network']
        for c in classifier_types:
            classifier = SAIClassifier(c, 10, 1, plot=False)
            self.assertTrue(classifier.cls_type in classifier_types, 'The classifier type is not correct')

    def test_sc_fit(self):
        print(inspect.stack()[0][3])
        classifier_types = ['decision_tree', 'naive_bayes', 'logistic_regression', 'neural_network']
        in_size = np.shape(self.xor_ds['features'])[1]
        out_size = np.shape(self.xor_ds['labels'])[1]
        for c in classifier_types:
            classifier = SAIClassifier(c, in_size, out_size, plot=False)
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
            classifier = SAIClassifier(c, in_size, out_size, plot=False)
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
            classifier = SAIClassifier(c, in_size, out_size, plot=False)
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
            classifier = SAIClassifier(c, in_size, out_size, plot=False)
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
            self.assertFalse(FormatDetector(f) is None, 'The object of class FormatDetector could not be created')

    def test_fd_file_exists(self):
        tmp_file = tempfile.NamedTemporaryFile()
        try:
            FormatDetector(tmp_file.name)
        except:
            self.assertTrue(False, 'Error: file "{}" must exist'.format(tmp_file))

    def test_fd_delimiter_is_correct(self):
        print(inspect.stack()[0][3])
        f = '../hai/hai-20.07/test1.csv.gz'
        if os.path.isfile(f):
            fd = FormatDetector(f)
            self.assertTrue(fd.d in [';', ','], 'The delimiter is not correct')

    def test_dl_is_not_none(self):
        print(inspect.stack()[0][3])
        f = '../hai/hai-20.07/test1.csv.gz'

        class DataLoaderExample(DataLoader):
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
        classifiers = [SAIClassifier(c, in_size, out_size, plot=False) for c in classifier_types]
        for classifier in classifiers:
            self.assertFalse(ClsEstimator(self.xor_ds['features'], self.xor_ds['labels'], self.xor_ds['labels'],
                                          [classifier]) is None,
                             'The object of class ClsEstimator could not be created')

    def test_ce_estimate(self):
        print(inspect.stack()[0][3])
        classifier_types = ['decision_tree', 'naive_bayes', 'logistic_regression', 'neural_network']
        in_size = np.shape(self.xor_ds['features'])[1]
        out_size = np.shape(self.xor_ds['labels'])[1]
        classifiers = [SAIClassifier(c, in_size, out_size, plot=False) for c in classifier_types]
        for classifier in classifiers:
            classifier.fit(self.xor_ds['features'], self.xor_ds['labels'])
        try:
            ClsEstimator(self.xor_ds['features'], self.xor_ds['labels'], self.xor_ds['labels'], classifiers).estimate()
        except:
            self.assertTrue(False, 'Error while calling estimate')


if __name__ == '__main__':
    unittest.main()
