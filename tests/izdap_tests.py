import unittest

from aopssop import IzdapAlgo
# from aopssop import Predicate
from sklearn.datasets import make_classification
import pandas as pd
import inspect


class IzdapTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(IzdapTest, self).__init__(*args, **kwargs)
        IzdapTest.n = 1

    def setUp(self):
        print('Test {} BEGIN'.format(IzdapTest.n))

    def tearDown(self):
        print('Test {} END'.format(IzdapTest.n))
        IzdapTest.n += 1

    def test_regression_coefficient(self):
        print(inspect.stack()[0][3])
        message = 'The regression coefficient is calculated incorrectly'
        self.assertEqual(round(IzdapAlgo().calculate_regression_coefficient(nA=200, nB=200, nAB=100, N=1000), 3), 0.375,
                         message)

    def test_klosgen_measure(self):
        print(inspect.stack()[0][3])
        message = 'The klosgen measure is calculated incorrectly'
        self.assertEqual(round(IzdapAlgo().calculate_klosgen_measure(nA=200, nB=200, nAB=100, N=1000), 3), 0.134,
                         message)

    def test_string_column(self):
        print(inspect.stack()[0][3])
        message = 'String columns were identified incorrectly'

        data = make_classification(n_samples=200, n_features=4,
                                   n_informative=2, n_classes=2,
                                   random_state=42)
        df = pd.DataFrame(data[0], columns=['col1', 'col2', 'col3', 'col4', ])
        df['class'] = pd.Series(data[1])
        algo = IzdapAlgo(0.2)
        algo.fit(df, class_column="class", positive_class_label='1')

        self.assertEqual(len(algo.string_columns), 0, message)

    def test_number_column(self):
        print(inspect.stack()[0][3])
        message = 'String columns were identified incorrectly'

        data = make_classification(n_samples=200, n_features=4,
                                   n_informative=2, n_classes=2,
                                   random_state=42)
        df = pd.DataFrame(data[0], columns=['col1', 'col2', 'col3', 'col4', ])
        df['class'] = pd.Series(data[1])
        algo = IzdapAlgo(0.2)
        algo.fit(df, class_column="class", positive_class_label='1')

        self.assertEqual(len(algo.number_columns), 4, message)

    def test_class_stats(self):
        print(inspect.stack()[0][3])
        message = 'Class stats was calculated incorrectly'

        data = make_classification(n_samples=200, n_features=4,
                                   n_informative=2, n_classes=2,
                                   random_state=42)
        df = pd.DataFrame(data[0], columns=['col1', 'col2', 'col3', 'col4', ])
        df['class'] = pd.Series(data[1])
        algo = IzdapAlgo(0.2)
        algo.fit(df, class_column="class", positive_class_label=1)

        self.assertEqual(algo.class_stats[1], df['class'].value_counts()[1], message)

    def test_data_stats(self):
        print(inspect.stack()[0][3])
        message = 'Data stats was calculated incorrectly'

        data = make_classification(n_samples=200, n_features=4,
                                   n_informative=2, n_classes=2,
                                   random_state=42)
        df = pd.DataFrame(data[0], columns=['col1', 'col2', 'col3', 'col4', ])
        df['class'] = pd.Series(data[1])
        algo = IzdapAlgo(0.2)
        algo.fit(df, class_column="class", positive_class_label=1)

        self.assertEqual(len(algo.data_stats), 4, message)

    def test_rules(self):
        print(inspect.stack()[0][3])
        message = 'Rules were extracted incorrectly incorrectly'

        data = make_classification(n_samples=200, n_features=4,
                                   n_informative=2, n_classes=2,
                                   random_state=42)
        df = pd.DataFrame(data[0], columns=['col1', 'col2', 'col3', 'col4', ])
        df['class'] = pd.Series(data[1])
        algo = IzdapAlgo(0.2)
        algo.fit(df, class_column="class", positive_class_label=1)
        rule = "if col1 in [Interval(1.032, 1.661, closed='right'), Interval(0.404, 1.032, closed='right'), Interval(-0.224, 0.404, closed='right'), Interval(1.661, 2.289, closed='right'), Interval(2.289, 2.917, closed='right')] then 1 (regression_coef=0.759)"
        self.assertEqual(algo.get_rules()[0], rule, message)

    def test_predicates(self):
        print(inspect.stack()[0][3])
        message = 'Predicates is not chacking data correctly'

        data = make_classification(n_samples=200, n_features=4,
                                   n_informative=2, n_classes=2,
                                   random_state=42)
        df = pd.DataFrame(data[0], columns=['col1', 'col2', 'col3', 'col4', ])
        df['class'] = pd.Series(data[1])
        algo = IzdapAlgo(0.2)
        algo.fit(df, class_column="class", positive_class_label=1)

        predicate = algo.predicates[0]
        self.assertEqual(predicate.is_true(df[:1]).iloc[0], 1, message)


if __name__ == '__main__':
    unittest.main()
