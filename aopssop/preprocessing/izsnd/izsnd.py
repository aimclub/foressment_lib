import pandas as pd

from numbers import Number
from math import sqrt
from collections import defaultdict


class Predicate:
    """ 
    Class for predicates representation

    :param column: Name of the column
    :type column: str

    :param values: List of true values of the predicate
    :type values: array

    :param class_label: True class label for the predicate
    :type class_label: str

    :param metric: Metric that was calculalated to estimate predicst's quality
    :type metric: str

    :param metric_value: Value of the metric
    :type metric_value: float
    """

    def __init__(self, column, values, class_label):
        self.column = column
        self.values = values
        self.class_label = class_label
        self.metric = None
        self.metric_value = None

    def __str__(self):
        return f'Predicate({self.column}, {self.values}, {self.class_label}, {self.metric_value})'

    def __repr__(self):
        return f'Predicate({self.column}, {self.values}, {self.class_label}, {self.metric_value})'

    def __check_values(self, value):
        """
        Checks if predicate is true for the values

        :param values: Values to check
        :type values: list

        :return: Truthfulness of the predicate.
        """
        if isinstance(value, Number):
            return 1 if any([value in interval for interval in self.values]) else 0
        else:
            return 1 if value in self.values else 0

    def is_true(self, data):
        """
        Checks if predicate is true for the value

        :param value: Value to check

        :return: Truthfulness of the predicate
        """
        return data[self.column].apply(self.__check_values)

    def get_rule(self):
        """
        Return rule represented by the predicate in string format

        :return: Rule represented by the predicate in string format
        :rtype: str
        """
        return f'if {self.column} in {self.values} then {self.class_label} ({self.metric}={round(self.metric_value, 3)})'


class IzdapAlgo:
    """ 
    Class with IZDAP algo implementation 

    :param threshold: Treshold for values separation for predicates building
    :type threshold: float

    :param __N: Length of the dataset
    :type __N: int

    :param __rule_metric: Metric that should be used to evaluate predicates
    :type __rule_metric: str

    :param class_column: Name of column with class labels
    :type class_column: str

    :param data_stats: Dictionary with all dataset's values frequncies
    :type data_stats: dict

    :param class_stats: Dictionary with class labels frequncies
    :type class_stats: dict

    :param string_columns: String columns of the dataset
    :type string_columns: list

    :param number_columns: Numeric columns of the dataset
    :type number_columns: list

    :param aggregates: Aggregates built for the dataset
    :type aggregates: list

    :param predicates: Predicates built for the dataset
    :type predicates: list
    """

    def __init__(self, probability_threshold=0.1, verbose=0):
        self.__threshold = probability_threshold
        self.__verbose = verbose

    def __default_dict_to_regular(self, d):
        """
        Transforms defaultdict to dict ()

        :param d: defaultdict to transform

        :return: transfromed dict
        :rtype: dict
        """

        if isinstance(d, defaultdict):
            d = {k: self.__default_dict_to_regular(v) for k, v in d.items()}
        return d

    def calculate_regression_coefficient(self, nA, nB, nAB, N):
        """ 
        Calculates regression coefficient (pAB - pA*pB) / (pA * (1 - pA))

        :param nA: Frequency of event A - left part of the rule
        :type nA: int

        :param nB: Frequency of event B - right part of the rule
        :type nB: int

        :param nAB: Frequency of event A anв B together
        :type nAB: int

        :param N: Number of instances in the dataset
        :type N: int

        :return: value of the regression coefficient
        :rtype: float
        """

        pA = float(nA) / N
        pB = float(nB) / N
        pAB = float(nAB) / N
        return 0 if pA == 0 or pA == 1 else (pAB - pA * pB) / (pA * (1 - pA))

    def calculate_klosgen_measure(self, nA, nB, nAB, N):
        """ 
        Calculates Klosgen meausure sqrt(pB) * (pB|A-pB))

        :param nA: Frequency of event A - left part of the rule
        :type nA: int

        :param nB: Frequency of event B - right part of the rule
        :type nB: int

        :param nAB: Frequency of event A anв B together
        :type nAB: int

        :param N: Number of instanec in the dataset
        :type N: int

        :return: Value of the Klosgen meausure
        :rtype: float
        """

        pA = float(nA) / N
        pB = float(nB) / N
        pAB = float(nAB) / N
        pB_A = pAB / pA
        return sqrt(pB) * (pB_A - pB)

    def fit(self, data, class_column, positive_class_label=1, rule_metric='regression_coef'):
        """ 
        Builds predicates and rules

        :param data: Data to build predicates and rules for
        :type data: pandas.DataFrame

        :param class_column: Name of column with class labels
        :type class_column: str

        :param positive_class_label: Label for positive class
        :type positive_class_label: str

        :param rule_metric: Metric that should be used to evaluate predicates
        :type rule_metric: str
        """

        self.__N = len(data)
        self.class_column = class_column
        self.__count_statistics(data)
        self.__build_aggregates_and_predicates()
        self.__evaluate_predicates(rule_metric)

    def __evaluate_predicates(self, rule_metric_label):
        """ 
        Evaluates rules represented by predicates using metric specified by rule_metric_label

        :param rule_metric_label: Metric that should be used to evaluate predicates
        :type rule_metric_label: str
        """

        if rule_metric_label == 'regression_coef':
            self.__rule_metric = self.calculate_regression_coefficient

        if rule_metric_label == 'klosen':
            self.__rule_metric = self.klosgen_measure

        for predicate in self.predicates:

            predicate.metric = rule_metric_label
            nB = self.class_stats[predicate.class_label]
            nA = 0
            nAB = 0

            for value in predicate.values:
                nA += sum(self.data_stats[predicate.column][value].values())

                if predicate.class_label in self.data_stats[predicate.column][value]:
                    nAB += self.data_stats[predicate.column][value][predicate.class_label]

            predicate.metric_value = self.__rule_metric(nA, nB, nAB, self.__N)

        self.predicates.sort(key=lambda x: x.metric_value, reverse=True)

    def __count_statistics(self, data, bins=10):
        """ 
        Calculates data statistics

        :param data: Data that is used to calculate statistics
        :type data: pandas.DataFrame

        :param bind: Number of discretization bins used to build predicates for numeric attributes
        :type bind: int
        """

        self.data_stats = {}

        self.number_columns = list(data.select_dtypes(include=['int64', 'float64']).columns)
        self.string_columns = list(data.select_dtypes(include=['object']).columns)

        if self.class_column in self.string_columns:
            self.string_columns.remove(self.class_column)

        if self.class_column in self.number_columns:
            self.number_columns.remove(self.class_column)

        for col in self.string_columns:
            d = dict(data[[col, self.class_column]].value_counts())

            tree = lambda: defaultdict(tree)
            new_d = tree()

            for (k1, k2), val in d.items():
                new_d[k1][k2] = val

            self.data_stats[col] = self.__default_dict_to_regular(new_d)

        for col in self.number_columns:
            categories = pd.cut(data[col], bins)

            d = dict(pd.concat([pd.cut(data[col], bins), data[[self.class_column]]], axis=1).value_counts())

            tree = lambda: defaultdict(tree)
            new_d = tree()

            for (k1, k2), val in d.items():
                new_d[k1][k2] = val

            self.data_stats[col] = self.__default_dict_to_regular(new_d)

        self.class_stats = dict(data[self.class_column].value_counts())

    def __build_aggregates_and_predicates(self):
        """ 
        Builds aggregates and predicates 
        """

        self.aggregates = {k: [] for k in self.data_stats.keys()}
        self.predicates = []

        for column, value_stats in self.data_stats.items():

            self.aggregates[column] = {}

            for value, value_class_stats in value_stats.items():
                for class_label in [*self.class_stats]:

                    class_count = 0

                    if class_label in value_class_stats:
                        class_count = value_class_stats[class_label]

                    probability = class_count / sum(value_class_stats.values())

                    if 2 * probability - 1 > self.__threshold:
                        if class_label not in self.aggregates[column]:
                            self.aggregates[column][class_label] = []

                        self.aggregates[column][class_label].append(value)

            self.predicates.extend([Predicate(column,
                                              self.aggregates[column][class_label],
                                              class_label) for class_label in self.aggregates[column].keys()])

    def transform(self, data, rule_fraction=.8):
        """ 
        Transfroms data using top rule_fraction*100% of predicates

        :param rule_fraction: Fraction of rules that will be used to form new dataset
        :type rule_fraction: float

        :param data: Data to transform
        :type data: pandas.DataFrame

        :return: Transformed binary dataset
        :rtype: pandas.DataFrame
        """

        rule_set = self.predicates[:int(rule_fraction * len(self.predicates))]
        new_columns = []

        result = pd.DataFrame()

        for predicate in rule_set:
            new_columns.append(predicate.column + '=' + str(predicate.values))
            result = pd.concat([result, predicate.is_true(data)], axis=1)

        result = pd.concat([result, data[self.class_column]], axis=1)

        new_columns.append(self.class_column)

        result.columns = new_columns
        result.set_index(data.index)

        return result

    def get_rules(self):
        """ 
        Outputs rules built in string format

        :return: List of Rules
        :rtype: array
        """

        rules = [predicate.get_rule() for predicate in self.predicates]

        return rules
