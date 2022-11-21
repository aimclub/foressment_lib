import pandas as pd
from math import sqrt

from numbers import Number
from collections import defaultdict

class Predicate:
    """ Class for predicates representation"""
    
    def __init__ (self, column, values, class_label):
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
        if isinstance(value, Number):
            return 1 if any([value in interval for interval in self.values]) else 0
        else:
            return 1 if value in self.values else 0
    
    
    def is_true (self, data):
        return data[self.column].apply(self.__check_values) 
    
    
    def get_rule (self):
        return f'if {self.column} in {self.values} then {self.class_label} ({self.metric}={round(self.metric_value, 3)})'  


class IzdapAlgo:
    """ Class with IZDAP algo implementation """
    
    def __init__(self, probability_threshold,  verbose=0):
        self.__threshold = probability_threshold
        self.__verbose = verbose
    
    
    def __default_dict_to_regular(self, d):
        
        if isinstance(d, defaultdict):
            d = {k: self.__default_dict_to_regular(v) for k, v in d.items()}
        return d
    
    
    def __calculate_regression_coefficient(self, nA, nB, nAB, N):
        """ Calculates regression coefficient (pAB - pA*pB) / (pA * (1 - pA)) """
        
        pA = float(nA) / N
        pB = float(nB) / N
        pAB = float(nAB) / N
        return 0 if pA == 0 or pA == 1 else (pAB - pA*pB) / (pA * (1 - pA))

    
    def __klosgen_measure(self, nA, nB, nAB, N) :
        """ Calculates Klosgen meausure sqrt(pB) * (pB|A- pB)) """
        
        pA = float(nA) / N
        pB = float(nB) / N
        pAB = float(nAB) / N
        pB_A = pAB / pA
        return sqrt(pB) * (pB_A - pB)
        
            
    def fit(self, data, class_column, positive_class_label = 1, rule_metric='regression_coef'):
        """ Builds predicates and rules """
        
        self.__N = len(data)
        self.__class_column = class_column
        self.__count_statistics(data)
        self.__build_aggregates_and_predicates()
        self.__evaluate_predicates(rule_metric)
    
    
    def __evaluate_predicates(self, rule_metric_label):
        """ Evaluates rules represented by predicates using metric specified by rule_metric_label """
        
        if rule_metric_label == 'regression_coef':
            self.__rule_metric = self.__calculate_regression_coefficient

        if rule_metric_label == 'klosen':
            self.__rule_metric = self.__klosgen_measure
            
        for predicate in self.predicates:
            
            predicate.metric = rule_metric_label
            nB = self.__class_stats[predicate.class_label]
            nA = 0
            nAB = 0
            
            for value in predicate.values:
                nA += sum(self.__data_stats[predicate.column][value].values())
                
                if predicate.class_label in self.__data_stats[predicate.column][value]:
                    nAB += self.__data_stats[predicate.column][value][predicate.class_label]
            
            predicate.metric_value = self.__rule_metric(nA, nB, nAB, self.__N)
        
        self.predicates.sort(key=lambda x: x.metric_value, reverse=True)
            
            
    def __count_statistics(self, data, bins=10):
        """ Calculates data statistics """
        
        self.__data_stats = {}
        
        self.__number_columns = list(data.select_dtypes(include=['int64','int64']).columns)
        self.__string_columns = list(data.select_dtypes(include=['object']).columns)

        if self.__class_column in self.__string_columns:
            self.__string_columns.remove(self.__class_column)
        
        if self.__class_column in self.__number_columns:
            self.__number_columns.remove(self.__class_column)

        for col in self.__string_columns:
            d = dict(data[[col, self.__class_column]].value_counts())

            tree = lambda: defaultdict(tree)  
            new_d = tree()

            for (k1, k2), val in d.items():
                new_d[k1][k2] = val

            self.__data_stats[col] = self.__default_dict_to_regular(new_d)
    
        for col in self.__number_columns:
            categories = pd.cut(data[col], bins)

            d = dict(pd.concat([pd.cut(data[col], bins), data[[self.__class_column]]], axis=1).value_counts())

            tree = lambda: defaultdict(tree)  
            new_d = tree()

            for (k1, k2), val in d.items():
                new_d[k1][k2] = val

            self.__data_stats[col] = self.__default_dict_to_regular(new_d)
            
        self.__class_stats = dict(data[self.__class_column].value_counts())
        
        
    def __build_aggregates_and_predicates(self):
        """ Builds aggregates and predicates """
        
        self.aggregates = {k : [] for k in self.__data_stats .keys()}
        self.predicates = []

        for column, value_stats in self.__data_stats .items():

            self.aggregates[column] = {}

            for value, value_class_stats in value_stats.items():   
                for class_label in [*self.__class_stats]:

                    class_count = 0

                    if class_label in value_class_stats:
                        class_count = value_class_stats[class_label]

                    probability = class_count / sum(value_class_stats.values())

                    if 2 * probability  - 1 > self.__threshold:
                        if class_label not in self.aggregates[column]:
                            self.aggregates[column][class_label] = []

                        self.aggregates[column][class_label].append(value)

            self.predicates.extend([Predicate(column, 
                                                self.aggregates[column][class_label], 
                                                class_label) for class_label in self.aggregates[column].keys()])

            
    def transform (self, data, rule_fraction=.8):
        """ Transfroms data using top rule_fraction*100% of predicates """
        
        rule_set = self.predicates[:int(rule_fraction * len(self.predicates))]
        new_columns = []
        
        result = pd.DataFrame()
        
        for predicate in rule_set:
            new_columns.append(predicate.column + '=' + str(predicate.values))
            result = pd.concat([result, predicate.is_true(data)], axis=1)
        
        result = pd.concat([result, data[self.__class_column]], axis=1)   
        
        new_columns.append(self.__class_column)
        
        result.columns = new_columns
        result.set_index(data.index)
        
        return result
    
    
    def get_rules(self):
        """ Outputs rules built in string format """
        
        rules = [predicate.get_rule() for predicate in self.predicates]
        
        return rules
    
        
if __name__ == '__main__':
    
    import sys
    
    if sys.argv:
        
        test_path = sys.argv[0]

        test_data = pd.read_csv(test_path)

        algo = IzdapAlgo(0.1)
        algo.fit(test_data, class_column = "class", positive_class_label = ' >50K')

        rules = algo.get_rules()
        print(rules[0])

        transformed_data = algo.transform(test_data)
        print(transformed_data.info())    