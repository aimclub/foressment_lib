#!/usr/bin/python
# -*- coding: utf-8 -*-

import gzip
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from keras import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import pickle
import os.path


class _NNClassifier:
    """
    Class for the description of deep neural network

    :param model: model of the deep neural network
    :type: keras.models.Sequential

    :param out_size: number of neurons in the outer layer
    :type: int

    :param plot: boolean parameter that defines presentation of the neural network plot (default = True)
    :type: bool

    :param num_epochs: number of training epochs (default = 10)
    :type: int
    """

    def __init__(self, in_size, out_size, plot=True, num_epochs=10):
        """
        Initialization of the deep neural network model

        :param in_size: number of neurons in the inner layer (int).
        :param out_size: number of neurons in the outer layer (int).
        :param plot: boolean parameter that defines presentation of the neural network plot (bool). Default = True.
        :param num_epochs: number of training epochs (int). Default = 10.
        """
        self.model = Sequential()
        self.model.add(Dense(30, activation='relu', input_shape=(in_size,)))
        self.model.add(Dense(15, activation='relu'))
        self.model.add(Dense(7, activation='relu'))
        self.model.add(Dense(out_size, activation='relu'))
        self.out_size = out_size
        self.plot = plot
        self.num_epochs = num_epochs
        print(self.model.summary())

    def fit(self, x_train, y_train):
        """
        Training of the deep neural network model

        :param x_train: training data (features) (numpy.ndarray).
        :param y_train: training dat (labels) (numpy.ndarray).

        :return: trained model of the deep neural network.
        """
        self.model.compile(optimizer='adam', loss='mse', metrics=['accuracy', 'binary_accuracy'])
        self.history = self.model.fit(x_train, y_train, epochs=self.num_epochs, batch_size=32, verbose=1,
                                      validation_split=0.2)
        if self.plot:
            self.draw_plot()
        return self

    def test(self, x_test, y_test):
        """
        Testing of the deep neural network model

        :param x_test: testing data (features) (numpy.ndarray).
        :param y_test: testing data (labels) (numpy.ndarray).
        """
        loss, acc = self.model.evaluate(x_test, y_test, verbose=1)
        print('Test accuracy: %.3f' % acc)

    def predict(self, x):
        """
        Identification of class objects with the help of deep neural network model

        :param x: objects (numpy.ndarray).

        :return: class labels (numpy.ndarray).
        """
        predicted_labels = None
        if self.out_size > 1:
            predicted_labels = self.model.predict(x)
            for i in range(len(predicted_labels)):
                ind = np.argmax(predicted_labels[i])
                predicted_labels[i] = np.zeros(len(predicted_labels[i]))
                predicted_labels[i][ind] = 1
        else:
            predicted_labels = np.array([1 if e > 0 else 0 for e in self.model.predict(x)])
        return predicted_labels

    def draw_plot(self):
        """
        Output of the plot of the deep neural network training process
        """
        print(self.history.history.keys())
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        # plt.title(u'Точность модели')
        # plt.ylabel(u'точность')
        # plt.xlabel(u'номер эпохи')
        # plt.legend([u'обучающая выборка', u'тестовая выборка'], loc='lower right')
        plt.title(u'Accuracy of the model')
        plt.ylabel(u'Accuracy')
        plt.xlabel(u'Number of epoch')
        plt.legend([u'Training data', u'Testing data'], loc='lower right')
        plt.show()


class SAIClassifier:
    # """
    # Класс для описания параметров классификатора
    #
    # :param cls_type: тип классификатора (str).
    # :param classifier: Объект классиификатора (keras.models.Sequential и sklearn.*).
    # """
    """
    Class for the description of the classifier parameters

    :param cls_type: type of the classifier (str).
    :param classifier: object of the classifier (keras.models.Sequential and sklearn.*).
    """

    def __init__(self, cls_type='neural_network', in_size=None, out_size=None, plot=False, num_epochs=10):
        # """
        # Инициализация модели классификатора
        #
        # :param cls_type: тип классификатора (str). Default = 'neural_network'.
        # :param in_size: Количество нейронов на распределительном слое (int). Default = None.
        # :param out_size: Количество нейронов на выходном слое (int). Default = None.
        # :param plot: Булев параметр, задающий вывод графика обучения нейронной сети (bool). Default = False.
        # :param num_epochs: Количество эпох обучения (int). Default = 10.
        # """
        """
        Initialization of the classifier model

        :param cls_type: type of the classifier (str). Default = 'neural_network'.
        :param in_size: number of neurons in inner layer (int). Default = None.
        :param out_size: number of neuron in outer layer (int). Default = None.
        :param plot: boolean parameter that defines the presentation of the neural network plot (bool). Default = False.
        :param num_epochs: number of training epochs (int). Default = 10.
        """
        self.cls_type = cls_type
        self.classifier = None
        if cls_type == 'decision_tree':
            self.classifier = DTC()
        elif cls_type == 'naive_bayes':
            self.classifier = GNB()
        elif cls_type == 'logistic_regression':
            self.classifier = make_pipeline(StandardScaler(), LR())
        elif cls_type == 'neural_network':
            self.classifier = make_pipeline(StandardScaler(), _NNClassifier(in_size, out_size, plot, num_epochs))
        else:
            assert (False)

    def fit(self, x_train, y_train):
        # """
        # Обучение модели классификатора
        #
        # :param x_train: Обучающая выборка (признаки) (numpy.ndarray).
        # :param y_train: Обучающая выборка (метки классов) (numpy.ndarray).
        # :param plot: Булев параметр, задающий вывод графика обучения нейронной сети (bool). Default = True.
        # :param num_epochs: Количество эпох обучения (int). Default = 10.
        # :return: модель обученного классификатора.
        # """
        """
        Training of the classifier model

        :param x_train: training data (features) (numpy.ndarray).
        :param y_train: training data (labels) (numpy.ndarray).
        :param plot: boolean parameter that defines the presentation of the neural network plot (bool). Default = True.
        :param num_epochs: number of training epochs (int). Default = 10.
        :return: model of the trained classifier.
        """
        cls = self.classifier.fit(x_train, y_train)
        return cls

    def predict(self, x):
        # """
        # Определение класса объектов при помощи модели классификатора
        #
        # :param x: Объекты (numpy.ndarray).
        #
        # :return: Метки классов (numpy.ndarray).
        # """
        """
        Definition of classes of objects based on the classifier model

        :param x: objects (numpy.ndarray).

        :return: class labels (numpy.ndarray).
        """
        y = self.classifier.predict(x)
        return y

    def save(self, saved_file):
        # """
        # Сериализация модели классификатора
        #
        # :param saved_file: Наименование файла (str).
        # """
        """
        Serialization of the classifier model

        :param saved_file: name of the file to save (str).
        """
        if self.cls_type == 'neural_network':
            self.classifier.steps[1][1].model.save_weights(saved_file)  # [1][1] - the second classifier in pipeline
        else:
            with open(saved_file, 'wb') as f:
                pickle.dump(self.classifier, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, loaded_file):
        # """
        # Десериализация модели классификатора
        #
        # :param loaded_file: Наименование файла (str).
        # """
        """
        Deserialization of the classifier model

        :param loaded_file: name of the file to load (str).
        """
        if self.cls_type == 'neural_network':
            self.classifier.steps[1][1].model.load_weights(loaded_file)  # [1][1] - the second classifier in pipeline
        else:
            with open(loaded_file, 'rb') as f:
                self.classifier = pickle.load(f)


class FormatDetector:
    """
    Class for the identification of the format of the processed data

    :param n: number of fields in data (int).
    :param d: delimeter of fields in data (str).
    """

    def __init__(self, file):
        """
        Initialization of the object of FormatDetector class

        :param file: name of the file that contains data to process (str).
        """
        n, d = None, None
        if not os.path.isfile(file):
            raise Exception('File does not exist')
        with gzip.open(file, 'rb') if file.endswith('.gz') else open(file, 'r') as fh:
            try:
                line = fh.readline()
                if ';' in line:
                    d = ';'
                elif ',' in line:
                    d = ','
                n = len(fh.readline().split(d))
            except:
                line = fh.readline().decode('utf-8')
                if ';' in line:
                    d = ';'
                elif ',' in line:
                    d = ','
                n = len(fh.readline().decode('utf-8').split(d))
        self.n = n
        self.d = d


class DataLoader:
    """
    Class to load the data to process

    :param n: number of fields in data (int).
    :param d: delimiter of fields in data (str).
    :param features: features in data (numpy.ndarray).
    :param labels: labels in data (one-hot encoding) (numpy.ndarray).
    :param num_labels: labels in data (numpy.ndarray).
    """

    def __init__(self, file, n, d):
        """
        Initialization of the object of DataLoader class

        :param file: name of the file with data to process (str).
        :param n: number of fields in data (int).
        :param d: delimiter between fields in data (str).
        """
        self.n = n
        self.d = d
        self.features = None
        self.labels = None  # for multiclass task: one-hot encoding
        self.num_labels = None
        self.load(file)

    def load(self, file):  # override this method in derived class
        """
        Load of the data to process

        :param file: name of the file to load (str).
        """
        raise NotImplementedError()


class ClsEstimator:
    """
    Class for the assessment of the classifier effectiveness

    :param features: features in data (numpy.ndarray).
    :param labels: labels in data (one-hot encoding) (numpy.ndarray).
    :param num_labels: labels in data (numpy.ndarray).
    :param  classifiers: classifiers to check (list).
    """

    def __init__(self, features, labels, num_labels, classifiers):
        """
        Initialization of the object of the ClsEstimator class

        :param features: features in data (numpy.ndarray).
        :param labels: labels in data (one-hot encoding) (numpy.ndarray).
        :param num_labels: labels in data (numpy.ndarray).
        :param classifiers: classifiers to check (list).
        """
        self.features = features
        self.labels = labels
        self.num_labels = num_labels
        self.classifiers = classifiers

    def estimate(self, print_metrics=True):
        """
        Assessment of the classifiers effectiveness

        :param print_metric: boolean parameter that defines the presentation of the parameters of classifiers effectiveness (bool).

        :return: parameters of the classifiers effectiveness (dict).
        """
        train_features, test_features, train_labels_inds, test_labels_inds = train_test_split(self.features,
                                                                                              range(len(self.labels)),
                                                                                              test_size=0.2,
                                                                                              random_state=42)
        res = {}
        for cls in self.classifiers:
            res[cls.cls_type] = {}
            train_labels, test_labels = None, None
            cls_name = cls.cls_type
            if cls_name == 'neural_network':
                train_labels = np.array([self.labels[i] for i in train_labels_inds])
                test_labels = np.array([self.labels[i] for i in test_labels_inds])
            else:
                train_labels = np.array([self.num_labels[i] for i in train_labels_inds])
                test_labels = np.array([self.num_labels[i] for i in test_labels_inds])
            cls.fit(train_features, train_labels)
            for (f, l, n) in [(train_features, train_labels, 'training'), (test_features, test_labels, 'testing')]:
                predicted_labels = cls.predict(f)
                (precision, recall, fscore, _) = precision_recall_fscore_support(l, predicted_labels)
                accuracy = accuracy_score(l, predicted_labels)
                if print_metrics:
                    for (metrics, metrics_name) in [(precision, 'precision'), (recall, 'recall'), (fscore, 'fscore'),
                                                    (accuracy, 'accuracy')]:
                        print('{} of {} on {} sample ({} instances): {}'.format(metrics_name, cls_name, n, len(l),
                                                                                metrics))
                res[cls.cls_type][n + '_sample'] = {'size': len(l), 'precision': precision, 'recall': recall,
                                                    'fscore': fscore, 'accuracy': accuracy}
        return res
