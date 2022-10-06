#!/usr/bin/python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import gzip
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from collections import OrderedDict
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

arg_parser = ArgumentParser()
arg_parser.add_argument('-f', '--file', type=str, required=True, help='set input file name')
arg_parser.add_argument('-d', '--dataset', type=str, required=True, help='set dataset name (hai|edge-iiotset|dataport)')
arg_parser.add_argument('-e', '--nepochs', type=int, required=False, help='set number of epochs for neural network')
args = arg_parser.parse_args()

file = args.file
dataset = args.dataset
nepochs = 10 # default value
if args.nepochs is not None:
    nepochs = args.nepochs

class NNClassifier:
    def __init__(self, in_size, out_size):
        self.model = Sequential()
        self.model.add(Dense(30, activation='relu', input_shape=(in_size,)))
        self.model.add(Dense(15, activation='relu'))
        self.model.add(Dense(7, activation='relu'))
        self.model.add(Dense(out_size, activation='relu'))
        self.out_size = out_size
        print(self.model.summary())

    def fit(self, x_train, y_train, plot=True):
#        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_accuracy'])
        self.model.compile(optimizer='adam', loss='mse', metrics=['accuracy', 'binary_accuracy'])
        self.history = self.model.fit(x_train, y_train, epochs=nepochs, batch_size=32, verbose=1, validation_split=0.2)
        if plot:
            self.draw_plot()

    def test(self, x_test, y_test):
        loss, acc = self.model.evaluate(x_test, y_test, verbose=1)
        print('Test accuracy: %.3f' % acc)

    def predict(self, x):
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
        print(self.history.history.keys())
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title(u'Точность модели')
        plt.ylabel(u'точность')
        plt.xlabel(u'номер эпохи')
        plt.legend([u'обучающая выборка', u'тестовая выборка'], loc='lower right')
        plt.show()

class FormatDetector:
    def __init__(self, file):
        n, d = None, None
        with gzip.open(file, 'rb') if file.endswith('.gz') else open(file, 'r') as fh:
            line = fh.readline()
            if ';' in line:
                d = ';'
            elif ',' in line:
                d = ','
            n = len(fh.readline().split(d))
        self.n = n
        self.d = d

class DataLoader:
    def __init__(self, file, n, d):
        features, labels = None, None
        if dataset == 'hai':
            if n in [64, 84]:
                features = np.genfromtxt(file, delimiter=d, dtype=float, skip_header=1, usecols=range(1,n-4))
                attacks = np.genfromtxt(file, delimiter=d, dtype=float, skip_header=1, usecols=range(n-4,n))
                labels = np.array([[0] if sum(a) == 0 else [1] for a in attacks])
            elif n == 88:
                features = np.genfromtxt(file, delimiter=d, dtype=float, skip_header=1, usecols=range(1,n-1))
                labels = np.genfromtxt(file, delimiter=d, dtype=float, skip_header=1, usecols=range(n-1,n))
            self.features = features
            self.labels = labels
            self.num_labels = labels
        elif dataset == 'edge-iiotset':
            df = pd.read_csv(file, low_memory=False, sep=d)
            drop_columns = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
                            "http.request.method", "http.file_data", "http.referer", "http.request.full_uri", "http.request.version", 
                            "icmp.transmit_timestamp", "http.request.uri.query", "tcp.options", "tcp.payload", "tcp.srcport",
                            "tcp.dstport", "udp.port", "dns.qry.name", "dns.qry.name.len", "dns.qry.qu",
                            "mqtt.conack.flags", "mqtt.msg", "mqtt.protoname", "mqtt.topic", "Attack_label"]
            df.drop(drop_columns, axis=1, inplace=True)
            df.dropna(axis=0, how='any', inplace=True)
            df.drop_duplicates(subset=None, keep="first", inplace=True)
            df = shuffle(df)
            str_labels = df.iloc[:,-1].tolist()
            self.features = np.array(df.iloc[:,:-1].values.tolist())
            unique_labels = list(OrderedDict.fromkeys(str_labels))
            print('Number of classes: ' + str(len(unique_labels)))
            labels = list(map(lambda x: unique_labels.index(x), str_labels))
            self.labels = np.array([np.zeros(len(unique_labels))] * len(labels))
            for i in range(len(labels)): # one-hot encoding
                self.labels[i][labels[i]] = 1
            self.num_labels = labels
        elif dataset == 'dataport':
            self.features = np.genfromtxt(file, delimiter=d, dtype=float, skip_header=1, usecols=range(1,n-1))
            labels = np.genfromtxt(file, delimiter=d, dtype=float, skip_header=1, usecols=range(n-1,n))
            self.labels = np.array([np.zeros(len(set(labels)))] * len(labels))
            print('Number of classes: ' + str(len(set(labels))))
            for i in range(len(labels)): # one-hot encoding
                self.labels[i][int(labels[i]) - 1] = 1
            self.num_labels = labels

class ClsEstimator:
    def __init__(self, features, labels, num_labels, classifiers):
        self.features = features
        self.labels = labels
        self.num_labels = num_labels
        self.classifiers = classifiers

    def estimate(self):
        train_features, test_features, train_labels_inds, test_labels_inds = train_test_split(self.features, range(len(self.labels)), test_size=0.2, random_state=42)
        for (cls, cls_name) in self.classifiers:
            train_labels, test_labels = None, None
            if cls_name == 'neural network':
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
                for (metrics, metrics_name) in [(precision, 'precision'), (recall, 'recall'), (fscore, 'fscore'), (accuracy, 'accuracy')]:
                    print('{} of {} on {} sample ({} instances): {}'.format(metrics_name, cls_name, n, len(l), metrics))

fd = FormatDetector(file)
dl = DataLoader(file, fd.n, fd.d)
classifiers = [
               # (make_pipeline(StandardScaler(), SVC()), 'SVM'),
               (DTC(), 'decision tree'),
               (GNB(), 'naive bayes'),
               (make_pipeline(StandardScaler(), LR()), 'logistic regression'),
               (make_pipeline(StandardScaler(), NNClassifier(np.shape(dl.features)[1], np.shape(dl.labels)[1])), 'neural network')]
ce = ClsEstimator(dl.features, dl.labels, dl.num_labels, classifiers)
ce.estimate()
