import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.utils import shuffle
from aopssop import DataLoader


class DataLoaderHai(DataLoader):
    """
    Data Loader for HAI dataset
    """
    def __init__(self, file, n, d):
        DataLoader.__init__(self, file, n, d)

    def load(self, file):
        n, d = self.n, self.d
        if n in [64, 84]:
            self.features = np.genfromtxt(file, delimiter=d, dtype=float, skip_header=1, usecols=range(1, n - 4))
            attacks = np.genfromtxt(file, delimiter=d, dtype=float, skip_header=1, usecols=range(n - 4, n))
            self.labels = np.array([[0] if sum(a) == 0 else [1] for a in attacks])
            self.num_labels = self.labels
        elif n == 88:
            self.features = np.genfromtxt(file, delimiter=d, dtype=float, skip_header=1, usecols=range(1, n - 1))
            self.labels = np.genfromtxt(file, delimiter=d, dtype=float, skip_header=1, usecols=range(n - 1, n))
            self.num_labels = self.labels


class DataLoaderEdgeIIoTSet(DataLoader):
    """
    Data Loader for Edge IIoT dataset
    """
    def __init__(self, file, n, d):
        DataLoader.__init__(self, file, n, d)

    def load(self, file):
        df = pd.read_csv(file, low_memory=False, sep=self.d)
        drop_columns = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
                        "http.request.method", "http.file_data", "http.referer", "http.request.full_uri",
                        "http.request.version",
                        "icmp.transmit_timestamp", "http.request.uri.query", "tcp.options", "tcp.payload",
                        "tcp.srcport",
                        "tcp.dstport", "udp.port", "dns.qry.name", "dns.qry.name.len", "dns.qry.qu",
                        "mqtt.conack.flags", "mqtt.msg", "mqtt.protoname", "mqtt.topic", "Attack_label"]
        df.drop(drop_columns, axis=1, inplace=True)
        df.dropna(axis=0, how='any', inplace=True)
        df.drop_duplicates(subset=None, keep="first", inplace=True)
        df = shuffle(df)
        str_labels = df.iloc[:, -1].tolist()
        self.features = np.array(df.iloc[:, :-1].values.tolist())
        unique_labels = list(OrderedDict.fromkeys(str_labels))
        labels = list(map(lambda x: unique_labels.index(x), str_labels))
        self.labels = np.array([np.zeros(len(unique_labels))] * len(labels))
        for i in range(len(labels)):  # one-hot encoding
            self.labels[i][labels[i]] = 1
        self.num_labels = labels


class DataLoaderDataPort(DataLoader):
    """
    Data Loader for Dataport dataset
    """
    def __init__(self, file, n, d):
        DataLoader.__init__(self, file, n, d)

    def load(self, file):
        self.features = np.genfromtxt(file, delimiter=self.d, dtype=float, skip_header=1, usecols=range(1, self.n - 1))
        labels = np.genfromtxt(file, delimiter=self.d, dtype=float, skip_header=1, usecols=range(self.n - 1, self.n))
        self.labels = np.array([np.zeros(len(set(labels)))] * len(labels))
        for i in range(len(labels)):  # one-hot encoding
            self.labels[i][int(labels[i]) - 1] = 1
        self.num_labels = labels