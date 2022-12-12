import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

from aopssop import SAIClassifier, FormatDetector, ClsEstimator
from aopssop import DataLoaderHai, DataLoaderEdgeIIoTSet, DataLoaderDataPort

# set value of DATASET
DATASET = 'hai'
#DATASET = 'edge-iiotset'
#DATASET = 'dataport'

# set value of HAI_FILE
HAI_FILE = '../datasets/HAI_test1.csv.gz'

# EDGE_IIOTSET_FILE = '../edge-iiotset/ML-EdgeIIoT-dataset.csv'
# DATAPORT_FILE = '../dataport/combined_csv.csv'

datasets = {
    'hai': HAI_FILE,
    # 'edge-iiotset': EDGE_IIOTSET_FILE,
    # 'dataport': DATAPORT_FILE
}


if __name__ == '__main__':
    file = datasets[DATASET]
    fd = FormatDetector(file)

    dl = None
    if DATASET == 'hai':
        dl = DataLoaderHai(file, fd.n, fd.d)

    # elif DATASET == 'edge-iiotset':
    #     dl = DataLoaderEdgeIIoTSet(file, fd.n, fd.d)
    #
    # elif DATASET == 'dataport':
    #     dl = DataLoaderDataPort(file, fd.n, fd.d)

    else:
        assert False

    classifiers = [
        SAIClassifier(
            cls_type=c,
            in_size=np.shape(dl.features)[1],
            out_size=np.shape(dl.labels)[1]
        )
        for c in [
            'decision_tree', 'naive_bayes',
            'logistic_regression', 'neural_network'
        ]
    ]
    ce = ClsEstimator(dl.features, dl.labels, dl.num_labels, classifiers)
    r = ce.estimate(print_metrics=True)
    print(r)
