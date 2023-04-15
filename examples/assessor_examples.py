import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

from foressment_ai import SAIClassifier, FormatDetector, ClsEstimator
from foressment_ai import DataLoaderHai, DataLoaderEdgeIIoTSet, DataLoaderDataPort

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

def assessor_dataset_examples():
    """
    Examples of the application of this algorithm cover the task of ensuring the cybersecurity of critical resources
    and objects, as well as the task of determining the trajectory of a vehicle (crane). In the first case,
    the data obtained from the sensors of the system of steam turbines and pumped storage power plants are
    considered as input data. In the second case, the input data are parameters that describe the operation
    and movement of the overhead crane under various loads.

    The essence of the experiment was to test the suitability of a pre-configured model as part of the task of
    assessing the state of a critically important object. During the experiment, two phases were distinguished:
    the training phase and the testing phase. At the first phase, the weights of the neural network were adjusted,
    and at the second phase, the calculation of performance indicators for estimating the state
    of the analyzed object was carried out.
    """
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


if __name__ == '__main__':
    assessor_dataset_examples()