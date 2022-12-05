import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from aopssop import DataScaler, AIForecaster, ForecastEstimator
from aopssop import AopssopData as PDATA

from threading import Thread
import psutil
import pandas as pd

pid = os.getpid()
proc = psutil.Process(pid)
cpu_num = psutil.cpu_count()
proc.as_dict()

from logger import Logger
LOG0 = Logger(proc, cpu_num)


class DataLoaderAndPreprocessor:
    """
    Class for loading and preprocessing data.

    :param DATA_PATH: Path to th e directory with datasets (string).
    :param drop_features: Set of features to remove from the data (list).
    :param categorical_features: A set of categorical features in the data (list).
    :param data: Data feature matrix (pd.DataFrame).
    """
    def __init__(self, dataset_name, mode=1, suf=''):
        """
        Initializing.

        :param dataset_name: Data set name (srting).
        :param mode: Boot mode, for developers (integer).
        """

        DATA_PATH = "../datasets/"

        self.features_names = []
        self.drop_features = []
        self.categorical_features = []
        self.data = None

        if not os.path.exists('apssop_models/'):
            os.makedirs('apssop_models/')

        self.forecasting_model_path = 'apssop_models/forecasting_model_' + dataset_name + suf
        self.normalization_model_path = 'apssop_models/scaler_' + dataset_name + suf + '.pkl'

        if dataset_name == 'hai':
            DATA_PATH = DATA_PATH + 'hai-22.04/train'

            if mode == 0:
                self.data = pd.read_csv(DATA_PATH + '/train1.csv', nrows=10000)

            else:
                self.data = pd.DataFrame()
                for file in os.listdir(DATA_PATH):
                    self.data = pd.concat([self.data, pd.read_csv(DATA_PATH + '/' + file)],
                                           ignore_index=True)

            self.drop_features = ['timestamp', 'Attack']
            self.preprocessing()
            self.features_names = self.data.columns.values


        # elif dataset_name == 'alarms':
        #     DATA_PATH = DATA_PATH + 'IEEE_alarms_log_data/raw'
        #     if mode == 'simple_test':
        #         self.data = pd.read_csv(DATA_PATH + '/alarms.csv', nrows=10000)
        #     else:
        #         self.data = pd.read_csv(DATA_PATH + '/alarms.csv')
        #     self.drop_features = ['timestamp']
        #     self.preprocessing()
        #     self.features_names = self.data.columns.values

        # elif dataset_name == 'edge-iiotset':
        #     DATA_PATH = DATA_PATH + 'Edge-IIoTset/'
        #     self.data = pd.read_csv(DATA_PATH + 'Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv',
        #                             low_memory=False)
        #
        #     self.data['mqtt.conack.flags'] = self.data['mqtt.conack.flags'].map(lambda x: 0 if x == '0x00000000' else x)
        #     self.data['mqtt.conack.flags'] = self.data['mqtt.conack.flags'].astype(float).round()
        #
        #     self.data['mqtt.protoname'] = self.data['mqtt.protoname'].map(lambda x: '0' if x == '0.0' else x)
        #     self.data['mqtt.topic'] = self.data['mqtt.topic'].map(lambda x: '0' if x == '0.0' else x)
        #
        #     self.drop_features = ['frame.time', 'Attack_label', 'Attack_type',
        #                          'tcp.options', 'tcp.payload',
        #                           'http.file_data', 'http.request.uri.query', 'http.request.method',
        #                           'http.referer', 'http.request.full_uri', 'http.request.version',
        #                           'tcp.srcport', 'dns.qry.name.len',
        #                           'mqtt.msg']
        #
        #     self.categorical_features = ['ip.src_host', 'ip.dst_host',
        #                                  'arp.dst.proto_ipv4', 'arp.src.proto_ipv4',
        #                                   'mqtt.protoname', 'mqtt.topic']
        #     self.preprocessing()
        #     self.features_names = self.data.columns.values

        elif dataset_name == 'smart_crane':
            # DATA_PATH = DATA_PATH + 'IEEE_smart_crane'
            # self.data = pd.read_csv(DATA_PATH + '/combined_csv.csv')
            self.data = pd.read_csv(DATA_PATH + 'IEEE_smart_crane.csv')

            if mode not in range(1, 9):
                print('Wrong cycle number')
                exit()
            self.data = self.data[self.data['Cycle'] == mode]

            self.drop_features = ['Date', 'Alarm', 'Cycle']
            self.preprocessing()
            self.features_names = self.data.columns.values

        elif dataset_name == 'test':
            self.data = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)),
                                     columns=list('ABCD'))
            self.features_names = self.data.columns.values

        else:
            print('Unknown dataset name')
            exit()

    def preprocessing(self):
        """
        Data feature preprocessing.
        """
        if len(self.drop_features) > 0:
            self.data = self.data.drop(self.drop_features, axis=1)

        if len(self.categorical_features) > 0:
            for feature in self.categorical_features:
                if 'ip' in feature:
                    encoded_feature_data = pd.DataFrame([x.split('.')
                                                         if x != '0' else [0, 0, 0, 0]
                                                         for x in self.data[feature].tolist()])

                    for col in encoded_feature_data.columns:
                        encoded_feature_data = encoded_feature_data.rename(columns={col: feature + '_' + str(col)})
                else:
                    try:
                        encoded_feature_data = pd.get_dummies(self.data[feature])
                        for col in encoded_feature_data.columns:
                            encoded_feature_data = encoded_feature_data.rename(columns={col: feature + '_' + col})
                    except:
                        encoded_feature_data = pd.DataFrame()

                if not encoded_feature_data.empty:
                    old_data_columns = self.data.columns.tolist()
                    feature_index = old_data_columns.index(feature)
                    new_data_columns = old_data_columns[:feature_index] + \
                                       encoded_feature_data.columns.tolist() + \
                                       old_data_columns[feature_index+1:]

                    self.data = pd.concat([self.data, encoded_feature_data], axis=1)
                    self.data = self.data[new_data_columns]
                else:
                    print('Too many values for categorical feature "' + feature +'". Delete feature from data')
                    self.data = self.data.drop(feature, axis=1)
        self.data = self.data.fillna(0)


    def train_test_split(self, train_size=0.9):
        """
        Split data into training and test sets.

        :param train_size: Share of the training sample (float). Default=0.9.
        :return: train: Feature matrix of training data (pd.DataFrame).
        :return test: Feature matrix of test data (pd.DataFrame).
        """
        if (train_size < 0) and (train_size > 1):
            print('The proportion of the training sample is not in the interval (0, 1)')
            return None, None

        train_ind = round(train_size*self.data.shape[0])

        train = self.data.iloc[:train_ind]
        test = self.data.iloc[train_ind:]
        return train, test


def example_appsop_model_training(dataset_name, suf='', mode=1):
    """
    Testing APPSOP module methods.
    In this example, forecasting model and  normalization model are trained on the dataset.
    """

    LOG0.create(dataset_name + suf + '_training_res.log', rewrite=True)
    LOG0.run = True
    thread1 = Thread(target=LOG0.daemon_logger)
    thread1.start()

    LOG0.event_init(text='Start with dataset "' + dataset_name + '"')
    LOG0.event_init(event_name='preproc', text='Input data preprocessing')
    data = DataLoaderAndPreprocessor(dataset_name, mode=mode, suf=suf)
    PDATA.features_names = data.features_names
    LOG0.event_init(event_name='preproc',
                    text='Input data preprocessed. Shape ({0}, {1})'.format(data.data.shape[0], data.data.shape[1]))

    PDATA.forecasting_model_path = data.forecasting_model_path
    PDATA.normalization_model_path = data.normalization_model_path

    LOG0.event_init(event_name='preproc', text='Input data splitting')
    train, test = data.train_test_split(train_size=0.9)
    LOG0.event_init(event_name='preproc',
                    text='Data is divided into train and test samples of length {0} and {1}'.format(len(train),
                                                                                                    len(test)))
    LOG0.event_init(event_name='norm', text='Normalization model training')
    normalization_model = DataScaler(scaler_path=PDATA.normalization_model_path)
    normalization_model.fit(train)
    LOG0.event_init(event_name='norm', text='Data normalization')
    scaled_train = normalization_model.transform(train)
    LOG0.event_init(event_name='norm', text='Data normalized')

    LOG0.event_init(event_name='prepare', text='Forecasting model creation:' + PDATA.forecasting_model_path)

    forecasting_model = AIForecaster(n_epochs=3,
                                     time_window_length=PDATA.time_window_length,
                                     n_features=len(PDATA.features_names),
                                     model_path=PDATA.forecasting_model_path,
                                     )

    LOG0.event_init(event_name='prepare', text='Data generator creation')
    train_generator = forecasting_model.data_to_generator(scaled_train)
    LOG0.event_init(event_name='prepare', text='Data generator created')

    LOG0.event_init(event_name='train', text='Model training')
    LOG0.show_off()
    loss = forecasting_model.train(train_generator)
    LOG0.show_on()
    LOG0.event_init(event_name='train', text='Training completed (loss:' + str(loss) + ')')

    LOG0.run = False
    thread1.join()
    LOG0.close()
    print('Done')


def example_appsop_forecasting(dataset_name, suf='', mode=1,
                                                   train_size=0.9,
                                                   independently=True,
                                                   sample_type='test'):
    """
    Testing APPSOP module methods
    In this example, test set samples are predicted one at a time based on incoming test set data.
    """
    if sample_type not in ['train', 'test']:
        print('Wrong sample type')
        exit()

    if independently:
        file_suf = '_' + sample_type + '_independently'
    else:
        file_suf = '_' + sample_type + '_dependently'

    LOG0.create(dataset_name + suf + file_suf + '_res.log', rewrite=True)
    LOG0.run = True
    thread1 = Thread(target=LOG0.daemon_logger)
    thread1.start()

    LOG0.event_init(text='Start with dataset "' + dataset_name + '"')
    LOG0.event_init(event_name='preproc', text='Input data preprocessing')
    data = DataLoaderAndPreprocessor(dataset_name, mode=mode, suf=suf)
    PDATA.features_names = data.features_names
    LOG0.event_init(event_name='preproc',
                    text='Inpit data preprocessed. Shape ({0}, {1})'.format(data.data.shape[0], data.data.shape[1]))

    PDATA.forecasting_model_path = data.forecasting_model_path
    PDATA.normalization_model_path = data.normalization_model_path

    LOG0.event_init(event_name='preproc', text='Input data splitting')
    train, test = data.train_test_split(train_size=train_size)
    LOG0.event_init(event_name='preproc',text='Data is divided into train and test samples of length {0} and {1}'.format(len(train),
                                                                                                    len(test)))
    LOG0.event_init(event_name='norm', text='Normalization model open')
    normalization_model = DataScaler(scaler_path=PDATA.normalization_model_path,
                                     open=True)
    if not normalization_model:
        print('Wrong normalization model filename')
        exit()

    LOG0.event_init(event_name='norm', text='Data normalization')
    scaled_train = normalization_model.transform(train)
    scaled_test = normalization_model.transform(test)
    LOG0.event_init(event_name='norm', text='Data normalized')

    LOG0.event_init(event_name='prepare', text='Forecasting model creation:' + PDATA.forecasting_model_path)
    forecasting_model = AIForecaster(model_path=PDATA.forecasting_model_path,
                                     open=True)

    LOG0.event_init(event_name='prepare', text='Data generator creation')
    PDATA.time_window_length = forecasting_model.time_window_length
    train_generator = forecasting_model.data_to_generator(scaled_train)
    LOG0.event_init(event_name='prepare', text='Data generator created')

    predictions = []
    current_batch = []
    target_true = []

    LOG0.event_init(event_name='prepare', text='Get batch for forecasting')
    if sample_type == 'train':
        PDATA.forecasting_time_window = len(scaled_train) - PDATA.time_window_length
        current_batch = forecasting_model.get_batch(train_generator, 0)
        target_true = scaled_train[PDATA.time_window_length:]

    elif sample_type == 'test':
        PDATA.forecasting_time_window = len(scaled_test)
        current_batch = forecasting_model.get_batch(train_generator, -1)
        target_true = scaled_test

    LOG0.event_init(event_name='forecast', text='Forecasting')
    if independently:
        for i in range(PDATA.forecasting_time_window):
            current_pred = forecasting_model.forecasting(current_batch,
                                                         forecasting_data_length=1,
                                                         verbose=True)
            predictions.append(current_pred[0])
            new_event = target_true[i]
            current_batch = np.append(current_batch[:, 1:, :], [[new_event]], axis=1)

            LOG0.event_init(event_name='forecast', text='Forecasting: {0}/{1}'.format(i,
                                                                                      PDATA.forecasting_time_window - 1))

    else:
        LOG0.show_off()
        predictions = forecasting_model.forecasting(current_batch,
                                                    forecasting_data_length=PDATA.forecasting_time_window,
                                                    verbose=True)
        LOG0.show_on()

    predictions = pd.DataFrame(predictions).values
    PDATA.forecasting_results = normalization_model.inverse(predictions)

    LOG0.event_init(event_name='forecast', text='Forecasting complited')
    LOG0.event_init(event_name='eval', text='Evaluation')

    estimator = ForecastEstimator()
    PDATA.forecasting_quality = estimator.estimate(true=target_true,
                                                   pred=predictions,
                                                   feature_names=PDATA.features_names)

    estimator.save(file_name=dataset_name + suf + file_suf)
    LOG0.event_init(event_name='eval', text='Evaluation done')
    print(PDATA.forecasting_quality)

    LOG0.run = False
    thread1.join()
    LOG0.close()
    print('Done')


if __name__ == '__main__':
    PDATA.time_window_length = 10
    train_size = 0.9

    dataset_name = 'smart_crane'
    for mode in range(1, 9):
        suf = '_ex2_c' + str(mode)
        example_appsop_model_training(dataset_name, suf, mode)
        example_appsop_forecasting(dataset_name, suf, mode, sample_type='train')
        example_appsop_forecasting(dataset_name, suf, mode, sample_type='test')
        example_appsop_forecasting(dataset_name, suf, mode, sample_type='test', independently=False)

    suf = '_ex2'
    dataset_name = 'hai'
    example_appsop_model_training(dataset_name, suf)
    example_appsop_forecasting(dataset_name, suf, sample_type='train')
    example_appsop_forecasting(dataset_name, suf, sample_type='test')
    example_appsop_forecasting(dataset_name, suf, sample_type='test', independently=False)