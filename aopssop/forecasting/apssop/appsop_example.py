from apssop import *
from data_classes import AopssopData as PDATA


class DataLoaderAndPreprocessor:
    """
    Class for loading and preprocessing data.

    Attributes:
        DATA_PATH: Path to the directory with datasets (string).
        drop_features: Set of features to remove from the data (list).
        categorical_features: A set of categorical features in the data (list).
        data: Data feature matrix (pd.DataFrame).
    """
    def __init__(self, dataset_name, mode=''):
        """
        Initializing.
        :param dataset_name: Data set name (srting).
        :param mode:Boot mode, for developers (string).
        """

        DATA_PATH = "../_datasets/"

        self.features_names = []
        self.drop_features = []
        self.categorical_features = []
        self.data = None

        if dataset_name == 'hai':
            DATA_PATH = DATA_PATH + 'HAI Security Dataset/hai-22.04/train'

            if mode == 'simple_test':
                self.data = pd.read_csv(DATA_PATH + '/train1.csv', nrows=1000)

            else:
                self.data = pd.DataFrame()
                for file in os.listdir(DATA_PATH)[:1]:
                    self.data = pd.concat([self.data, pd.read_csv(DATA_PATH + '/' + file)],
                                           ignore_index=True)

            self.drop_features = ['timestamp', 'Attack']
            self.preprocessing()
            self.features_names = self.data.columns.values

        elif dataset_name == 'alarms':
            DATA_PATH = DATA_PATH + 'IEEE_alarms_log_data/raw'
            if mode == 'simple_test':
                self.data = pd.read_csv(DATA_PATH + '/alarms.csv', nrows=1000)
            else:
                self.data = pd.read_csv(DATA_PATH + '/alarms.csv')
            self.drop_features = ['timestamp']
            self.preprocessing()
            self.features_names = self.data.columns.values

        elif dataset_name == 'edge-iiotset':
            DATA_PATH = DATA_PATH + 'Edge-IIoTset/'
            self.data = pd.read_csv(DATA_PATH + 'Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv',
                                    low_memory=False)

            self.data['mqtt.conack.flags'] = self.data['mqtt.conack.flags'].map(lambda x: 0 if x == '0x00000000' else x)
            self.data['mqtt.conack.flags'] = self.data['mqtt.conack.flags'].astype(float).round()

            self.data['mqtt.protoname'] = self.data['mqtt.protoname'].map(lambda x: '0' if x == '0.0' else x)
            self.data['mqtt.topic'] = self.data['mqtt.topic'].map(lambda x: '0' if x == '0.0' else x)

            self.drop_features = ['frame.time', 'Attack_label', 'Attack_type',
                                 'tcp.options', 'tcp.payload',]

            self.categorical_features = ['ip.src_host', 'ip.dst_host',
                                         'arp.dst.proto_ipv4', 'arp.src.proto_ipv4',
                                          'mqtt.protoname', 'mqtt.topic']
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
                    encoded_feature_data = pd.DataFrame([x.split('.') if x != '0' else [0, 0, 0, 0]
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


    def train_test_split(self, train_size=0.9):
        """
        Split data into training and test sets.
        :param train_size: Share of the training sample (float). Default=0.9.
        :return: train: Feature matrix of training data (pd.DataFrame).
                 test: Feature matrix of test data (pd.DataFrame).
        """
        if (train_size < 0) and (train_size > 1):
            print('The proportion of the training sample is not in the interval (0, 1)')
            return None, None

        train_ind = round(train_size*self.data.shape[0])

        train = self.data.iloc[:train_ind]
        test = self.data.iloc[train_ind:]
        return train, test


def test_appsop_example(dataset_name):
    """
    Testing APPSOP module methods
    """

    PDATA.forecasting_model_path = 'models/forecasting_model_' + dataset_name + '.h5'
    PDATA.normalization_model_path = 'models/scaler_' + dataset_name + '.pkl'

    PDATA.time_window_length = 10

    print('Start with dataset "' + dataset_name + '"')

    data = DataLoaderAndPreprocessor(dataset_name)
    PDATA.features_names = data.features_names

    train, test = data.train_test_split(train_size=0.9)

    normalization_model = DataScaler(scaler_path=PDATA.normalization_model_path)
    normalization_model.fit(train)

    scaled_train = normalization_model.transform(train)
    scaled_test = normalization_model.transform(test)

    forecasting_model = AIForecaster(n_epochs=3,
                                     time_window_length=PDATA.time_window_length,
                                     n_features=len(PDATA.features_names),
                                     model_path=PDATA.forecasting_model_path,
                                     open=True
                                     )

    train_generator = forecasting_model.data_to_generator(scaled_train)
    forecasting_model.train(train_generator)

    PDATA.forecasting_time_window = len(scaled_test)
    last_batch = forecasting_model.get_last_batch(train_generator)

    pred = forecasting_model.forecasting(last_batch,
                                         forecasting_data_length=PDATA.forecasting_time_window)
    PDATA.forecasting_results = normalization_model.inverse(pred)

    PDATA.forecasting_quality = ForecastEstimator().estimate(true=scaled_test, pred=pred,
                                                             feature_names=PDATA.features_names)

    PDATA.forecasting_quality.save(file_name=dataset_name)

    print(PDATA.forecasting_quality)

    print('Done')


if __name__ == '__main__':
    for dataset_name in ['hai', 'alarms', 'edge-iiotset']:
        test_appsop_example('alarms')