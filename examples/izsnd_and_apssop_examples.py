import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from aopssop import DataScaler, AIForecaster, ForecastEstimator, \
    Logger, DataLoaderIZDAP
from aopssop import AopssopData as PDATA

from threading import Thread
import psutil
import pandas as pd
import numpy as np

pid = os.getpid()
proc = psutil.Process(pid)
cpu_num = psutil.cpu_count()
proc.as_dict()
LOG0 = Logger(proc, cpu_num)


def train_test_split(data, train_size=0.9):
        """
        Split data into training and test sets.

        :param train_size: Share of the training sample (default=0.9)
        :type train_size: float

        :return train: Feature matrix of training data
        :rtype train: pandas.DataFrame

        :return test: Feature matrix of test data
        :rtype test: pandas.DataFrame
        """
        if (train_size < 0) and (train_size > 1):
            print('The proportion of the training sample is not in the interval (0, 1)')
            return None, None

        train_ind = round(train_size*data.shape[0])

        train = data.iloc[:train_ind]
        test = data.iloc[train_ind:]
        return train, test


def example_appsop_model_training(logname='example'):
    """
    Testing APPSOP module methods.
    In this example, forecasting model and normalization model are trained on the dataset.

    :param logname: name of log file
    :type logname: str

    :return result: dictionary with params of resources
    :rtype result: dict
    """
    logfile = logname + '_training_res.log'
    # Create log file.
    LOG0.create(logfile, rewrite=True)
    # Start logging.
    LOG0.run = True
    thread1 = Thread(target=LOG0.daemon_logger)
    thread1.start()

    LOG0.event_init(text='Start with dataset "')

    LOG0.event_init(event_name='preproc', text='Input data splitting')
    # Input data splitting to train and test samples.
    train, test = train_test_split(PDATA.features_matrix, train_size=0.9)

    result = {'test': logfile, 'data_size': train.shape}

    LOG0.event_init(event_name='preproc',
                    text='Data is divided into train and test samples of length {0} and {1}'.format(len(train),
                                                                                                    len(test)))
    # Create normalization model.
    LOG0.event_init(event_name='norm', text='Normalization model training')
    normalization_model = DataScaler(scaler_path=PDATA.normalization_model_path)
    # Normalization model training.
    normalization_model.fit(train)
    LOG0.event_init(event_name='norm', text='Data normalization')
    # Data normalization.
    scaled_train = normalization_model.transform(train)
    LOG0.event_init(event_name='norm', text='Data normalized')

    LOG0.event_init(event_name='prepare', text='Forecasting model creation:' + PDATA.forecasting_model_path)
    # Forecasting model creation.
    forecasting_model = AIForecaster(n_epochs=3,
                                     time_window_length=PDATA.time_window_length,
                                     n_features=len(PDATA.features_names),
                                     model_path=PDATA.forecasting_model_path,
                                     )

    LOG0.event_init(event_name='prepare', text='Data generator creation')
    # Create data generator for training.
    train_generator = forecasting_model.data_to_generator(scaled_train)
    LOG0.event_init(event_name='prepare', text='Data generator created')

    LOG0.event_init(event_name='train', text='Model training')
    LOG0.show_off()
    # Train forecasting model. Get MSE of training.
    result['mse'] = forecasting_model.train(train_generator)
    LOG0.show_on()
    LOG0.event_init(event_name='train', text='Training completed')

    # Stop logging.
    LOG0.run = False
    thread1.join()
    LOG0.close()
    # Resources calculation.
    result.update(LOG0.get_resources(event_name='train'))

    print('Done')
    return result


def example_appsop_forecasting(logname='example',
                               apriori=True):
    """
    Testing APPSOP module methods
    Example of data forecasting based on an existing model, including predictive estimation

    :param logname: name of log file
    :type logname: str

    :param apriori: sequence is predicted depending on past values or not
    :type apriori: boolean

    :return result: dictionary with params of resources
    :rtype result: dict
    """

    if apriori:
        logname = logname + '_apriori'
    else:
        logname = logname + '_aposteriori'

    logfile = logname + '_forecasting_res.log'

    # Create log.
    LOG0.create(logfile, rewrite=True)
    # Start logging.
    LOG0.run = True
    thread1 = Thread(target=LOG0.daemon_logger)
    thread1.start()

    LOG0.event_init(text='Start with dataset "')
    LOG0.event_init(event_name='preproc', text='Input data splitting')
    # Input data splitting to train and test samples.
    train, test = train_test_split(PDATA.features_matrix, train_size=0.9)
    # Use one size of test sample for all experiment.
    test = test[:1000]

    result = {'test': logfile, 'data_size': test.shape}

    LOG0.event_init(event_name='preproc',text='Data is divided into train and test samples of length {0} and {1}'.format(len(train),
                                                                                                    len(test)))
    LOG0.event_init(event_name='norm', text='Normalization model open')
    # Load normalization model.
    normalization_model = DataScaler(scaler_path=PDATA.normalization_model_path,
                                     open=True)
    if not normalization_model:
        print('Wrong normalization model filename')
        exit()

    LOG0.event_init(event_name='norm', text='Data normalization')
    # Data normalization.
    scaled_train = normalization_model.transform(train)
    scaled_test = normalization_model.transform(test)
    LOG0.event_init(event_name='norm', text='Data normalized')

    LOG0.event_init(event_name='prepare', text='Forecasting model creation:' + PDATA.forecasting_model_path)
    # Load forecasting model.
    forecasting_model = AIForecaster(model_path=PDATA.forecasting_model_path,
                                     open=True)

    LOG0.event_init(event_name='prepare', text='Data generator creation')
    PDATA.time_window_length = forecasting_model.time_window_length
    # Create train data generator.
    train_generator = forecasting_model.data_to_generator(scaled_train)
    LOG0.event_init(event_name='prepare', text='Data generator created')

    predictions = []

    LOG0.event_init(event_name='prepare', text='Get batch for forecasting')
    # For estimation on test data, forecasting time window is equal to the length of all test sample.
    PDATA.forecasting_time_window = len(scaled_test)
    # Batch for forecasting is the last batch of the training sample.
    current_batch = forecasting_model.get_batch(train_generator, -1)

    LOG0.event_init(event_name='forecast', text='Forecasting')

    if apriori:
        # All feature value vectors are predicted independently of each other.
        for i in range(PDATA.forecasting_time_window):
            current_pred = forecasting_model.forecasting(current_batch,
                                                         forecasting_data_length=1,
                                                         verbose=False)
            predictions.append(current_pred[0])
            new_event = scaled_test[i]
            # At each stage, an element of the target sample is added to the batch.
            current_batch = np.append(current_batch[:, 1:, :], [[new_event]], axis=1)

            LOG0.event_init(event_name='forecast', text='Forecasting: {0}/{1}'.format(i,
                                                                                      PDATA.forecasting_time_window - 1))
    else:
        LOG0.show_off()
        # Each predicted vector becomes an element of a new package for subsequent forecasting.
        predictions = forecasting_model.forecasting(current_batch,
                                                    forecasting_data_length=PDATA.forecasting_time_window,
                                                    verbose=True)
        LOG0.show_on()

    predictions = pd.DataFrame(predictions).values
    # Inverse data transformation.
    PDATA.forecasting_results = normalization_model.inverse(predictions)

    LOG0.event_init(event_name='forecast', text='Forecasting complited')
    LOG0.event_init(event_name='eval', text='Evaluation')

    # Quality evaluation of the forecasting model.
    estimator = ForecastEstimator()
    PDATA.forecasting_quality = estimator.estimate(true=scaled_test,
                                                   pred=predictions,
                                                   feature_names=PDATA.features_names)

    result['mse'] = PDATA.forecasting_quality.loc['ALL_FEATURES', 'MSE']
    result['mae'] = PDATA.forecasting_quality.loc['ALL_FEATURES', 'MAE']
    result['1-mae'] = 1 - PDATA.forecasting_quality.loc['ALL_FEATURES', 'MAE']

    LOG0.event_init(event_name='eval', text='Evaluation done')

    # Stop logging.
    LOG0.run = False
    thread1.join()
    LOG0.close()
    # Resources calculation.
    result.update(LOG0.get_resources(event_name='forecast'))

    print('Done')
    return result


def example_join_test_smart_crane():
    PDATA.time_window_length = 10

    result = pd.DataFrame(columns=['test', 'data_size', 'mse', 'mae', '1-mae',
                                   'duration_sec', 'cpu%_min', 'cpu%_mean', 'cpu%_max',
                                   'ram_mb_min', 'ram_mb_mean', 'ram_mb_max'
                                   ])

    dataset_name = 'smart_crane'
    for mode in range(1, 9):
        for test_name, use_izdap in {'c' + str(mode) + '_appsop_izdap0': False,
                                     'c' + str(mode) + '_appsop_izdap1': True}.items():

            Loader = DataLoaderIZDAP(test_name=test_name)
            PDATA.features_matrix = Loader(dataset_name, mode=mode, use_izdap=use_izdap)
            PDATA.features_names = Loader.features_names
            PDATA.forecasting_model_path = Loader.forecasting_model_path
            PDATA.normalization_model_path = Loader.normalization_model_path

            for ex in [str(i) + '_' for i in range(3)]:
                logname = ex + dataset_name + '_' + test_name
                if ex == '0_':
                    result = result.append(example_appsop_model_training(logname=logname), ignore_index=True)
                if ex == '1_':
                    result = result.append(example_appsop_forecasting(logname=logname), ignore_index=True)
                if ex == '2_':
                    result = result.append(example_appsop_forecasting(logname=logname, apriori=False), ignore_index=True)
    result = result.sort_values(by='test').reset_index(drop=True)
    result.to_csv('appsop_logs/join_test_' + dataset_name + '.csv', index=False, sep=';', decimal=',')


def example_join_test_hai():
    PDATA.time_window_length = 10
    result = pd.DataFrame(columns=['test', 'data_size', 'mse', 'mae', '1-mae',
                                   'duration_sec', 'cpu%_min', 'cpu%_mean', 'cpu%_max',
                                   'ram_mb_min', 'ram_mb_mean', 'ram_mb_max'
                                   ])

    dataset_name = 'hai'
    for test_name, use_izdap in {'appsop_izdap0': False,
                                 'appsop_izdap1': True}.items():

        Loader = DataLoaderIZDAP(test_name=test_name)
        PDATA.features_matrix = Loader(dataset_name, use_izdap=use_izdap)
        PDATA.features_names = Loader.features_names
        PDATA.forecasting_model_path = Loader.forecasting_model_path
        PDATA.normalization_model_path = Loader.normalization_model_path

        for ex in [str(i) + '_' for i in range(3)]:
            logname = ex + dataset_name + '_' + test_name
            if ex == '0_':
                result = result.append(example_appsop_model_training(logname=logname), ignore_index=True)
            if ex == '1_':
                result = result.append(example_appsop_forecasting(logname=logname), ignore_index=True)
            if ex == '2_':
                result = result.append(example_appsop_forecasting(logname=logname, apriori=False), ignore_index=True)
    result = result.sort_values(by='test').reset_index(drop=True)
    result.to_csv('appsop_logs/join_test_' + dataset_name + '.csv', index=False, sep=';', decimal=',')


if __name__ == '__main__':
    example_join_test_smart_crane()
    # example_join_test_hai()


