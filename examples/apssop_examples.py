import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from aopssop import DataScaler, AIForecaster, ForecastEstimator, Logger, DataLoader
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


def example_appsop_model_training(dataset_name, suf='', mode=1):
    """
    Testing APPSOP module methods.
    In this example, forecasting model and  normalization model are trained on the dataset.

    :param dataset_name: name of dataset
    :type dataset_name: str
    
    :param suf: suffix for naming the output
    :type suf: str
    
    :param mode: boot mode, for developers
    :type mode: integer
    """

    # Create log file.
    LOG0.create(dataset_name + suf + '_training_res.log', rewrite=True)
    # Start logging.
    LOG0.run = True
    thread1 = Thread(target=LOG0.daemon_logger)
    thread1.start()

    LOG0.event_init(text='Start with dataset "' + dataset_name + '"')
    LOG0.event_init(event_name='preproc', text='Input data preprocessing')
    # Load data.
    data = DataLoader(dataset_name, mode=mode, suf=suf)
    PDATA.features_names = data.features_names
    LOG0.event_init(event_name='preproc',
                    text='Input data preprocessed. Shape ({0}, {1})'.format(data.data.shape[0], data.data.shape[1]))

    PDATA.forecasting_model_path = data.forecasting_model_path
    PDATA.normalization_model_path = data.normalization_model_path

    # Input data splitting to train and test samples.
    LOG0.event_init(event_name='preproc', text='Input data splitting')
    train, test = data.train_test_split(train_size=0.9)
    LOG0.event_init(event_name='preproc',
                    text='Data is divided into train and test samples of length {0} and {1}'.format(len(train), len(test)))

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
    # Create data generator for training
    train_generator = forecasting_model.data_to_generator(scaled_train)
    LOG0.event_init(event_name='prepare', text='Data generator created')

    LOG0.event_init(event_name='train', text='Model training')
    LOG0.show_off()
    loss = forecasting_model.train(train_generator)
    # Train forecasting model.
    LOG0.show_on()
    LOG0.event_init(event_name='train', text='Training completed (loss:' + str(loss) + ')')

    # Stop logging.
    LOG0.run = False
    thread1.join()
    LOG0.close()
    print('Done')


def example_appsop_forecasting(dataset_name, suf='', mode=1,
                                                   train_size=0.9,
                                                   independently=True,
                                                   sample_type='test'):
    """
    Testing APPSOP module methods.
    Example of data forecasting based on an existing model, including predictive estimation

    :param dataset_name: name of dataset
    :type dataset_name: str
    
    :param suf: suffix for naming the output
    :type suf: str
    
    :param mode: boot mode, for developers
    :type mode: integer
    
    :param independently: sequence is predicted depending on past values or not
    :type independently: boolean
    
    :param sample_type: type of forecasting sample for estimation - train or test
    :type sample_type: str
    """
    if sample_type not in ['train', 'test']:
        print('Wrong sample type')
        exit()

    if independently:
        file_suf = '_' + sample_type + '_independently'
    else:
        file_suf = '_' + sample_type + '_dependently'

    # Create log.
    LOG0.create(dataset_name + suf + file_suf + '_res.log', rewrite=True)
    LOG0.run = True
    # Start logging.
    thread1 = Thread(target=LOG0.daemon_logger)
    thread1.start()

    LOG0.event_init(text='Start with dataset "' + dataset_name + '"')
    LOG0.event_init(event_name='preproc', text='Input data preprocessing')
    # Data load.
    data = DataLoader(dataset_name, mode=mode, suf=suf)
    PDATA.features_names = data.features_names
    LOG0.event_init(event_name='preproc',
                    text='Inpit data preprocessed. Shape ({0}, {1})'.format(data.data.shape[0], data.data.shape[1]))

    PDATA.forecasting_model_path = data.forecasting_model_path
    PDATA.normalization_model_path = data.normalization_model_path

    LOG0.event_init(event_name='preproc', text='Input data splitting')
    # Input data splitting to train and test samples.
    train, test = data.train_test_split(train_size=train_size)
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
    current_batch = []
    target_true = []

    LOG0.event_init(event_name='prepare', text='Get batch for forecasting')
    if sample_type == 'train':
        # For estimation on training data, forecasting time window is equal to the length of training sample
        # from the second to the last batch.
        PDATA.forecasting_time_window = len(scaled_train) - PDATA.time_window_length
        # Batch for forecasting is the first batch of the training sample.
        current_batch = forecasting_model.get_batch(train_generator, 0)
        # True values for estimation is values of the training sample from the second to the last batch.
        target_true = scaled_train[PDATA.time_window_length:]

    elif sample_type == 'test':
        # For estimation on test data, forecasting time window is equal to the length of all test sample.
        PDATA.forecasting_time_window = len(scaled_test)
        # Batch for forecasting is the last batch of the training sample.
        current_batch = forecasting_model.get_batch(train_generator, -1)
        # True values is values of the test sample.
        target_true = scaled_test

    LOG0.event_init(event_name='forecast', text='Forecasting')
    if independently:
        # All feature value vectors are predicted independently of each other.
        for i in range(PDATA.forecasting_time_window):
            current_pred = forecasting_model.forecasting(current_batch,
                                                         forecasting_data_length=1,
                                                         verbose=True)
            predictions.append(current_pred[0])
            new_event = target_true[i]
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
    PDATA.forecasting_quality = estimator.estimate(true=target_true,
                                                   pred=predictions,
                                                   feature_names=PDATA.features_names)
    # Save evaluation result.
    estimator.save(file_name=dataset_name + suf + file_suf)
    LOG0.event_init(event_name='eval', text='Evaluation done')
    print(PDATA.forecasting_quality)

    # Stop logging.
    LOG0.run = False
    thread1.join()
    LOG0.close()
    print('Done')


if __name__ == '__main__':
    PDATA.time_window_length = 10
    train_size = 0.9

    dataset_name = 'smart_crane'
    for mode in range(1, 9):
        suf = '_ex1_c' + str(mode)
        example_appsop_model_training(dataset_name, suf, mode)
        example_appsop_forecasting(dataset_name, suf, mode, sample_type='train')
        example_appsop_forecasting(dataset_name, suf, mode, sample_type='test')
        example_appsop_forecasting(dataset_name, suf, mode, sample_type='test', independently=False)

    suf = '_ex1'
    dataset_name = 'hai'
    example_appsop_model_training(dataset_name, suf)
    example_appsop_forecasting(dataset_name, suf, sample_type='train')
    example_appsop_forecasting(dataset_name, suf, sample_type='test')
    example_appsop_forecasting(dataset_name, suf, sample_type='test', independently=False)