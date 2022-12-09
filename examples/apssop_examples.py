import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from aopssop import DataScaler, AIForecaster, ForecastEstimator, Logger, DataLoaderAndPreprocessor
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