import os
import pandas as pd
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from foressment_ai.forecasting.forecaster_ai.forecaster import AIForecaster, ForecastEstimator, TSGenerator


def generate_test_data(shape=(1000,1)):
    time = np.arange(0, shape[0]*0.1, 0.1)
    data = pd.DataFrame(index=time)
    for i in range(shape[1]):
        data['feature_'+ str(i)] = np.sin(time) + np.random.normal(scale=0.5, size=len(time))
    return data

def main(mode='single-step'):

    data = generate_test_data(shape=(1000,1))
    train_size = int(len(data) * 0.8)
    train, test = data.iloc[0:train_size].values, data.iloc[train_size:len(data)].values

    param_filename = 'lstm_3l_50lb_1h.json'
    ts = TSGenerator()
    ts.set_generator_params(filename='ai_models/params/'+param_filename)

    X_train, y_train = ts.temporalize(train)
    X_test, y_test = ts.temporalize(test)

    fm = AIForecaster()
    fm.set_model_params(filename='ai_models/params/'+param_filename)
    fm.set_model_params({'n_features': data.shape[1]})
    fm.build()
    fm.save_model_config(filename='ai_models/configs/' + param_filename)

    fm.train(X_train, y_train, batch_size=16, n_epochs=30, validation_split=0.1)

    if mode == 'single-step':
        test_pred = fm.forecasting(X_test)

    elif mode == 'multi-step':
        current_batch = ts.get_window(X_test, 0)
        test_pred = fm.forecasting(current_batch, forecasting_data_length=y_test.shape[0])
        
    else:
        exit()

    y_test_eval = ts.flatten(y_test)[:test_pred.shape[0]]

    estimator = ForecastEstimator()
    forecasting_quality = estimator.estimate(true=y_test_eval, pred=test_pred, feature_names=data.columns.tolist())
    estimator.draw(true=y_test_eval, pred=test_pred)
    print(forecasting_quality)


if __name__ == '__main__':
    # main('single-step')
    main('multi-step')