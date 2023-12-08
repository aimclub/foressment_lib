import keras
from keras.models import Model
from keras.metrics import Recall, Precision, AUC
import matplotlib.pyplot as plt
from keras.utils import plot_model


class AutoEncoder:
    """
    Class for implementing an autoencoder with the option to add a classifier and visualize the model.

    :param input_shape: The shape of the input data
    :type input_shape: tuple.

    :param model_type: The type of model to use, choose from 'dnn', 'cnn', 'rnn', 'lstm', 'bidirectional_lstm', or 'gru'
    :type model_type: str

    :param ae_start_dim: The starting dimension for the autoencoder
    :type ae_start_dim: int

    :param classifier: Whether to add a classifier to the autoencoder
    :type classifier: bool

    :param num_categories: Number of categories for the classifier (if classifier is True)
    :type num_categories: int, optional
    """

    def __init__(self, input_shape, model_type='dnn', ae_start_dim=64, classifier=False, num_categories=None):
        """
        Initialize the AutoEncoder with the specified parameters

        :param input_shape: The shape of the input data
        :type input_shape: tuple

        :param model_type: The type of model to use, choose from 'dnn', 'cnn', 'rnn', 'lstm', 'bidirectional_lstm', or 'gru'
        :type model_type: str

        :param ae_start_dim: The starting dimension for the autoencoder
        :type ae_start_dim: int

        :param classifier: Whether to add a classifier to the autoencoder
        :type classifier: bool

        :param num_categories: Number of categories for the classifier (if classifier is True)
        :type num_categories: int, optional
        """
        self.input_shape = input_shape
        self.model_type = model_type
        self.ae_start_dim = ae_start_dim
        self.classifier = classifier
        self.num_categories = num_categories
        self.model = self.define_model()

    def define_model(self):
        """
        Define the structure of the autoencoder model based on the specified model type
        """
        if self.model_type == 'cnn':
            return self.cnn_ae()
        elif self.model_type == 'dnn':
            return self.dnn_ae()
        elif self.model_type == 'rnn':
            return self.rnn_ae()
        elif self.model_type == 'lstm':
            return self.lstm_ae()
        elif self.model_type == 'bidirectional_lstm':
            return self.bilstm_ae()
        elif self.model_type == 'gru':
            return self.gru_ae()
        else:
            raise ValueError(
                "Invalid model_type. To specify the model structure, choose 'cnn', 'dnn', 'rnn', 'lstm', 'bidirectional_lstm', or 'gru'.")

    def add_classifier(self, autoencoder):
        """
        Add a classifier to the autoencoder model

        :param autoencoder: The autoencoder model
        :type autoencoder: keras.Model

        :return: The combined autoencoder and classifier model
        :rtype: keras.Model
        """
        inputs = autoencoder.inputs
        clf = keras.layers.Dense(128, activation='relu')(autoencoder.layers[-1].output)
        clf = keras.layers.Dense(128, activation='relu')(clf)
        clf = keras.layers.Dense(128, activation='relu')(clf)
        clf = keras.layers.Dense(128, activation='relu')(clf)
        clf = keras.layers.Dense(128, activation='relu')(clf)
        if self.num_categories == 2:
            clf_output = keras.layers.Dense(1, activation='sigmoid', name="output_clf")(clf)
        else:
            clf_output = keras.layers.Dense(self.num_categories, activation='softmax', name="output_clf")(clf)
        classifier_model = Model(inputs=inputs, outputs=clf_output, name="ae_classifier")
        return classifier_model

    def cnn_ae(self):
        inputs = keras.Input(shape=self.input_shape)
        input_dim = self.ae_start_dim
        origin_dim = self.input_shape[0]
        # Reshape the input to match the expected shape for 1D convolution
        x = keras.layers.Reshape((origin_dim, 1))(inputs)
        # Define the encoder layers
        x = keras.layers.Conv1D(input_dim, 3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.MaxPooling1D()(x)
        x = keras.layers.Conv1D(input_dim * 2, 3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.MaxPooling1D()(x)
        # Define the decoder layers
        x = keras.layers.Convolution1D(input_dim * 2, 3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.UpSampling1D()(x)
        x = keras.layers.Conv1D(input_dim, 3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.UpSampling1D()(x)
        x = keras.layers.Conv1D(1, 3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        # Define the output layer
        x = keras.layers.Flatten()(x)
        outputs = keras.layers.Dense(origin_dim)(x)
        autoencoder = Model(inputs=inputs, outputs=outputs, name='dnn_ae')
        if self.classifier == False:
            model = autoencoder
        else:
            model = self.add_classifier(autoencoder)
        return model

    def dnn_ae(self):
        inputs = keras.Input(shape=self.input_shape)
        input_dim = self.input_shape[0]
        # Define the encoder layers
        x = keras.layers.Dense(input_dim, activation=keras.layers.LeakyReLU(alpha=0.5),
                                  kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(0.01))(
            inputs)
        x = keras.layers.BatchNormalization(momentum=0.8)(x)
        x = keras.layers.Dense(int(input_dim / 2), activation=keras.layers.LeakyReLU(alpha=0.5),
                                  kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = keras.layers.BatchNormalization(momentum=0.8)(x)
        x = keras.layers.Dense(int(input_dim / 4), activation=keras.layers.LeakyReLU(alpha=0.5),
                                  kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = keras.layers.BatchNormalization(momentum=0.8)(x)
        # Define the decoder layers
        x = keras.layers.Dense(int(input_dim / 2), activation=keras.layers.LeakyReLU(alpha=0.5),
                                  kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = keras.layers.BatchNormalization(momentum=0.8)(x)
        x = keras.layers.Dense(input_dim, activation=keras.layers.LeakyReLU(alpha=0.5),
                                  kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l2(0.01))(x)
        outputs = keras.layers.BatchNormalization(momentum=0.8)(x)
        autoencoder = Model(inputs=inputs, outputs=outputs, name='convolutional_ae')
        if self.classifier == False:
            model = autoencoder
        else:
            model = self.add_classifier(autoencoder)
        return model

    def rnn_ae(self):
        inputs = keras.Input(shape=self.input_shape)
        input_dim = self.ae_start_dim
        origin_dim = self.input_shape[0]
        # Reshape the input to match the expected shape for 1D convolution
        x = keras.layers.Reshape((origin_dim, 1))(inputs)
        # Define the encoder layers
        x = keras.layers.SimpleRNN(input_dim, activation='relu', return_sequences=True)(x)
        x = keras.layers.SimpleRNN(int(input_dim / 2), activation='relu', return_sequences=True)(x)
        x = keras.layers.SimpleRNN(int(input_dim / 4), activation='relu', return_sequences=True)(x)
        x = keras.layers.Dropout(0.2)(x)
        # Define the decoder layers
        x = keras.layers.SimpleRNN(int(input_dim / 2), activation='relu', return_sequences=True)(x)
        x = keras.layers.SimpleRNN(origin_dim, activation='relu', return_sequences=False)(x)
        outputs = keras.layers.Dropout(0.2)(x)
        autoencoder = keras.Model(inputs=inputs, outputs=outputs, name='rnn_ae')
        if self.classifier == False:
            model = autoencoder
        else:
            model = self.add_classifier(autoencoder)
        return model

    def lstm_ae(self):
        inputs = keras.Input(shape=self.input_shape)
        input_dim = self.ae_start_dim
        origin_dim = self.input_shape[0]
        timesteps = 1
        # Reshape the input to match the expected shape for 1D convolution
        x = keras.layers.Reshape((origin_dim, 1))(inputs)
        # Define the encoder layers
        x = keras.layers.LSTM(input_dim, activation='relu', return_sequences=True)(x)
        x = keras.layers.LSTM(int(input_dim / 2), activation='relu', return_sequences=False)(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.RepeatVector(timesteps)(x)
        # Define the decoder layers
        x = keras.layers.LSTM(int(input_dim / 2), activation='relu', return_sequences=True)(x)
        x = keras.layers.LSTM(input_dim, activation='relu', return_sequences=True)(x)
        x = keras.layers.LSTM(origin_dim, activation='relu', return_sequences=False)(x)
        outputs = keras.layers.Dropout(0.2)(x)
        autoencoder = keras.Model(inputs=inputs, outputs=outputs, name='rnn_ae')
        if self.classifier == False:
            model = autoencoder
        else:
            model = self.add_classifier(autoencoder)
        return model

    def bilstm_ae(self):
        inputs = keras.Input(shape=self.input_shape)
        input_dim = self.ae_start_dim
        origin_dim = self.input_shape[0]
        timesteps = 1
        # Reshape the input to match the expected shape for 1D convolution
        x = keras.layers.Reshape((origin_dim, 1))(inputs)
        # Define the encoder layers
        x = keras.layers.Bidirectional(keras.layers.LSTM(input_dim, activation='relu', return_sequences=True))(x)
        x = keras.layers.Bidirectional(
            keras.layers.LSTM(int(input_dim / 2), activation='relu', return_sequences=False))(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.RepeatVector(timesteps)(x)
        # Define the decoder layers
        x = keras.layers.Bidirectional(
            keras.layers.LSTM(int(input_dim / 2), activation='relu', return_sequences=True))(x)
        x = keras.layers.Bidirectional(keras.layers.LSTM(input_dim, activation='relu', return_sequences=True))(x)
        x = keras.layers.Bidirectional(keras.layers.LSTM(origin_dim, activation='relu', return_sequences=False))(
            x)
        outputs = keras.layers.Dropout(0.2)(x)
        autoencoder = keras.Model(inputs=inputs, outputs=outputs, name='rnn_ae')
        if self.classifier == False:
            model = autoencoder
        else:
            model = self.add_classifier(autoencoder)
        return model

    def gru_ae(self):
        inputs = keras.Input(shape=self.input_shape)
        input_dim = self.ae_start_dim
        origin_dim = self.input_shape[0]
        timesteps = 1
        # Reshape the input to match the expected shape for 1D convolution
        x = keras.layers.Reshape((origin_dim, 1))(inputs)
        # Define the encoder layers
        x = keras.layers.GRU(input_dim, activation='relu', return_sequences=True)(x)
        x = keras.layers.GRU(int(input_dim / 2), activation='relu', return_sequences=False)(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.RepeatVector(timesteps)(x)
        # Define the decoder layers
        x = keras.layers.GRU(int(input_dim / 2), activation='relu', return_sequences=True)(x)
        x = keras.layers.GRU(input_dim, activation='relu', return_sequences=True)(x)
        x = keras.layers.GRU(origin_dim, activation='relu', return_sequences=False)(x)
        outputs = keras.layers.Dropout(0.2)(x)
        autoencoder = keras.Model(inputs=inputs, outputs=outputs, name='rnn_ae')
        if self.classifier == False:
            model = autoencoder
        else:
            model = self.add_classifier(autoencoder)
        return model

    def viz_model(self, show_shapes=True, show_layer_names=True):
        """
        Visualize the model architecture and save the plot to a file

        :param show_shapes: Whether to display shapes on the plot
        :type show_shapes: bool, optional

        :param show_layer_names: Whether to display layer names on the plot
        :type show_layer_names: bool, optional
        """
        model = self.define_model()
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        plt.show()

    def fit(self, X_train, y_train, validation_data, epochs=10, batch_size=32, verbose=0):
        """
        Train the autoencoder model

        :param X_train: The input training data
        :type X_train: array-like

        :param y_train: The target training data
        :type y_train: array-like

        :param validation_data: The validation data
        :type validation_data: tuple

        :param epochs: The number of epochs for training
        :type epochs: int, optional

        :param batch_size: The batch size for training
        :type batch_size: int, optional

        :param verbose: Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch)
        :type verbose: int, optional

        :return: The training history
        :rtype: keras.callbacks.History
        """
        if self.num_categories is None:
            loss = 'mse'
            metrics = [keras.metrics.MeanSquaredError()]
        elif self.num_categories > 2:
            loss = 'categorical_crossentropy'
            metrics = ['accuracy', Recall(), Precision(), AUC(name='auc')]
        else:
            loss = 'binary_crossentropy'
            metrics = ['accuracy', Recall(), Precision(), AUC(name='auc')]
        # model = self.define_model()
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=loss, metrics=metrics)
        self.history = self.model.fit(X_train, y_train, validation_data=validation_data, epochs=epochs,
                                      batch_size=batch_size, verbose=verbose)
        return self.history

    def test(self, X_test, y_test=None):
        """
        Evaluate the autoencoder model on the test data

        :param X_test: The input test data
        :type X_test: array-like

        :param y_test: The target test data (optional)
        :type y_test: array-like, optional

        :return: Evaluation results
        :rtype: list or dict
        """
        if y_test is None:
            return self.model.evaluate(X_test, X_test)
        else:
            return self.model.evaluate(X_test, y_test)

    def predict(self, X):
        """
        Generate output predictions for the input data

        :param X: The input data for prediction
        :type X: array-like

        :return: Predicted output
        :rtype: array-like
        """
        return self.model.predict(X)

    def draw_mse_plot(self, plot_type="mse"):
        """
        Draw a plot of Mean Squared Error (MSE) or Loss over training epochs

        :param plot_type: The type of plot to draw, choose from 'loss' or 'mse'
        :type plot_type: str
        """
        if plot_type == "loss":
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(['Train', 'Validation'])
        elif plot_type == "mse":
            plt.plot(self.history.history['mean_squared_error'])
            plt.plot(self.history.history['val_mean_squared_error'])
            plt.title('MSE')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(['Train', 'Validation'])
        else:
            print("Invalid plot_type. Choose 'loss' or 'mse'.")
        plt.show()

    def draw_clf_plot(self, plot_type="accuracy"):
        """
        Draw a plot of classifier metrics over training epochs

        :param plot_type: The type of plot to draw, choose from 'accuracy', 'loss', or 'auc'
        :type plot_type: str
        """
        if plot_type == "accuracy":
            plt.plot(self.history.history['accuracy'])
            plt.plot(self.history.history['val_accuracy'])
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend(['Train', 'Validation'])
        elif plot_type == "loss":
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(['Train', 'Validation'])
        elif plot_type == "auc":
            plt.plot(self.history.history['auc'])
            plt.plot(self.history.history['val_auc'])
            plt.title('Model AUC')
            plt.xlabel('Epoch')
            plt.ylabel('AUC')
            plt.legend(['Train', 'Validation'])
        else:
            print("Invalid plot_type. Choose 'accuracy', 'loss', or 'auc'.")
        plt.show()