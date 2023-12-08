import keras
import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import classification_report
from keras import layers
from keras.metrics import Recall, Precision, AUC
import matplotlib.pyplot as plt
from keras.utils import plot_model
import numpy as np


class AutoEncoder_:
    """
    To implement the basic auto encoder with chosen structure.
    """

    def __init__(self, input_shape, model_type='dnn', ae_start_dim=64):
        """
        Initialize the AutoEncoder_ class.

        :param input_shape: The shape of the input data
        :type input_shape: tuple

        :param model_type: Type of model structure to use ('dnn', 'cnn', 'rnn')
        :type model_type: str

        :param ae_start_dim: Starting dimension for the autoencoder
        :type ae_start_dim: int
        """
        self.input_shape = input_shape
        self.model_type = model_type
        self.ae_start_dim = ae_start_dim
        self.num_categories = None
        self.autoencoder = self.define_model()

    def define_model(self):
        """
        Define the autoencoder model based on the specified model type.

        :return: Autoencoder model
        :rtype: tf.keras.Model
        """
        if self.model_type == 'cnn':
            return self.cnn_ae()
        elif self.model_type == 'dnn':
            return self.dnn_ae()
        elif self.model_type == 'rnn':
            return self.rnn_ae()
        else:
            raise ValueError(
                "Invalid model_type. To specify the model structure, choose 'cnn', 'dnn', 'rnn', 'lstm', 'bidirectional_lstm', or 'gru'.")

    def cnn_ae(self):
        """
        Define the CNN autoencoder model.

        :return: CNN autoencoder model
        :rtype: tf.keras.Model
        """
        inputs = tf.keras.Input(shape=self.input_shape)
        input_dim = self.ae_start_dim
        origin_dim = self.input_shape[0]
        # Reshape the input to match the expected shape for 1D convolution
        x = tf.keras.layers.Reshape((origin_dim, 1))(inputs)
        # Define the encoder layers
        x = tf.keras.layers.Conv1D(input_dim, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling1D()(x)
        x = tf.keras.layers.Conv1D(input_dim * 2, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling1D()(x)
        # Define the decoder layers
        x = tf.keras.layers.Convolution1D(input_dim * 2, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.UpSampling1D()(x)
        x = tf.keras.layers.Conv1D(input_dim, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.UpSampling1D()(x)
        x = tf.keras.layers.Conv1D(1, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        # Define the output layer
        x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(origin_dim)(x)
        autoencoder = Model(inputs=inputs, outputs=outputs, name='dnn_ae')
        return autoencoder

    def dnn_ae(self):
        """
        Define the DNN autoencoder model.

        :return: DNN autoencoder model
        :rtype: tf.keras.Model
        """
        inputs = tf.keras.Input(shape=self.input_shape)
        input_dim = self.input_shape[0]
        # Define the encoder layers
        x = tf.keras.layers.Dense(input_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.5),
                                  kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01))(
            inputs)
        x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
        x = tf.keras.layers.Dense(int(input_dim / 2), activation=tf.keras.layers.LeakyReLU(alpha=0.5),
                                  kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
        x = tf.keras.layers.Dense(int(input_dim / 4), activation=tf.keras.layers.LeakyReLU(alpha=0.5),
                                  kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
        # Define the decoder layers
        x = tf.keras.layers.Dense(int(input_dim / 2), activation=tf.keras.layers.LeakyReLU(alpha=0.5),
                                  kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
        x = tf.keras.layers.Dense(input_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.5),
                                  kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        outputs = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
        autoencoder = Model(inputs=inputs, outputs=outputs, name='convolutional_ae')
        return autoencoder

    def rnn_ae(self):
        """
        Define the RNN autoencoder model.

        :return: RNN autoencoder model
        :rtype: tf.keras.Model
        """
        inputs = tf.keras.Input(shape=self.input_shape)
        input_dim = self.ae_start_dim
        origin_dim = self.input_shape[0]
        # Reshape the input to match the expected shape for 1D convolution
        x = tf.keras.layers.Reshape((origin_dim, 1))(inputs)
        # Define the encoder layers
        x = tf.keras.layers.SimpleRNN(input_dim, activation='relu', return_sequences=True)(x)
        x = tf.keras.layers.SimpleRNN(int(input_dim / 2), activation='relu', return_sequences=True)(x)
        x = tf.keras.layers.SimpleRNN(int(input_dim / 4), activation='relu', return_sequences=True)(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        # Define the decoder layers
        x = tf.keras.layers.SimpleRNN(int(input_dim / 2), activation='relu', return_sequences=True)(x)
        x = tf.keras.layers.SimpleRNN(origin_dim, activation='relu', return_sequences=False)(x)
        outputs = tf.keras.layers.Dropout(0.2)(x)
        autoencoder = tf.keras.Model(inputs=inputs, outputs=outputs, name='rnn_ae')
        return autoencoder

    def viz_model(self, show_shapes=True, show_layer_names=True):
        """
        Visualize the defined model.

        :param show_shapes: Whether to display shapes in the visualization
        :type show_shapes: bool

        :param show_layer_names: Whether to display layer names in the visualization
        :type show_layer_names: bool
        """
        model = self.define_model()
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        plt.show()

    def summary(self):
        model = self.define_model()

    def fit(self, X_train, y_train, validation_data, epochs=10, batch_size=32, verbose=0):

        if self.num_categories is None:
            loss = 'mse'
            metrics = [tf.keras.metrics.MeanSquaredError()]
        model = self.define_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=loss, metrics=metrics)
        self.history = model.fit(X_train, y_train, validation_data=validation_data, epochs=epochs,
                                 batch_size=batch_size, verbose=verbose)
        return self.history

    def test(self, X_test):
        return self.autoencoder.evaluate(X_test, X_test)

    def predict(self, X):
        return self.autoencoder.predict(X)

    def draw_plot(self, plot_type="accuracy"):
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


def cnn_discriminator(input_shape, depth=3, classes=2):
    """
    Define a CNN discriminator model for classification.

    :param input_shape: The shape of the input data
    :type input_shape: tuple

    :param depth: Depth of the CNN model
    :type depth: int

    :param classes: Number of classes for classification
    :type classes: int

    :return: CNN discriminator model
    :rtype: tf.keras.Model
    """
    input_layer = layers.Input(shape=input_shape)
    if len(input_shape) == 1:
        x = layers.Reshape((input_shape[0], 1))(input_layer)
    else:
        x = input_layer
    if input_shape[0] < 20:
        width = 64
    else:
        width = 128

    for _ in range(depth):
        x = layers.Conv1D(width, kernel_size=3, padding="same")(x)
        x = layers.BatchNormalization(scale=False)(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling1D()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(width, activation='relu')(x)
    x = layers.Dense(width, activation='relu')(x)
    x = layers.Dense(width, activation='relu')(x)
    x = layers.Dense(width, activation='relu')(x)
    x = layers.Dense(width, activation='relu')(x)
    if classes > 2:
        outputs = layers.Dense(classes, activation="softmax", name="prediction_multiclass")(x)
    elif classes == 2:
        outputs = layers.Dense(1, activation="sigmoid", name="prediction_binary")(x)
    discriminator = Model(input_layer, outputs, name="clf")
    if classes > 2:
        loss = 'categorical_crossentropy'
    else:
        loss = 'binary_crossentropy'
    metrics = ['accuracy', Recall(), Precision(), AUC(name='auc')]
    discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)
    return discriminator


class GAN:
    def __init__(self, input_shape, ae_model_type='dnn', ae_start_dim=64, depth=3, classes=2):
        """
        Initialize the GAN class.

        :param input_shape: The shape of the input data
        :type input_shape: tuple

        :param ae_model_type: Type of model structure for the autoencoder ('dnn', 'cnn', 'rnn')
        :type ae_model_type: str

        :param ae_start_dim: Starting dimension for the autoencoder
        :type ae_start_dim: int

        :param depth: Depth of the CNN model for discriminator
        :type depth: int

        :param classes: Number of classes for classification
        :type classes: int
        """
        self.input_shape = input_shape
        self.ae_model_type = ae_model_type
        self.ae_start_dim = ae_start_dim
        self.depth = depth
        self.classes = classes
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()

    def build_generator(self):
        """
        Build the generator using the specified autoencoder model.

        :return: Generator model
        :rtype: tf.keras.Model
        """
        autoencoder = AutoEncoder_(self.input_shape, model_type=self.ae_model_type, ae_start_dim=self.ae_start_dim)
        return autoencoder.autoencoder

    def build_discriminator(self):
        """
        Build the discriminator model for classification.

        :return: Discriminator model
        :rtype: tf.keras.Model
        """
        discriminator = cnn_discriminator(self.input_shape, depth=self.depth, classes=self.classes)
        return discriminator

    def build_gan(self):
        """
        Build the GAN model by combining the generator and discriminator.

        :return: GAN model
        :rtype: tf.keras.Model
        """
        # self.discriminator.trainable = False
        gan_input = layers.Input(shape=self.input_shape)
        generated_data = self.generator(gan_input)
        gan_output = self.discriminator(generated_data)
        gan = Model(gan_input, gan_output)
        gan.compile(loss='binary_crossentropy', optimizer=Adam())
        # gan.summary()
        return gan

    def train(self, x_train, y_train, x_valid, y_valid, epochs, batch_size):
        """
        Train the GAN model using the given training and validation data.

        :param x_train: Training input data
        :type x_train: numpy.ndarray

        :param y_train: Training target data
        :type y_train: numpy.ndarray

        :param x_valid: Validation input data
        :type x_valid: numpy.ndarray

        :param y_valid: Validation target data
        :type y_valid: numpy.ndarray

        :param epochs: Number of training epochs
        :type epochs: int

        :param batch_size: Batch size for training
        :type batch_size: int

        :return: Training history
        :rtype: dict
        """
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        history = {
            'd_loss': [],
            'd_accuracy': [],
            'd_recall': [],
            'd_precision': [],
            'd_auc': [],
            'g_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_recall': [],
            'val_precision': [],
            'val_auc': []
        }
        for epoch in range(epochs):
            # Generate fake instances using the generator
            num_instances = len(x_train)
            num_fake = int(num_instances * 0.1)

            fake_instances = self.generator.predict(x_train[:num_fake])

            # Concatenate the real and fake instances
            x_combined = tf.concat([x_train, fake_instances], axis=0)
            y_combined = tf.concat([y_train, y_train[:num_fake]], axis=0)

            # Train the discriminator
            d_loss, d_accuracy, d_recall, d_precision, d_auc = self.discriminator.train_on_batch(x_combined, y_combined)

            # Train the generator
            gan_input = tf.random.normal(shape=(len(x_train), self.input_shape[0]))
            g_loss = self.gan.train_on_batch(gan_input, y_train)

            # Calculate validation loss
            val_loss, val_accuracy, val_recall, val_precision, val_auc = self.discriminator.evaluate(x_valid, y_valid,
                                                                                                     verbose=0)

            print(
                f"Epoch {epoch + 1}/{epochs}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}, Validation Loss: {val_loss}")

            # save history
            history['d_loss'].append(d_loss)
            history['d_accuracy'].append(d_accuracy)
            history['d_recall'].append(d_recall)
            history['d_precision'].append(d_precision)
            history['d_auc'].append(d_auc)
            history['g_loss'].append(g_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            history['val_recall'].append(val_recall)
            history['val_precision'].append(val_precision)
            history['val_auc'].append(val_auc)
            self.history = history
        return self.history

    def predict(self, data):
        """
        Generate predictions using the trained discriminator.

        :param data: Input data for prediction
        :type data: numpy.ndarray

        :return: Predicted values
        :rtype: numpy.ndarray
        """
        return self.discriminator.predict(data)

    def print_classification_report(self, x_test, y_test, target_names=None):
        """
        Print the classification report based on the test data.

        :param x_test: Test input data
        :type x_test: numpy.ndarray

        :param y_test: Test target data
        :type y_test: numpy.ndarray

        :param target_names: Optional list of target class names
        :type target_names: list
        """
        y_pred = self.gan.predict(x_test)
        # Check if it's a binary classification problem
        if y_pred.shape[1] == 1:
            # Round predicted probabilities to obtain binary labels
            y_pred_classes = np.round(y_pred).flatten()
            y_test_classes = y_test.flatten()
        else:
            # For multi-label classification, get the class with the highest probability
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_test_classes = np.argmax(y_test, axis=1)
        # Generate classification report
        report = classification_report(y_test_classes, y_pred_classes, target_names=target_names)
        print(report)

    def draw_plot(self, plot_type="accuracy"):
        """
        Draw the specified type of plot based on the training history.

        :param plot_type: Type of plot to draw ('accuracy', 'loss', 'auc')
        :type plot_type: str
        """
        if plot_type == "accuracy":
            plt.plot(self.history['d_accuracy'])
            plt.plot(self.history['val_accuracy'])
            plt.title('Discriminator Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend(['Train', 'Validation'])
        elif plot_type == "loss":
            plt.plot(self.history['d_loss'])
            plt.plot(self.history['val_loss'])
            plt.title('Discriminator Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(['Train', 'Validation'])
        elif plot_type == "auc":
            plt.plot(self.history['d_auc'])
            plt.plot(self.history['val_auc'])
            plt.title('Discriminator AUC')
            plt.xlabel('Epoch')
            plt.ylabel('AUC')
            plt.legend(['Train', 'Validation'])
        else:
            print("Invalid plot_type. Choose 'accuracy', 'loss', or 'auc'.")
        plt.show()