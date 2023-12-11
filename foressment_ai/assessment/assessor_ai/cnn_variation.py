import keras
from keras.models import Model
from keras import layers
from keras.metrics import Recall, Precision, AUC
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np

# residual

def residual_block(x, filters, kernel_size=3):
    """
    Implement a residual block for a 1D convolutional neural network.

    param x: Input tensor
    type x: tensor

    param filters: The dimensionality of the output space (i.e., the number of output filters in the convolution)
    type filters: int

    param kernel_size: Size of the convolution kernel
    type kernel_size: int, optional

    :return: Output tensor after applying the residual block
    type: tensor
    """
    y = layers.Conv1D(filters, kernel_size, padding='same')(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    y = layers.Conv1D(filters, kernel_size, padding='same')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    y = layers.Add()([x, y])  # Skip connection
    return y


class hybrid_variation:
    def __init__(self, input_shape=None, units=64, classes=2, block="Xception", loop_number=1):
        """
        Initialize the hybrid model with specified parameters and build the model.

        param input_shape: The shape of the input data
        type input_shape: tuple, optional

        param units: Number of units for convolutional and GRU layers
        type units: int, optional

        param classes: Number of classes for classification
        type classes: int, optional

        param block: Type of block to use in the model ("Xception" or "residual")
        type block: str, optional

        param loop_number: Number of times to apply the selected block
        type loop_number: int, optional
        """
        self.model = self.build_model(input_shape, units, classes, block, loop_number)

    def build_model(self, input_shape, units, classes, block, loop_number):
        """
        Define the structure of the hybrid model based on the specified parameters.

        param input_shape: The shape of the input data
        type input_shape: tuple

        param units: Number of units for convolutional and GRU layers
        type units: int

        param classes: Number of classes for classification
        type classes: int

        param block: Type of block to use in the model ("Xception" or "residual")
        type block: str

        param loop_number: Number of times to apply the selected block
        type loop_number: int
        """
        inputs = layers.Input(shape=input_shape)
        unit = units
        x = layers.Conv1D(unit, 1, padding="same", use_bias=False)(inputs)
        # Implementation of selected block
        for i in range(loop_number):
            if block == "Xception":
                # Xception block implementation
                if unit * 2 > 256:
                    unit = 256
                else:
                    unit = unit * 2

                residual = layers.Conv1D(unit, 1, strides=2, padding="same", use_bias=False)(x)
                residual = layers.BatchNormalization()(residual)

                x = layers.SeparableConv1D(unit, 3, padding="same", use_bias=False)(x)
                x = layers.BatchNormalization()(x)
                x = layers.Activation("relu")(x)
                x = layers.SeparableConv1D(unit, 3, padding="same", use_bias=False)(x)
                x = layers.BatchNormalization()(x)
                x = layers.MaxPooling1D(3, strides=2, padding="same")(x)
                x = layers.add([x, residual])

            elif block == "residual":
                # Residual block implementation
                x = residual_block(x, units)

            else:
                print("Please choose one of these options:'Xception' or 'residual'.")

        x = layers.Dropout(0.2)(x)
        x = layers.GRU(units, return_sequences=True)(x)
        x = layers.GRU(units * 2, return_sequences=True)(x)
        x = layers.GRU(units * 2, return_sequences=False)(x)

        if classes > 2:
            outputs = layers.Dense(classes, activation="softmax", name="prediction_multiclass")(x)
        elif classes == 2:
            outputs = layers.Dense(1, activation="sigmoid", name="prediction_binary")(x)

        model = Model(inputs=inputs, outputs=outputs)

        if classes > 2:
            loss = 'categorical_crossentropy'
        else:
            loss = 'binary_crossentropy'

        metrics = ['accuracy', Recall(), Precision(), AUC(name='auc')]
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=loss, metrics=metrics)

        return model

    def print_summary(self):
        """
        Print a summary of the model architecture.
        """
        self.model.summary()

    def fit(self, X_train, y_train, validation_data, epochs=10, batch_size=128, verbose=0):
        """
        Train the hybrid model.

        :param X_train: The input training data
        :type X_train: array-like

        :param y_train: The target training data
        :type y_train: array-like

        :param validation_data: The validation data
        :type validation_data: tuple

        :param epochs: The number of epochs for training
        :type epochs: int, optional

        :param batch_size: The batch size for training
        :type batch_size: int

        :param verbose: Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch)
        :type verbose: int, optional

        :return: The training history
        :rtype: keras.callbacks.History
        """
        self.history = self.model.fit(X_train, y_train, validation_data=validation_data, epochs=epochs,
                                      batch_size=batch_size, verbose=verbose)
        return self.history

    def test(self, X_test, y_test):
        """
        Evaluate the model on the test data.

        :param X_test: The input test data
        :type X_test: array-like

        :param y_test: The target test data
        :type y_test: array-like

        :return: Evaluation results
        :rtype: list
        """
        return self.model.evaluate(X_test, y_test)

    def predict(self, X):
        """
        Generate predictions for the input data.

        :param X: The input data for prediction
        :type X: array-like

        :return: Predicted output
        :rtype: array-like
        """
        return self.model.predict(X)

    def print_classification_report(self, x_test, y_test, target_names=None):
        """
        Print the classification report based on the test data and true labels.

        :param x_test: The input test data
        :type x_test: array-like

        :param y_test: The target test data
        :type y_test: array-like

        :param target_names: Optional target class names
        :type target_names: list, optional
        """
        y_pred = self.model.predict(x_test)
        if y_pred.shape[1] == 1:
            y_pred_classes = np.round(y_pred).flatten()
            y_test_classes = y_test.flatten()
        else:
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_test_classes = np.argmax(y_test, axis=1)
        report = classification_report(y_test_classes, y_pred_classes, target_names=target_names)
        print(report)

    def draw_plot(self, plot_type="accuracy"):
        """
        Draw a plot to visualize the training process with either accuracy, loss, or AUC.

        :param plot_type: The type of plot to draw, choose from 'accuracy', 'loss', or 'auc'
        :type plot_type: str, optional
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

    def save_model(self, filepath):
        """
        Save the model to the specified file path.

        :param filepath: The file path to save the model
        :type filepath: str
        """
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(filepath, units=64, classes=2):
        """
        Load a saved model from the specified file path and create an instance of a hybrid model.

        :param filepath: The file path to load the model from
        :type filepath: str

        :param units: Number of units for convolutional and GRU layers
        :type units: int, optional

        :param classes: Number of classes for classification
        :type classes: int, optional

        :return: An instance of the loaded hybrid model
        :rtype: hybrid_variation
        """
        try:
            loaded_model = keras.models.load_model(filepath)
            hybrid_model_instance = hybrid_variation(input_shape=loaded_model.input_shape[1:], units=units,
                                                     classes=classes)
            hybrid_model_instance.model = loaded_model
            return hybrid_model_instance
        except:
            print("The provided parameters does not match loaded model, please check.")