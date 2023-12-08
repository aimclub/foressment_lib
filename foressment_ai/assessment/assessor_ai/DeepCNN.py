import keras
from keras import layers, models
from keras.metrics import Recall, Precision, AUC
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np

class DeepCNN:
    """
    build_model - define model structure including blocks and output units;
    Draw plot fear lies the training process with either accuracy or loss.
    """

    def __init__(self, input_shape=None, blocks=3, units=64, classes=2, classifier_activation="sigmoid"):
        self.model = self.build_model(input_shape, blocks, units, classes, classifier_activation)

    def build_model(self, input_shape, blocks, units, classes, classifier_activation):
        model = models.Sequential()  # Use Sequential model
        model.add(layers.Input(shape=input_shape))

        # Define the configuration dictionary
        configuration = {
            "conv_activation": "relu",
            "pooling_type": "max",
            "use_bias": False,
            "batch_norm": True,
            "kernel_size": 3,
            "padding": "same"
        }

        for _ in range(blocks):
            # Add convolutional layers based on configuration
            model.add(layers.Conv1D(units, configuration["kernel_size"], padding=configuration["padding"],
                                    use_bias=configuration["use_bias"]))
            if configuration["batch_norm"]:
                model.add(layers.BatchNormalization())
            model.add(layers.Activation(configuration["conv_activation"]))
            model.add(layers.Conv1D(units, configuration["kernel_size"], padding=configuration["padding"],
                                    use_bias=configuration["use_bias"]))
            if configuration["batch_norm"]:
                model.add(layers.BatchNormalization())

            # Add pooling layer based on configuration
            if configuration["pooling_type"] == "max":
                model.add(layers.MaxPooling1D(2, padding=configuration["padding"]))
            elif configuration["pooling_type"] == "average":
                model.add(layers.AveragePooling1D(2, padding=configuration["padding"]))

        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dense(128, activation="relu"))
        if classes > 2:
            model.add(layers.Dense(classes, activation=classifier_activation, name="predictions"))
            loss = 'categorical_crossentropy'
        else:
            model.add(layers.Dense(1, activation=classifier_activation, name="predictions"))
            loss = 'binary_crossentropy'

        metrics = ['accuracy', Recall(), Precision(), AUC(name='auc')]
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=loss, metrics=metrics)

        return model

    def fit(self, X_train, y_train, validation_data, epochs=10, batch_size=128, verbose=0):
        """
        if len(X_train.shape) == 2:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            validation_data[0] = validation_data[0].reshape(validation_data[0].shape[0], validation_data[0].shape[1], 1)
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

        :param X_test: The input test data
        :type X_test: array-like

        :param y_test: The target test data
        :type y_test: array-like

        :param target_names: Optional target class names
        :type target_names: list, optional
        """
        y_pred = self.model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        # Generate classification report
        report = classification_report(y_test_classes, y_pred_classes, target_names=target_names)
        print(report)

    def draw_plot(self, plot_type="accuracy"):
        """
        Draw a plot to visualize the training process with either accuracy or loss.

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
        Load a saved model from the specified file path and create an instance of Hybrid_CNN_GRU.

        :param filepath: The file path to load the model from
        :type filepath: str

        :param units: Number of units for convolutional and GRU layers
        :type units: int, optional

        :param classes: Number of classes for classification
        :type classes: int, optional
        """
        try:
            loaded_model = keras.models.load_model(filepath)
            hybrid_model_instance = DeepCNN(input_shape=loaded_model.input_shape[1:], units=units, classes=classes)
            hybrid_model_instance.model = loaded_model
            return hybrid_model_instance
        except:
            print("The provided parameters does not match loaded model, please check.")