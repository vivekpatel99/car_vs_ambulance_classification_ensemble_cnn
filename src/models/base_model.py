from abc import ABC,  abstractmethod
from ast import Tuple
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.python.data.ops import NumpyIterator
# Template Method Pattern


class BaseModel(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.params = None
        self.model_params = None
        self._model = None
        self.history = None
        self.image_size = None
        self.log = None
        self.loss = None
        self.batch_size = None
        self.epochs = None
        self.metrics = None

    @abstractmethod
    def build(self) -> Sequential:
        """Abstract method to build the model architecture."""
        pass

    def get_summary(self):  # -> Any:
        """Get the summary of the model architecture."""
        return self._model.summary() if self._model else None

    # @ensure_annotations
    def train(self, train_data: tf.data.Dataset,
              val_data: tf.data.Dataset,
              optimizer: tf.keras.Optimizer,
              verbose: int = 1) -> dict:
        """Train the model on the provided data."""
        if not self._model:
            raise ValueError(
                "Model has not been built. Call `build` method first.")

        self._model.compile(loss=self.loss,
                            optimizer=optimizer,
                            metrics=self.metrics)

        history = self._model.fit(train_data,
                                  validation_data=val_data,
                                  batch_size=self.batch_size,
                                  epochs=self.epochs,
                                  verbose=verbose)
        self._history = history.history
        self.log.info('Training completed.')
        return self._history

    # @ensure_annotations
    def evaluate(self, test_data: NumpyIterator) -> np.ndarray:
        """Evaluate the model on the provided test data."""
        if not self._model:
            raise ValueError(
                "Model has not been built. Call `build` method first.")

        # assert test_data.ndim == 4,  "Input array/image must be 4-dimensional"

        predictions = self._model.predict(test_data,
                                          batch_size=self.batch_size)

        y_pred = (predictions > 0.5).astype(int)
        self.log.info('Evaluation completed')
        return y_pred

    # Plot accuracies
    def plot_accuracies(self):
        train_accuracy = self._history['accuracy']
        val_accuracy = self._history['val_accuracy']
        plt.plot(np.arange(0, 20), train_accuracy,
                 '-x', label='train_accuracy')
        plt.plot(np.arange(0, 20), val_accuracy, '-x', label='val_accuracy')
        plt.xlabel('epoch')
        plt.legend(['Train Accuracy', 'Validation Accuracy'])
        plt.ylabel('train_accuracy/val_accuracy')
        plt.title('Accuracy vs. No. of epochs')

    def plot_losses(self):
        train_losses = self._history['loss']
        val_losses = self._history['val_loss']
        plt.plot(np.arange(0, 20), train_losses, '-bx', label='train_losses')
        plt.plot(np.arange(0, 20), val_losses, '-rx', label='val_losses')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['Train Loss', 'Validation Loss'])
        plt.title('Loss vs. No. of epochs')
