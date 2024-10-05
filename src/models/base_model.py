from abc import ABC,  abstractmethod
from git import Object
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
        self.learning_rate = None
        self.optimizer = None

    @abstractmethod
    def build(self) -> Sequential:
        """Abstract method to build the model architecture."""
        pass

    def get_summary(self):  # -> Any:
        """Get the summary of the model architecture."""
        return self._model.summary() if self._model else None

    def model_compile(self) -> Object:
        """Compile the model with the provided optimizer."""
        return self._model.compile(loss=self.loss,
                                   optimizer=self.optimizer,
                                   metrics=self.metrics)

    # @ensure_annotations

    def train(self, train_data: tf.data.Dataset,
              val_data: tf.data.Dataset,
              verbose: int = 1, ) -> dict:
        """Train the model on the provided data."""
        if not self._model:
            raise ValueError(
                "Model has not been built. Call `build` method first.")

        self.model_compile()

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
