from abc import ABC,  abstractmethod
from git import Object
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from utils.logs import get_logger
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers

# Template Method Pattern


class BaseModel(ABC):
    def __init__(self) -> None:
        """
        Initializes the BaseModel object.

        The BaseModel object is initialized with the following
        properties:
        - _model: the compiled model
        - _history: the history of the training process
        - log: the logger object
        - params: the parameters of the experiment
        - model_params: the parameters of the model
        - image_size: the size of the input images
        - loss: the loss function used during training
        - batch_size: the batch size used during training
        - epochs: the number of epochs used during training
        - metrics: the metrics used during training
        - learning_rate: the learning rate used during training
        - optimizer: the optimizer used during training
        - optimizer_name: the name of the optimizer used during training
        """
        super().__init__()
        self._model = None
        self._history = None
        self.log = get_logger(__name__)
        self.params = None
        self.model_params = None
        self.image_size = None
        self.loss = None
        self.batch_size = None
        self.epochs = None
        self.metrics = None
        self.learning_rate = None
        self.optimizer = None
        self.optimizer_name = None

    @abstractmethod
    def build(self) -> Sequential:
        """Abstract method to build the model architecture."""
        pass

    def get_summary(self) -> None:
        """Get the summary of the model architecture."""
        return self._model.summary() if self._model else None

    def get_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """
        Selects the optimizer based on the name passed to the model.
        Returns:
            optimizer object
        """
        self.log.info(f'Selecting optimizer {self.optimizer_name}')
        # Select the optimizer based on the name passed to the model
        optimizers_dict = {
            'adam': optimizers.Adam,
            'sgd': optimizers.SGD,
            'rmsprop': optimizers.RMSprop,
        }
        if self.optimizer_name is None:
            raise ValueError(f'Invalid optimizer name: {self.optimizer_name}')

        self.optimizer = optimizers_dict.get(self.optimizer_name)
        return self.optimizer(learning_rate=self.learning_rate)

    def model_compile(self) -> tf.keras.Model:
        """
        Compile the model with the provided optimizer.

        Returns:
            compiled model
        """
        self.optimizer = self.get_optimizer()
        return self._model.compile(loss=self.loss,
                                   optimizer=self.optimizer,
                                   metrics=self.metrics)

    def train(self, train_data: tf.data.Dataset,
              val_data: tf.data.Dataset,
              verbose: int = 1) -> dict:
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

    def model_evaluate(self, test_data: tf.data.Dataset) -> np.ndarray:
        """Evaluate the model on the provided test data."""
        if not self._model:
            raise ValueError(
                "Model has not been built. Call `build` method first.")

        predictions = self._model.predict(test_data,
                                          batch_size=self.batch_size)

        y_pred = (predictions > 0.5).astype(int)
        self.log.info('Evaluation completed')
        return y_pred
