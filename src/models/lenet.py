import numpy as np
import tensorflow as tf

from utils import utils
from models.base_model import BaseModel
import logging
from utils.logs import get_logger
# from ensure import ensure_annotations


class LeNet(BaseModel):

    def __init__(self,
                 optimizer_name: str,
                 image_size: int,
                 loss: str,
                 batch_size: int = 32,
                 epochs: int = 10,
                 learning_rate: float = 1e-3,
                 metrics: list = [],
                 ) -> None:
        """
        Initialize the LeNet model.

        Parameters
        ----------
        optimizer_name : str
            The name of the optimizer to use.
        image_size : int
            The size of the input images.
        loss : str
            The loss function to use.
        batch_size : int, optional
            The batch size to use during training. Defaults to 32.
        epochs : int, optional
            The number of epochs to train for. Defaults to 10.
        learning_rate : float, optional
            The learning rate to use. Defaults to 1e-3.
        metrics : list, optional
            A list of metrics to use during training. Defaults to [].

        Returns
        -------
        None
        """
        super().__init__()
        self.log = get_logger(__name__, log_level=logging.INFO)
        self.image_size = image_size
        self.loss = loss
        self.batch_size = batch_size
        self.epochs = epochs
        self.metrics = metrics
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate

    def build(self) -> tf.keras.Sequential:
        """Build the LeNet model architecture.

        Returns
        -------
        model : tf.keras.Sequential
            The built LeNet model.
        """
        _height: int = self.image_size
        _width: int = self.image_size
        _depth: int = 3

        # initialize the model
        self._model = tf.keras.Sequential([
            tf.keras.Input(shape=(_height, _width, _depth)),
            # first set of CONV => RELU => POOL layers
            tf.keras.layers.Conv2D(
                20, (5, 5), padding='same',  activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            tf.keras.layers.Dropout(0.25),

            # second set of CONV => RELU => POOL layers
            tf.keras.layers.Conv2D(
                50, (5, 5), padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(500, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])

        self.log.info('LeNet model built successfully.')
        return self._model
