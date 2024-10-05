import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dense, Input
from utils import utils
from models.base_model import BaseModel
import logging
from utils.logs import get_logger
# from ensure import ensure_annotations


class LeNet(BaseModel):

    def __init__(self,
                 optimizer: str,
                 image_size: int,
                 loss: str,
                 batch_size: int = 32,
                 epochs: int = 10,
                 learning_rate: float = 1e-3,
                 metrics: list = [],
                 ) -> None:
        super().__init__()
        self.log = get_logger(__name__, log_level=logging.INFO)
        self.image_size = image_size
        self.loss = loss
        self.batch_size = batch_size
        self.epochs = epochs
        self.metrics = metrics
        self.optimizer = optimizer
        self.learning_rate = learning_rate

    def build(self) -> Sequential:
        """Build the LeNet model architecture."""

        _height = self.image_size
        _width = self.image_size
        _depth = 3

        # initialize the model
        self._model = Sequential([
            Input(shape=(_height, _width, _depth)),
            # first set of CONV => RELU => POOL layers
            Conv2D(20, (5, 5), padding='same',  activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Dropout(0.25),

            # second set of CONV => RELU => POOL layers
            Conv2D(50, (5, 5), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Dropout(0.25),

            Flatten(),
            Dropout(0.5),
            Dense(500, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid'),
        ])

        self.log.info('LeNet model built successfully.')
        return self._model
