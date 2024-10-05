import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dense, Input
from models.base_model import BaseModel
import logging
from utils import utils
from utils.logs import get_logger


class MiniVGGNet(BaseModel):
    def __init__(self,
                 optimizer_name: str,
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
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate

    def build(self) -> Sequential:
        """Build the MiniVGGNet model architecture."""

        _height = self.image_size
        _width = self.image_size
        _depth = 3
        _dim = -1

        # initialize the model
        self._model = Sequential(
            [
                Input(shape=(_height, _width, _depth)),
                # first CONV => RELU => CONV => RELU => POOL layer set
                Conv2D(32, (3, 3), padding="same", activation='relu'),
                BatchNormalization(axis=_dim),

                Conv2D(32, (3, 3), padding="same", activation='relu'),
                BatchNormalization(axis=_dim),

                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                # second CONV => RELU => CONV => RELU => POOL layer set
                Conv2D(64, (3, 3), padding="same", activation='relu'),
                BatchNormalization(axis=_dim),

                Conv2D(64, (3, 3), padding="same", activation='relu'),
                BatchNormalization(axis=_dim),

                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),

                # first (and only) set of FC => RELU layers
                Flatten(),
                Dense(512, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),

                # softmax classifier
                Dense(1, activation='sigmoid')
            ]
        )
        self.log.info('MiniVGGNet model built successfully.')
        return self._model
