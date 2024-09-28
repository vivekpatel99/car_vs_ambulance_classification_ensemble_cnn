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

    def __init__(self, params_yaml: str) -> None:
        super().__init__()

        self.params = utils.read_yaml(yaml_path=params_yaml)
        self.model_params = self.params.train.lenet
        self.log = get_logger(__name__, log_level=logging.INFO)
        self.image_size = self.params.train.image_size
        self.loss = self.params.train.binary_crossentropy
        self.batch_size = self.params.train.batch_size
        self.epochs = self.model_params.epochs
        self.metrics = self.params.evaluate.metrics

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
