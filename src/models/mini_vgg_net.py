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

logger = get_logger(__name__, log_level=logging.INFO)


class MiniVGGNet(BaseModel):
    def __init__(self, params_yaml: str) -> None:
        super().__init__()
        self.params = utils.read_yaml(yaml_path=params_yaml)
        self._model = None
        self.model_params = self.params.train.mini_vgg_net
        self.history = None
        self.logger = get_logger(__name__, log_level=logging.INFO)
        self.image_size = self.params.train.image_size
        self.loss = self.params.train.binary_crossentropy
        self.batch_size = self.params.train.batch_size
        self.epochs = self.model_params.epochs
        self.metrics = self.params.evaluate.metrics

    def build(self) -> Sequential:
        _height = self.params.train.image_size
        _width = self.params.train.image_size
        _depth = 3
        _dim = -1
        # initialize the model
        self._model = Sequential([
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

        ])
        logger.info('model is built')
        return self._model

    # def get_summary(self):
    #     return self._model.summary()

    # def train(self,
    #           train_data: tf.data.Dataset,
    #           val_data: tf.data.Dataset,
    #           optimizer: tf.keras.Optimizer,
    #           verbose: int = 1) -> dict:

    #     self._model.compile(loss=self.params.train.binary_crossentropy,
    #                         optimizer=optimizer,
    #                         metrics=self.params.evaluate.metrics)

    #     history = self._model.fit(train_data,
    #                               validation_data=val_data,
    #                               batch_size=self.params.train.batch_size,
    #                               epochs=self.model_params.epochs,
    #                               verbose=verbose)
    #     self.history = history.history
    #     logger.info('trained')
    #     return history

    # # @ensure_annotations
    # def evaluate(self, test_data: np.ndarray) -> np.ndarray:
    #     """
    #     """
    #     assert test_data.ndim == 4,  "Input array/image must be 4-dimensional"

    #     predictions = self._model.predict(test_data,
    #                                       batch_size=self.batch_size)
    #     y_pred = (predictions > 0.5).astype(int)
    #     return y_pred
