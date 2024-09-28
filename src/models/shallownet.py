from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dense, Input
from models.base_model import BaseModel
import logging
from utils import utils
from utils.logs import get_logger


class ShallowNet(BaseModel):
    def __init__(self, params_yaml: str) -> None:
        super().__init__()
        self.params = utils.read_yaml(yaml_path=params_yaml)
        self.model_params = self.params.train.shallownet
        self.log = get_logger(__name__, log_level=logging.INFO)
        self.image_size = self.params.train.image_size
        self.loss = self.params.train.binary_crossentropy
        self.batch_size = self.params.train.batch_size
        self.epochs = self.model_params.epochs
        self.metrics = self.params.evaluate.metrics

    def build(self) -> Sequential:
        """Build the ShallowNet model architecture."""

        _height = self.image_size
        _width = self.image_size
        _depth = 3

        # define the first and only CONV => RELU layer
        self._model = Sequential(
            [
                Input(shape=(_height, _width, _depth)),
                Conv2D(32, (3, 3),
                       padding="same",  activation='relu'),
                Flatten(),
                Dense(1, activation='sigmoid'),
            ]
        )
        self.log.info('ShallowNet model built successfully.')
        return self._model
