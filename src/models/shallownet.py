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
