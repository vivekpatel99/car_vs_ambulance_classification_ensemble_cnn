
import logging
import pathlib
import keras_tuner as kt
from ..models.mini_vgg_net import MiniVGGNet
from ..models.lenet import LeNet
from ..models.shallownet import ShallowNet
from utils import utils
from utils.logs import get_logger
from tensorflow.keras.optimizers import SGD, Adam
models_dict = {
    'mini_vgg_net': MiniVGGNet,
    'lenet': LeNet,
    'shallownet': ShallowNet}

optimizers_dict = {
    'adam': Adam,
    'sgd': SGD
}


def train_model(train_data, val_data, params_yaml: str) -> None:
    log = get_logger(__name__, log_level=logging.INFO)
    config = pathlib.Path(params_yaml)

    config = utils.read_yaml(yaml_path=config)

    if config.train.if_fine_tune:
        for model in config.train.fine_tune_args.models:
            optimizer = optimizers_dict[config.trin.fine_tune.optimizer]
            _model = models_dict[config.train.fine_tune.models]
            log.info(f'Building model {_model} with optimizer {optimizer}.')
            _model = _model()
            # _model.build()
            # history = _model.train(train_data=train_data,
            #                        val_data=val_data,
            #                        optimizer=optimizer(),
            #                        verbose=1)
