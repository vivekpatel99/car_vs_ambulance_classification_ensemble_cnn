
import logging

from train.train import train_model
from utils.image_data_loader import ImageDataLoader
from utils.logs import get_logger
from utils import utils

import argparse

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

logger = logging.getLogger(__name__)


def train(params_yaml: str = 'params.yaml') -> None:

    params_config = utils.utils.read_yaml(yaml_path=params_yaml)

    image_loader = ImageDataLoader(params_yaml=params_yaml)
    train_ds, val_ds = image_loader.load_image_dataset()

    models = train_model(config=params_config,
                         train_ds=train_ds, val_ds=val_ds)

    logger.info(f'Best')


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yaml')
    parsed_args = args.parse_args()
    train(params_yaml=parsed_args.config)
