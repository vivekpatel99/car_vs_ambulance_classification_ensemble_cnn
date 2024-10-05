
import logging
import tensorflow as tf
from train.train import train_model
from utils.image_data_loader import ImageDataLoader
from utils.logs import get_logger
from utils import utils

import argparse

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

logger = logging.getLogger(__name__)


def train(params_yaml: str = 'params.yaml') -> None:
    """
    Train the specified models according to the parameters in the
    configuration file.

    Parameters
    ----------
    params_yaml : str, optional
        The path to the configuration file, by default 'params.yaml'
    """

    params_config = utils.read_yaml(yaml_path=params_yaml)

    image_loader = ImageDataLoader(params_yaml=params_yaml)
    # Load the dataset with prefetching
    train_ds, val_ds = image_loader.load_image_dataset(prefetch=False)

    # Train the model and get the trained model
    models = train_model(config=params_config,
                         train_ds=train_ds, val_ds=val_ds)

    # Save the trained model
    for model_name, model in models.items():
        model_path = f'{params_config.paths.trained_model_path}/{model_name}.keras'
        model._model.save(model_path)
        logger.info(f'saved model: {model_name}')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', default='params.yaml')
    args = args_parser.parse_args()
    train(params_yaml=args.config)
