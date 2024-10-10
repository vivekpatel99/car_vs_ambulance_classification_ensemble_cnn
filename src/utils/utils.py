
import numpy as np

import pathlib
import opendatasets as od
import tensorflow as tf
from ensure import ensure_annotations
from box.exceptions import BoxValueError
from box import ConfigBox
import logging
import yaml
import pathlib
from utils.logs import get_logger
import inspect

logger = get_logger(__name__, log_level=logging.INFO)


def get_function_name() -> str:
    """_summary_

    Returns:
        str: _description_
    """
    return inspect.currentframe().f_back.f_code.co_name


@ensure_annotations
def read_yaml(yaml_path: str) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """

    try:
        with open(file=yaml_path, mode='r', encoding='UTF-8') as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info("yaml file: %s loaded successfully", yaml_file)

            return ConfigBox(content)
    except BoxValueError as exc:
        raise ValueError(f"yaml file is empty {exc}") from exc
    except Exception as e:
        raise e


# @ensure_annotations
def download_datasets_from_kaggle(config_path: str) -> None:
    """_summary_

    Args:
        dataset_url (str): data set url 
        dataset_dir (str): dataset download diractory 
    """
    config = read_yaml(yaml_path=config_path)

    logger.info(f'Raw dataset url: {config.paths.raw_dataset_dir}')

    # Dowload the dataset
    data_dir = pathlib.Path(config.paths.raw_dataset_dir)
    data_dir.mkdir(exist_ok=True)

    od.download(dataset_id_or_url=config.paths.dataset_url,
                data_dir=config.paths.raw_dataset_dir)

    if __name__ == '__main__':
        read_yaml(yaml_path='../../config/config.yml')


def convert_to_labels(dataset: tf.data.Dataset) -> list:
    labels_list = []
    for _, labels in dataset:
        labels_list.append(labels.numpy())

    # Convert to a flat numpy array
    return np.concatenate(labels_list)


def clear_tf_session() -> None:
    tf.keras.backend.clear_session()


def limit_gpu_memory_growth() -> None:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def gpu_clean_up() -> None:
    clear_tf_session()
    limit_gpu_memory_growth()
