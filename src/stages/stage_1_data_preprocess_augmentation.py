"""_summary_
"""

import logging
from utils.logs import get_logger
from utils import utils
from box import ConfigBox
from tqdm import tqdm
import glob
import cv2
from tensorflow.keras.layers import Rescaling, RandomFlip, RandomRotation
from sklearn.calibration import LabelEncoder
import numpy as np
import os
import pathlib
import argparse
from sklearn.model_selection import train_test_split
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

logger = logging.getLogger(__name__)


def load_and_resize_data(params_config: ConfigBox) -> tuple[np.ndarray, np.ndarray]:
    """_summary_

    Args:
        params_config (str): _description_

    Returns:
        tuple[np.ndarray, np.ndarray]: _description_
    """

    images = []
    labels = []

    train_dir = params_config.paths.train_dir
    lables_dir = params_config.paths.labels_dir
    image_size = params_config.train.image_size

    for _image_path in tqdm(glob.glob(train_dir+"/*.jpg")):
        _img = cv2.imread(filename=_image_path)

        # because reduce memory consumption while loading images
        _img = cv2.resize(_img, (image_size, image_size))
        # print(_image_path)
        file_name_without_ext = _image_path.split('/')[-1].split('.')[0]
        label_file = f'{lables_dir}/{file_name_without_ext}.txt'
        # print(label_file)
        with open(label_file, 'r') as _labels:
            for line in _labels.readlines():
                _label, x_min, y_min, x_max, y_max = line.split()
                labels.append(_label)
                images.append(_img)

    images_arr = np.array(images)
    labels_arr = np.array(labels)
    logger.info('%s compeleted', utils.get_function_name())

    return images_arr, labels_arr


def save_data(config: ConfigBox, train_dataset_prefetch, label_names, train=True) -> None:
    """_summary_

    Args:
        config (ConfigBox): _description_
        train_dataset_prefetch (_type_): _description_
        label_names (_type_): _description_
    """

    if train:
        dir_name = 'train'
    else:
        dir_name = 'test'

    for cnt, (image, label) in tqdm(enumerate(train_dataset_prefetch)):
        # for cnt, (b_img, b_label) in enumerate(zip(images, labels)):
        # print()
        dir_path = os.path.join(config.paths.preprocess_dataset,
                                dir_name,
                                label_names[label])

        os.makedirs(dir_path, exist_ok=True)

        image_path = os.path.join(
            dir_path, f'{label_names[label].lower()}_{cnt}.png')

        tf.keras.preprocessing.image.save_img(image_path, image.numpy())

        logger.info('saved image %s', cnt)

    logger.info('%s compeleted', utils.get_function_name())


def data_preprocess_augmentation(config_path: str) -> None:
    """_summary_

    Args:
        config_path (str): _description_
    """

    config = utils.read_yaml(yaml_path=config_path)

    images_arr, labels_arr = load_and_resize_data(config)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels_arr)
    label_names = list(label_encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(images_arr,
                                                        encoded_labels,
                                                        test_size=config.data_split.test_size,
                                                        shuffle=True,
                                                        random_state=config.base.random_seed)

    #  Convert the split data into TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1000)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    # Prefetch to optimize performance
    train_dataset_prefetch = train_dataset.prefetch(
        buffer_size=tf.data.AUTOTUNE)
    val_dataset_prefech = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    # TODO: move to config file when move to hydra
    rescale = tf.keras.Sequential([
        Rescaling(1./255)
    ])

    data_augmentation = tf.keras.Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.2),
    ])

    # reference https://www.datacamp.com/tutorial/complete-guide-data-augmentation
    train_dataset_prefetch = train_dataset_prefetch.map(
        lambda x, y: (rescale(x, training=True), y))
    train_dataset_prefetch = train_dataset_prefetch.map(
        lambda x, y: (data_augmentation(x, training=True), y))

    val_dataset_prefech = val_dataset_prefech.map(
        lambda x, y: (rescale(x, training=True), y))

    save_data(config, train_dataset_prefetch, label_names)
    logger.info('%s - train dataset saved', utils.get_function_name())
    save_data(config, val_dataset_prefech, label_names, train=False)
    logger.info('%s test dataset saved', utils.get_function_name())

    logger.info('%s compeleted', utils.get_function_name())


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', default='params.yaml')
    args = args_parser.parse_args()

    data_preprocess_augmentation(config_path=args.config)
    # data_preprocess_augmentation(config_path='params.yaml')

    """
    dvc stage add -n data_preprocess_augmentation\
            -d src/stages/stage_01_data_preprocess_augmentation.py \\
          /usr/bin/python src/stages/stage_01_data_preprocess_augmentation.py --config=params.yaml 
    """
