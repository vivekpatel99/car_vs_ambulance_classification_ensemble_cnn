from ast import Str
import logging
from turtle import st
from ..utils.image_data_loader import ImageDataLoader
from utils.logs import get_logger
from utils import utils
from box import ConfigBox
from tqdm import tqdm
import glob
import cv2
from tensorflow.keras.optimizers import SGD
from sklearn.calibration import LabelEncoder
import numpy as np
import os
import pathlib
import argparse
from sklearn.model_selection import train_test_split
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

logger = logging.getLogger(__name__)


def train(params_yaml: str = 'params.yaml') -> None:

    params_config = utils.utils.read_yaml(yaml_path=params_yaml)

    image_loader = ImageDataLoader(params_yaml=params_yaml)
    train_ds, val_ds = image_loader.load_image_dataset()

    lenet_model_opt = SGD(learning_rate=1e-2,
                          weight_decay=0.01 / 40,
                          momentum=0.9,
                          nesterov=True)
    history = lenet_model.train(train_data=train_ds,
                                val_data=val_ds,
                                optimizer=lenet_model_opt)
