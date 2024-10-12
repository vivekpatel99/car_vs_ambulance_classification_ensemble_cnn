from utils import utils
from tensorflow.keras.utils import image_dataset_from_directory
import tensorflow as tf
from ensure import ensure_annotations
import logging
from utils.logs import get_logger


class ImageDataLoader:

    def __init__(self, params_yaml: str) -> None:
        params_config = utils.read_yaml(yaml_path=params_yaml)
        self.train_cfg = params_config.train
        self.random_seed = params_config.base.random_seed
        self.train_ds = None
        self.val_ds = None
        self.train_dir = params_config.paths.preprocessed_train
        self.val_dir = params_config.paths.preprocessed_val
        self.log = get_logger(__name__, log_level=logging.INFO)

    @ensure_annotations
    def load_image_dataset(self, prefetch: bool = True) -> tuple:
        """_summary_
        This function is loading and prefetching the images.
        Args:
            tf (_type_): _description_

        Returns:
            _type_: _description_
        """

        self.train_ds = image_dataset_from_directory(self.train_dir,
                                                     seed=self.random_seed,
                                                     image_size=(self.train_cfg.image_size,
                                                                 self.train_cfg.image_size),
                                                     batch_size=self.train_cfg.batch_size)

        self.val_ds = image_dataset_from_directory(self.val_dir,
                                                   seed=self.random_seed,
                                                   image_size=(self.train_cfg.image_size,
                                                               self.train_cfg.image_size),
                                                   batch_size=self.train_cfg.batch_size)
        if prefetch:
            # https://www.tensorflow.org/tutorials/images/classification
            AUTOTUNE = tf.data.AUTOTUNE
            self.train_ds = self.train_ds.cache()\
                .shuffle(1000).prefetch(buffer_size=AUTOTUNE)

            self.val_ds = self.val_ds.cache()\
                .prefetch(buffer_size=AUTOTUNE)
            self.log.info('load_image_dataset are prefetched')

        self.log.info('load_image_dataset completed')

        return (self.train_ds, self.val_ds)
