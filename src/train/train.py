
import logging
from pathlib import Path


from box import ConfigBox
from git import Object
import keras_tuner as kt
import tensorflow as tf
from models.mini_vgg_net import MiniVGGNet
from models.lenet import LeNet
from models.shallownet import ShallowNet
from utils import utils
import json
from utils.logs import get_logger


models_dict = {
    'mini_vgg_net': MiniVGGNet,
    'lenet': LeNet,
    'shallownet': ShallowNet}


def train_model(config: ConfigBox, train_ds, val_ds) -> dict[str, Object]:
    log = get_logger(__name__, log_level=logging.INFO)
    trained_model_dict = {}
    tune_params_dict = {}
    if config.train.if_fine_tune:
        for _model_name in config.train.fine_tune_args.models:
            utils.gpu_clean_up()

            def build_model(hp):
                """
                Hyperparameter tuner function for fine-tuning models.

                Parameters
                ----------
                hp : keras_tuner.HyperParameters
                    The hyperparameters to be tuned.

                Returns
                -------
                model.compile() : tensorflow.keras.Model
                    The compiled model with the specified hyperparameters.
                """
                optimizer_name = hp.Choice(
                    'optimizer', list(config.train.fine_tune_args.optimizers))
                learning_rate = hp.Float('learning_rate',
                                         min_value=config.train.fine_tune_args.lr.min,
                                         max_value=config.train.fine_tune_args.lr.max,
                                         sampling='log')

                log.info(f'optimizer: {optimizer_name}, '
                         f'learning rate: {learning_rate}, '
                         f'loss: {config.train.loss}, '
                         f'metrics: {config.evaluate.metrics}')

                if optimizer_name is None:
                    raise ValueError(
                        f'optimizer name is None: {optimizer_name}')

                if learning_rate is None:
                    raise ValueError(f'learning rate is None: {learning_rate}')

                if config.train.loss is None:
                    raise ValueError(f'loss is None: {config.train.loss}')

                if not config.evaluate.metrics:
                    raise ValueError(
                        f'metrics is empty: {config.evaluate.metrics}')

                _model = models_dict[_model_name](
                    optimizer_name=optimizer_name,
                    learning_rate=learning_rate,
                    loss=config.train.loss,
                    metrics=list(
                        config.evaluate.metrics),
                    image_size=config.train.image_size)

                _model.build()

                _model.model_compile()

                # Return the compiled model instance (self._model)
                return _model._model

            stop_early = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5)

            tuner = kt.RandomSearch(
                build_model,
                objective=kt.Objective('val_accuracy', direction='max'),
                seed=config.base.random_seed,
                max_trials=3,
                directory='tunning_dir',
                project_name='ensemble_tunning',
                overwrite=True

            )
            tuner.search(
                train_ds,
                epochs=config.train.fine_tune_args.epochs,
                validation_data=val_ds,
                callbacks=[stop_early])

            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

            best_optimizer_name = best_hps.get('optimizer')
            best_learning_rate = best_hps.get('learning_rate')
            log.info(f"Best hyperparameters: learning_rate: {best_learning_rate}, \
                     optimizer: {best_optimizer_name}")
            tune_params_dict[_model_name] = {
                'optimizer': best_optimizer_name,
                'learning_rate': best_learning_rate
            }
            _model = models_dict[_model_name](
                optimizer_name=best_optimizer_name,
                learning_rate=best_learning_rate,
                loss=config.train.loss,
                metrics=list(
                    config.evaluate.metrics),
                image_size=config.train.image_size)

            _model.build()

            _model.model_compile()
            _model.train(train_ds, val_ds)
            trained_model_dict[_model_name] = _model

    dump_hyper_params_to_json(
        config, log, tune_params_dict)
    log.info('Dumped hyperparameters to JSON file.')

    return trained_model_dict


def dump_hyper_params_to_json(config, log, tune_params_dict) -> None:
    """
    Dumps the best hyperparameters to a JSON file.

    Parameters
    ----------
    config : ConfigBox
        The configuration object.
    log : logging.Logger
        The logger object.
    tune_params_dict : dict
        The dictionary containing the hyperparameters for each model.
    _model_name : str
        The name of the model.
    best_optimizer_name : str
        The name of the best optimizer.
    best_learning_rate : float
        The best learning rate.
    """
    try:
        with open(config.paths.tune_hyperparameters_path, "w") as fp:
            json.dump(obj=tune_params_dict, fp=fp)

    except IOError as e:
        log.error(f'Error writing to file: {e}')
