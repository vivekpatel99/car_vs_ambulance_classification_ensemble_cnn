
import logging


from box import ConfigBox
import keras_tuner as kt
from models.mini_vgg_net import MiniVGGNet
from models.lenet import LeNet
from models.shallownet import ShallowNet

from utils.logs import get_logger
from tensorflow.keras.optimizers import SGD, Adam
import keras_tuner as kt

models_dict = {
    'mini_vgg_net': MiniVGGNet,
    'lenet': LeNet,
    'shallownet': ShallowNet}
optimizers_dict = {
    'adam': Adam,
    'sgd': SGD
}


def train_model(config: ConfigBox, train_ds, val_ds) -> None:
    log = get_logger(__name__, log_level=logging.INFO)

    if config.train.if_fine_tune:

        for _model in config.train.fine_tune_args.models:
            def build_model(hp):
                optimizer = hp.Choice(
                    'optimizer', config.trin.fine_tune.optimizers)
                learning_rate = hp.Choice('learning_rate',
                                          min_value=config.train.fine_tune.lr.min,
                                          max_value=config.train.fine_tune_lr.max)
                return _model(optimizer=optimizer,
                              learning_rate=learning_rate,
                              loss=config.train.fine_tune_args.loss,
                              metrics=config.train.fine_tune_args.metrics,
                              image_size=config.train.image_size).build().train()

            tuner = kt.RandomSearch(
                build_model,
                objective=kt.Objective('val_accuracy', 'max'),
                max_trials=config.train.fine_tune_args.max_trials,
                directory='tunning_dir',
                project_name='ensemble_tunning'
            )
            tuner.search(
                train_ds,
                epochs=config.train.fine_tune_args.epochs,
                validation_data=val_ds)

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(best_hps.get_config())

        # _model.build()
        # history = _model.train(train_data=train_data,
        #                        val_data=val_data,
        #                        optimizer=optimizer(),
        #                        verbose=1)
