import argparse
import glob
import json
import logging
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from utils.logs import get_logger
from utils import utils
from utils.image_data_loader import ImageDataLoader
from tensorflow import keras


def evaluate_models(params_yaml: str = 'params.yaml') -> None:
    """Evaluate the model on the provided test data."""
    log = get_logger(__name__, log_level=logging.INFO)
    metrics_dict = {}
    config = utils.read_yaml(yaml_path=params_yaml)
    image_loader = ImageDataLoader(params_yaml=params_yaml)
    _, test_ds = image_loader.load_image_dataset(prefetch=False)
    tuned_models = glob.glob(f'{config.paths.trained_model_path}/*.keras')

    if not tuned_models:
        raise FileNotFoundError(
            f'No models found in {config.paths.trained_model_path}')

    for cnt, model in enumerate(tuned_models):
        _model = keras.models.load_model(model)
        if not _model:
            raise ValueError(f'Failed to load model: {tuned_models[cnt]}')

        log.info(f'evaluating model: {tuned_models[cnt]}')

        # Lists to store predictions and true labels
        all_predictions = []
        all_labels = []

        # Assuming 'test_dataset' is your test dataset
        for batch_images, batch_labels in test_ds:
            # Get predictions for the current batch
            predictions = _model.predict(batch_images)
            predicted_classes = (predictions > 0.5).astype(int)

            # Extend the lists with the current batch's results
            all_predictions.extend(predicted_classes)
            all_labels.extend(batch_labels.numpy())

        report = classification_report(
            all_labels, all_predictions, target_names=test_ds.class_names)
        confusion_mat = confusion_matrix(all_labels, all_predictions)

        metrics_dict['classification_report'] = report
        metrics_dict['confusion_matrix'] = confusion_mat

        result_dump_path = f'{config.evaluate.reports_dir}/{tuned_models[cnt].split("/")[1].replace(".keras", "")}_metrics.json'

        # Create the directory if it doesn't exist
        # os.makedirs(os.path.dirname(result_dump_path), exist_ok=True)

        try:
            with open(result_dump_path, "w") as fp:
                json.dump(
                    obj=metrics_dict,
                    fp=fp
                )
        except IOError as e:
            raise IOError(
                f'Error writing to file: {e}') from e

        log.info(
            f'Classification report and confusion matrix saved for model: {tuned_models[cnt]}')
        log.info(f'Evaluation completed for model: {tuned_models[cnt]}')
        break


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', default='params.yaml')
    args = args_parser.parse_args()
    utils.gpu_clean_up()
    evaluate_models(params_yaml=args.config)
