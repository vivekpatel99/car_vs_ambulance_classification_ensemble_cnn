import argparse
import glob
import json
import logging
import os
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
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
            all_labels, all_predictions, target_names=test_ds.class_names, output_dict=True)

        confusion_mat = confusion_matrix(
            y_true=all_labels,  y_pred=all_predictions)
        confusion_disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat,
                                                display_labels=test_ds.class_names)

        metrics_dict['classification_report'] = report
        metrics_dict['confusion_matrix'] = confusion_mat

        # Convert NumPy arrays to lists
        for key, value in metrics_dict.items():
            if isinstance(value, np.ndarray):
                metrics_dict[key] = value.tolist()

        result_dump_path = f'{config.evaluate.results_dir}/{tuned_models[cnt].split("/")[1].replace(".keras", "")}'

        metrics_json_path = f'{result_dump_path}_metrics.json'

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(metrics_json_path), exist_ok=True)

        confusion_disp.plot()
        plt.savefig(f'{result_dump_path}_confusion_matrix.png')

        log.info(f'Saving evaluation metrics to: {result_dump_path}')

        try:
            with open(metrics_json_path, "w") as fp:
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


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', default='params.yaml')
    args = args_parser.parse_args()
    utils.gpu_clean_up()
    evaluate_models(params_yaml=args.config)
