from os import major
import numpy as np
from utils.logs import get_logger
import logging


class EnsembleStrategy:
    def __init__(self, strategy: str) -> None:
        self.strategy = strategy
        self.log = get_logger(__name__, log_level=logging.INFO)

    def ensemble_predict(self,  predictions: list):
        """
        Method to ensemble the predictions of each model in the ensemble.

        This method should be implemented in each strategy class.

        Parameters
        ----------
        data : list
            A list of prediction probabilities from each model.

        Returns
        -------
        ensemble_prediction : list
            The ensemble prediction.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the strategy class.
        """
        raise NotImplementedError(
            "Each strategy must implement the ensemble_predict method")


class VotingStrategy(EnsembleStrategy):
    def __init__(self) -> None:
        super().__init__('voting')

    def ensemble_predict(self, predictions: list) -> np.ndarray:
        """
        Ensemble prediction method for voting strategy.

        :param data: a list of prediction probabilities
        :return: the ensemble prediction
        """
        self.log.info('Using Voting strategy for ensemble predictions.')

        # Sum the predictions and take the majority vote
        preds = np.array(predictions)

        # Majority vote for binary classification
        majority_vote = np.mean(preds, axis=0)
        return (majority_vote > 0.5).astype(int)


class WeightedVotingStrategy(EnsembleStrategy):
    def __init__(self, weights: list) -> None:
        super().__init__('weighted_voting')
        self.weights = weights

    def ensemble_predict(self, predictions: list) -> np.ndarray:
        """
        Ensemble prediction method for weighted voting strategy.

        :param data: a list of prediction probabilities
        :return: the ensemble prediction
        """
        self.log.info(
            'Using Weighted Voting strategy for ensemble predictions.')

        # Convert to numpy array and apply weights
        preds = np.array(predictions)
        weighted_predictions = np.average(
            preds, axis=0, weights=self.weights)
        return (weighted_predictions > 0.5).astype(int)
