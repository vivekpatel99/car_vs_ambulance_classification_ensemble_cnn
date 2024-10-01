from utils.logs import get_logger
import logging


class EnsembleStrategy:
    def __init__(self, strategy: str) -> None:
        self.strategy = strategy
        self.log = get_logger(__name__, log_level=logging.INFO)

    def ensemble_predict(self, data):
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
    def ensemble_predict(self, data):
        """
        Ensemble prediction method for voting strategy.

        :param data: a list of prediction probabilities
        :return: the ensemble prediction
        """
        return super().ensemble_predict(data)


class WeightedVotingStrategy(EnsembleStrategy):
    def ensemble_predict(self, data):
        """
        Ensemble prediction method for weighted voting strategy.

        :param data: a list of prediction probabilities
        :return: the ensemble prediction
        """
        return super().ensemble_predict(data)
