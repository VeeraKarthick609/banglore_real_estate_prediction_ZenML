import logging
from abc import ABC, abstractmethod

from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

class EvaluationStrategy(ABC):
    """
    Evaluation strategy for model evaluations
    """

    @abstractmethod
    def calcuate_scores(self, Y_pred: np.ndarray, Y_truth: np.ndarray) -> None:
        """
        Method to calculate score

        Args:
            Y_pred: np.ndarray
            Y_truth: np.ndarray
        
        Returns:
            None
        """
        pass


class MSE(EvaluationStrategy):
    """
    Class to calculate MSE
    """

    def calcuate_scores(self, Y_pred: np.ndarray, Y_truth: np.ndarray) -> float :
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_pred=Y_pred, y_true= Y_truth, squared= True)
            logging.info(f"MSE value: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error while calculating MSE: {e}")
            raise e
        

class RMSE(EvaluationStrategy):
    """
    Class to calculate RMSE
    """

    def calcuate_scores(self, Y_pred: np.ndarray, Y_truth: np.ndarray) -> None:
        try:
            logging.info("Calculating RMSE")
            rmse = mean_squared_error(y_pred=Y_pred, y_true= Y_truth, squared=False)
            logging.info(f"RMSE value: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error while calculating RMSE: {e}")
            raise e
        
class R2_SCORE(EvaluationStrategy):
    """
    Class to calculate r2_score
    """

    def calcuate_scores(self, Y_pred: np.ndarray, Y_truth: np.ndarray) -> None:
        try:
            logging.info("Calculating r2_score")
            r2score = r2_score(y_pred=Y_pred, y_true= Y_truth)
            logging.info(f"r2_score value: {r2score}")
            return r2score
        except Exception as e:
            logging.error(f"Error while calculating r2_score: {e}")
            raise e