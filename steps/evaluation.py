import logging
import pandas as pd
from sklearn.base import RegressorMixin
from typing_extensions import Tuple
from typing import Annotated
import mlflow

from zenml import step
from zenml.client import Client

from src.model_evaluation import R2_SCORE, MSE, RMSE

experiment_tracker_object = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker_object.name)
def evaluate_model(
    model: RegressorMixin, X_test: pd.DataFrame, Y_test: pd.Series
) -> Tuple[
    Annotated[float, "MSE"],
    Annotated[float, "RMSE"],
    Annotated[float, "R2_SCORE"],
]:
    """
    Evaluates the model

    Args:
        model: RegressorMixin
        X_test: pd.DataFrame
        Y_test: pd.Series
    """

    try:
        y_preiction = model.predict(X_test)

        ##MSE
        mse = MSE()
        mse_value = mse.calcuate_scores(Y_pred=y_preiction, Y_truth=Y_test)
        mlflow.log_metric("mse", mse_value)

        #RMSE
        rmse = RMSE()
        rmse_value = rmse.calcuate_scores(Y_pred=y_preiction, Y_truth=Y_test)
        mlflow.log_metric("rmse", rmse_value)

        #R2 Score
        r2_score = R2_SCORE()
        r2_score_value = r2_score.calcuate_scores(Y_pred=y_preiction, Y_truth=Y_test)
        mlflow.log_metric("r2_score", r2_score_value)

        return mse_value, rmse_value, r2_score_value

    except Exception as e:
        logging.error(f"Error while evaluating the model: {e}")
        raise e
