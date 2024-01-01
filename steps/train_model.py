import logging
import pandas as pd
from sklearn.base import RegressorMixin
import mlflow

from zenml import step
from zenml.client import Client

from src.model_dev import LinearRegressionModel
from .config import ModelNameConfig

experiment_tracker1 = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker1.name)
def train_model(
    X_train: pd.DataFrame,
    Y_train: pd.Series,
    config: ModelNameConfig,
) -> RegressorMixin:
    """
    Trains the model on the ingested data

    Args:
        X_train: pd.DataFrame
        X_test: pd.DataFrame
        Y_train: pd.DataFrame
        Y_test:pd.DataFrame

    Returns:
        model: RegressorMixin
    """

    if config.model_name == "LinearRegression":
        mlflow.sklearn.autolog()
        model = LinearRegressionModel()
        trained_model = model.train(X_train, Y_train)
        logging.info("Model training completed")
        
        return trained_model

    else:
        logging.error(f"Model {config.model_name} is not supported")
