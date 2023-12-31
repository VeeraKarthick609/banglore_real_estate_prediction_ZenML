from abc import ABC, abstractmethod
import logging
import pandas as pd
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract class for all models
    """

    @abstractmethod
    def train(self, X_train: pd.DataFrame, Y_train: pd.Series) -> None:
        """
        Trains the model

        Args:
            X_train: pd.DataFrame
            Y_train: pd.DataFrame

        Returns:
            None
        """

        pass

class LinearRegressionModel(Model):
    """
    Linear Regression model
    """

    def train(self,X_train: pd.DataFrame, Y_train: pd.Series, **kwargs) -> LinearRegression:
        """
        Linear Regression model to train 

        Args:
            X_train: pd.DataFrame
            Y_train: pd.DataFrame

        Returns:
            None
        """

        try:
            model = LinearRegression(**kwargs)
            model.fit(X_train, Y_train)
            logging.info("Model training completed")
            return model
        except Exception as e:
            logging.error(f"Error while training the model: {e}")
            raise e