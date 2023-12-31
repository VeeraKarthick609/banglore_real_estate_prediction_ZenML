import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from pandas.core.api import Series as Series
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    """
    Abstract class defining strategy for data cleaning
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreprocessStrategy(DataStrategy):
    """
    Strategy for cleaning data
    """

    @staticmethod
    def get_size(x):
        return float(x.split(" ")[0])

    @staticmethod
    def is_float(x):
        try:
            float(x)
            return True
        except:
            return False

    @staticmethod
    def get_avg(x):
        tokens = x.split("-")
        if len(tokens) == 2:
            return (float(tokens[0]) + float(tokens[1])) / 2

        try:
            return float(x)
        except:
            return None

    @staticmethod
    def change_price(x):
        return float(x)

    @staticmethod
    def check_loc_stats(x, other_loc):
        if x in other_loc.values:
            return "other"
        else:
            return x
    
    @staticmethod
    def remove_pps_outlier(df, df2):
        df_out = pd.DataFrame()
        for key, subdf in df2.groupby("location"):
            mean = np.mean(subdf["price_per_sqft"])
            std_dev = np.std(subdf["price_per_sqft"])

            ## checking validitiy
            reduced_df = subdf[
                (subdf["price_per_sqft"] > (mean - std_dev))
                & (subdf["price_per_sqft"] <= (mean + std_dev))
            ]
            df_out = pd.concat([df_out, reduced_df], ignore_index=True)
        return df_out

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Method to clean the data
        """

        try:
            data = data.drop(
                columns=["area_type", "society", "availability", "balcony"],
                axis="columns",
            )
            data = data.dropna()

            data["BHK"] = data["size"].apply(self.get_size)
            data = data[~data["total_sqft"].apply(self.is_float)]

            data_copy = data.copy()

            data_copy["total_sqft"] = data_copy["total_sqft"].apply(self.get_avg)
            data_copy["price"] = data_copy["price"].apply(self.change_price)

            data_copy["location"] = data_copy["location"].apply(lambda x: x.strip())
            locations = data_copy["location"].value_counts(ascending=False)

            other_loc = locations[locations <= 10]
            data_copy["location"] = data_copy["location"].apply(
                lambda x: DataPreprocessStrategy.check_loc_stats(x, other_loc)
            )

            data_copy = data_copy[(data_copy["total_sqft"] / data_copy["BHK"]) >= 300]

            data_copy = data_copy[data_copy["bath"] <= data_copy["BHK"]]

            data_copy = data_copy.drop(
                columns=["size"], axis="columns"
            )

            dummies_loc = pd.get_dummies(data_copy["location"], dtype="int")

            final_df = pd.concat(
                [data_copy.drop(columns=["location"], axis="columns"), dummies_loc],
                axis="columns",
            )

            return final_df

        except Exception as e:
            logging.error(f"Error while cleaning data: {e}")
            raise e


class DataDivideStrategy(DataStrategy):
    """
    Strategy for data dividing
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame , pd.Series]:
        """
        divide data into train data and test data

        Args:
            data: pd.DataFrame

        Returns:
            split_data: Union[pd.DataFrame, pd.Series]
        """

        try:
            X = data.drop(columns=["price"], axis="columns")
            Y = data["price"]
            X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=42, shuffle=True, train_size=0.75)
            return X_train, X_test, Y_train, Y_test
        except Exception as e:
            logging.error(f"Error while splitting data: {e}")
            raise e
        

class DataCleaning:
    """
    Class for cleaning the data (data preprocessing and data split)
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        self.data = data
        self.strategy = strategy

    
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle data
        """

        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error while cleaning data: {e}")
            raise e
        