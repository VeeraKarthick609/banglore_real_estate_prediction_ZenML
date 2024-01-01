import logging

import pandas as pd
from src.data_cleaning import DataCleaning, DataPreprocessStrategy


def get_data_for_test():
    try:
        df = pd.read_csv("data/banglore_real_estate_data.csv")
        df = df.sample(n=100)
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        df = data_cleaning.handle_data()
        df = df.drop(["price"], axis=1, inplace=True)
        return df
    except Exception as e:
        logging.error(e)
        raise e