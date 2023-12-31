import logging 
import pandas as pd 
from typing import Tuple
from typing_extensions import Annotated

from zenml import step
from src.data_cleaning import DataDivideStrategy, DataPreprocessStrategy, DataCleaning



@step
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "Training features"],
    Annotated[pd.DataFrame, "Testing features"],
    Annotated[pd.Series, "Training labels"],
    Annotated[pd.Series, "Testing labels"]
]:
    """
    Cleans the data and returns splitted data

    Args:
        raw_data: pd.DataFrame
    
    Returns:
        X_train: pd.DataFrame
        X_test: pd.DataFrame
        Y_train: pd.Series
        Y_test: pd.Series
    """

    try:
        data_preprocess = DataCleaning(data= df, strategy= DataPreprocessStrategy())
        preprocessed_data = data_preprocess.handle_data()

        data_split = DataCleaning(data= preprocessed_data, strategy= DataDivideStrategy())    
        X_train, X_test, Y_train, Y_test = data_split.handle_data()
        
        logging.info("Data cleaning completed")

        return X_train, X_test, Y_train, Y_test
    
    except Exception as e:
        logging.error(f"Error while cleaning the data: {e}")
        raise e

