from zenml import pipeline

from steps.clean_data import clean_data
from steps.ingest_data import ingest_data
from steps.train_model import train_model
from steps.evaluation import evaluate_model

@pipeline
def train_pipeline(data_path: str):
    df = ingest_data(data_path)
    X_train, X_test, Y_train, Y_test  = clean_data(df)
    model = train_model(X_train, Y_train)
    mse, rmse, r2 = evaluate_model(model ,X_test, Y_test)
    