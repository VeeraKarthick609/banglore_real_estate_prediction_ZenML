from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == "__main__":  
    #to get the traacking url
    print(Client().active_stack.experiment_tracker.get_tracking_uri())

    #run the pipeline
    train_pipeline(data_path="./data/banglore_real_estate_data.csv")
