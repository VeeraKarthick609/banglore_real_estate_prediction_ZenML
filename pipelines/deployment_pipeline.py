import numpy as np
import pandas as pd

from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output

from steps.clean_data import clean_data
from steps.ingest_data import ingest_data
from steps.train_model import train_model
from steps.evaluation import evaluate_model

docker_settings = DockerSettings(required_integrations=[MLFLOW])


class DeploymentTriggerConfig(BaseParameters):
    """Deployment Trigger Config"""

    min_accuracy: float = 0.85


@step
def deployment_trigger(accuracy: float, config: DeploymentTriggerConfig) -> bool:
    return accuracy >= config.min_accuracy


@pipeline(enable_cache=True, settings={"docker": docker_settings})
def continous_deployment_pipeline(
    data_path: str,
    min_accuracy: float = 0.92,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    df = ingest_data(data_path)
    X_train, X_test, Y_train, Y_test = clean_data(df)
    model = train_model(X_train, Y_train)
    mse, rmse, r2 = evaluate_model(model, X_test, Y_test)
    deployment_decision = deployment_trigger(r2)
    mlflow_model_deployer_step(
        model = model,
        deploy_decision = deployment_decision,
        workers = workers,
        timeout = timeout,
    )

