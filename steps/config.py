from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    """Model Config"""

    model_name: str = "LinearRegression"
    