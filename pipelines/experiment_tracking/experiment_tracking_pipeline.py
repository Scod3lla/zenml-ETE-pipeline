from zenml.config import DockerSettings
from zenml.integrations.constants import WANDB, PYTORCH
from zenml import pipeline

from steps.dataloader.dataloader_step import dataloader
from steps.load_model.load_model_step import load_model
from steps.train_evaluate.train_test_step import train_test



docker_settings = DockerSettings(required_integrations=[WANDB, PYTORCH])


@pipeline(settings={"docker": docker_settings})
def pytorch_experiment_tracking_pipeline(
            data_path: str, batch_size: int, 
            learning_rate: float, epochs: int
):
    """A `pipeline` to load data, load model, and train/evaluate the model."""
    train_dataloader, test_dataloader, inputSize, outputSize = dataloader(data_path, batch_size)
    model = load_model(inputSize, outputSize)
    model, test_acc = train_test(model, train_dataloader, test_dataloader, learning_rate, epochs, batch_size)
