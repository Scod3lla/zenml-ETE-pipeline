
from zenml.pipelines import pipeline

from zenml.config import DockerSettings
from zenml.integrations.constants import WANDB, PYTORCH
from zenml.pipelines import pipeline

docker_settings = DockerSettings(required_integrations=[WANDB, PYTORCH])


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def pytorch_experiment_tracking_pipeline(
    load_data,
    load_model,
    train_test,
):
    """A `pipeline` to load data, load model, and train/evaluate the model."""
    train_dataloader, test_dataloader, inputSize, outputSize = load_data()
    model = load_model(inputSize, outputSize)
    model, test_acc = train_test(model, train_dataloader, test_dataloader)
