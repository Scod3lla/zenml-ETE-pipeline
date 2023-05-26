import torch
from torch import nn
from torch.utils.data import DataLoader


from zenml.pipelines import pipeline
from zenml.steps import step, Output



@pipeline
def pytorch_experiment_tracking_pipeline(
    load_data,
    load_model,
    train_test,
):
    """A `pipeline` to load data, load model, and train/evaluate the model."""
    train_dataloader, test_dataloader, inputSize, outputSize = load_data()
    model = load_model(inputSize, outputSize)
    model, test_acc = train_test(model, train_dataloader, test_dataloader)
