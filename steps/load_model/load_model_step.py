import torch
from torch import nn
from zenml.steps import step

device = "cuda" if torch.cuda.is_available() else "cpu"

class LinearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out




@step(enable_cache=False)
def load_model(inputSize: int, outputSize: int) -> nn.Module:
    """A `step` to define a PyTorch model."""
    model = LinearRegression(inputSize, outputSize)
    print(model)
    return model
