import torch
from zenml.steps import Output
from zenml import step

from torch.utils.data import DataLoader
from torch import nn

import wandb


from zenml.integrations.wandb.flavors.wandb_experiment_tracker_flavor import WandbExperimentTrackerSettings

wandb_settings = WandbExperimentTrackerSettings(settings=wandb.Settings(magic=True), tags=["some_tag"])


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def train(dataloader, model, loss_fn, optimizer, global_step):
    """A function to train a model for one epoch."""
    size = len(dataloader.dataset)

    mloss = 0 
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # print(y.shape)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mloss+=loss.item()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            # 🔥 W&B tracking
    mloss /= len(dataloader)
    wandb.log({"Train Loss": mloss}, step=global_step)

def test(dataloader, model, loss_fn, global_step):
    """A function to test a model on the validation / test dataset."""
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

    # 🔥 W&B tracking
    wandb.log({"Test Loss": test_loss}, step=global_step)

    return test_loss


# --------------------------------------------------------------------------------





@step(enable_cache=False, experiment_tracker="wandb_tracker",
      settings={
        "experiment_tracker.wandb": wandb_settings
    }
)
def train_test(
    model: nn.Module,
    train_dataloader: DataLoader, 
    test_dataloader: DataLoader,
    learning_rate : float,
    epochs : int,
    batch_size: int
) -> Output(trained_model=nn.Module, test_acc=float):
    """A `step` to train and evaluate a torch model on given dataloaders."""

    model = model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    test_acc = 0

    wandb.config.update({"lr": learning_rate, "epochs": epochs, 'batch_size': batch_size})

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        global_step = t+1
        train(train_dataloader, model, loss_fn, optimizer, global_step)
        test_acc = test(test_dataloader, model, loss_fn, global_step)
    print("Done!")

    return model, test_acc