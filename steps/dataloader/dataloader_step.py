import pandas as pd

from zenml.steps import Output
from zenml import step

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader




class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path, header=None, delim_whitespace=True)
        x = self.data.iloc[:,:-2].values
        y = self.data.iloc[:,-1].values

        x = torch.tensor(x, dtype =torch.float32)
        self.x_train = torch.nn.functional.normalize(x, p=1.0, dim=1, eps=1e-12, out=None)

        self.y_train = torch.tensor(y, dtype =torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.x_train)
    
    def __getitem__ (self,idx):
        return self.x_train[idx], self.y_train[idx]


@step(enable_cache=False)
def dataloader(data_path: str, batch_size : int) -> Output(
    train_dataloader=DataLoader, test_dataloader=DataLoader, inputSize=int, outputSize=int
    ):
    data = CustomDatasetFromCSV(data_path)

    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    x, _ = train_dataset[0]
    inputSize = x.shape[-1]
    outputSize = 1


    return train_dataloader, test_dataloader, inputSize, outputSize
