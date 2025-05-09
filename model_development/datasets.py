from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split  # Added imports
import lightning as L
import torch

import lightning as L
from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST
from torchvision import transforms

# Define the PyTorch dataset
def get_mnist_datasets(data_dir="./data"):
    """
    Create PyTorch datasets for MNIST.

    Args:
        data_dir (str): Directory to store the MNIST data.

    Returns:
        train_dataset, val_dataset, test_dataset: PyTorch datasets for training, validation, and testing.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load the full training dataset and split into train/val
    mnist_full = MNIST(data_dir, train=True, download=True, transform=transform)
    train_dataset, val_dataset = random_split(mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42))

    # Load the test dataset
    test_dataset = MNIST(data_dir, train=False, download=True, transform=transform)

    return train_dataset, val_dataset, test_dataset

# Define the PyTorch dataloaders
def get_mnist_dataloaders(data_dir="./data", batch_size=32):
    """
    Create PyTorch dataloaders for MNIST.

    Args:
        data_dir (str): Directory to store the MNIST data.
        batch_size (int): Batch size for the dataloaders.

    Returns:
        train_loader, val_loader, test_loader: PyTorch dataloaders for training, validation, and testing.
    """
    train_dataset, val_dataset, test_dataset = get_mnist_datasets(data_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./data"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=1)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        self.mnist_train = None
        self.mnist_val = None
        self.mnist_test = None
        self.mnist_predict = None