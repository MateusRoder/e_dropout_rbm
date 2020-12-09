import torch
import torchvision as tv

# A constant used to hold a dictionary of possible datasets
DATASETS = {
    'mnist': tv.datasets.MNIST,
    'fmnist': tv.datasets.FashionMNIST,
    'kmnist': tv.datasets.KMNIST
}


def load_dataset(name='mnist', val_split=0.2):
    """Loads an input dataset.

    Args:
        name (str): Name of dataset to be loaded.
        val_split (float): Percentage of split for the validation set.

    Returns:
        Training, validation and testing sets of loaded dataset.

    """

    # Defining the torch seed
    torch.manual_seed(0)

    # Loads the training data
    train = DATASETS[name](root='./data', train=True, download=True,
                           transform=tv.transforms.ToTensor())

    # Splitting the training data into training/validation
    train, val = torch.utils.data.random_split(
        train, [int(len(train) * (1 - val_split)), int(len(train) * val_split)])

    # Loads the testing data
    test = DATASETS[name](root='./data', train=False, download=True,
                          transform=tv.transforms.ToTensor())

    return train, val, test


def dataset_as_tensor(dataset):
    """Transforms a PyTorch dataset into tensors.

    Args:
        dataset (torch.utils.data.Dataset): PyTorch dataset.

    Returns:
        Data and labels into a tensor formatting.

    """

    # Creates batches using PyTorch's DataLoader
    batches = torch.utils.data.DataLoader(
        dataset, batch_size=len(dataset), shuffle=False, num_workers=1)

    # Iterates through the single batch
    for batch in batches:
        # Returns data and labels
        return batch[0], batch[1]
