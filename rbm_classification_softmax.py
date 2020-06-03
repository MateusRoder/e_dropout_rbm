import argparse

import learnergy.utils.logging as logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import utils.load as l

logger = logging.get_logger(__name__)


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(
        usage='Loads a pre-trained RBM and classifies using a softmax layer.')

    parser.add_argument('dataset', help='Dataset identifier',
                        choices=['mnist', 'fmnist', 'kmnist'])

    parser.add_argument('trained_model', help='Path to pre-trained model', type=str)

    parser.add_argument('-batch_size', help='Batch size', type=int, default=256)

    parser.add_argument('-epochs', help='Number of fine-tuning epochs', type=int, default=10)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    dataset = args.dataset
    trained_model = args.trained_model
    batch_size = args.batch_size
    epochs = args.epochs

    # Loads the data
    train, _, test = l.load_dataset(name=dataset)

    # Loads pre-trained model
    model = torch.load(trained_model)

    # Creating the Fully Connected layer to append on top of DBNs
    fc = torch.nn.Linear(model.n_hidden, 10)

    # Check if model uses GPU
    if model.device == 'cuda':
        # If yes, put fully-connected on GPU
        fc = fc.cuda()

    # Cross-Entropy loss is used for the discriminative fine-tuning
    criterion = nn.CrossEntropyLoss()

    # Creating the optimizers
    optimizer = [optim.Adam(fc.parameters(), lr=0.001)]

    # Creating training and validation batches
    train_batch = DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=1)
    test_batch = DataLoader(test, batch_size=10000, shuffle=False, num_workers=1)

    # For amount of fine-tuning epochs
    for e in range(epochs):
        print(f'Epoch {e+1}/{epochs}')

        # Resetting metrics
        train_loss = 0
        
        # For every possible batch
        for x_batch, y_batch in train_batch:
            # For every possible optimizer
            for opt in optimizer:
                # Resets the optimizer
                opt.zero_grad()
            
            # Flatenning the samples batch
            x_batch = x_batch.reshape(x_batch.size(0), model.n_visible)

            # Checking whether GPU is avaliable and if it should be used
            if model.device == 'cuda':
                # Applies the GPU usage to the data and labels
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()

            # Passing the batch down the model
            y = model(x_batch)

            # Calculating the fully-connected outputs
            y = fc(y)
            
            # Calculating loss
            loss = criterion(y, y_batch)
            
            # Propagating the loss to calculate the gradients
            loss.backward()
            
            # For every possible optimizer
            for opt in optimizer:
                # Performs the gradient update
                opt.step()

            # Adding current batch loss
            train_loss += loss.item()

        logger.info(f'Loss: {train_loss / len(train_batch)}')
            
    # Calculate the final accuracy for the model:
    for x_batch, y_batch in test_batch:
        # Flatenning the testing samples batch
        x_batch = x_batch.reshape(x_batch.size(0), model.n_visible)

        # Checking whether GPU is avaliable and if it should be used
        if model.device == 'cuda':
            # Applies the GPU usage to the data and labels
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()

        # Passing the batch down the model
        y = model(x_batch)

        # Calculating the fully-connected outputs
        y = fc(y)

        # Calculating predictions
        _, preds = torch.max(y, 1)

        # Calculating validation set accuracy
        acc = torch.mean((torch.sum(preds == y_batch).float()) / x_batch.size(0))

    logger.info(f'Accuracy: {acc}')
