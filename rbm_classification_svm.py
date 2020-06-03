import argparse

import learnergy.utils.logging as logging
import torch
from sklearn.svm import SVC

import utils.load as l

logger = logging.get_logger(__name__)


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(
        usage='Loads a pre-trained RBM and classifies using SVM.')

    parser.add_argument('dataset', help='Dataset identifier',
                        choices=['mnist', 'fmnist', 'kmnist'])

    parser.add_argument('trained_model', help='Path to pre-trained model', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    dataset = args.dataset
    trained_model = args.trained_model

    # Loads the data
    train, _, test = l.load_dataset(name=dataset)

    # Transforming datasets into tensors
    x_train, y_train = l.dataset_as_tensor(train)
    x_test, y_test = l.dataset_as_tensor(test)

    # Reshaping tensors
    x_train = x_train.view(len(train), 784)
    x_test = x_test.view(len(test), 784)

    # Loads pre-trained model
    model = torch.load(trained_model)

    # Checking model device type
    if model.device == 'cuda':
        # Applying data as cuda
        x_train = x_train.cuda()
        x_test = x_test.cuda()

    # Extract features from the pre-trained RBM
    f_train = model.forward(x_train)
    f_test = model.forward(x_test)

    # Instantiates an SVM
    clf = SVC(gamma='auto')

    # Fits a classifier
    clf.fit(f_train.detach().cpu().numpy(), y_train.detach().cpu().numpy())

    # Performs the final classification
    acc = clf.score(f_test.detach().cpu().numpy(),
                    y_test.detach().cpu().numpy())

    logger.info(f'Accuracy: {acc}')
