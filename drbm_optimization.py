import argparse

import torch

import utils.load as l
import utils.models as m
import utils.target as t


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(usage='Trains, reconstructs and saves an RBM model.')

    parser.add_argument('dataset', help='Dataset identifier', choices=['mnist', 'fmnist', 'kmnist'])

    parser.add_argument('mh', help='Meta-heuristic identifier', choices=['pso'])

    parser.add_argument('-n_visible', help='Number of visible units', type=int, default=784)

    parser.add_argument('-n_hidden', help='Number of hidden units', type=int, default=128)

    parser.add_argument('-steps', help='Number of CD steps', type=int, default=1)

    parser.add_argument('-lr', help='Learning rate', type=float, default=0.1)

    parser.add_argument('-momentum', help='Momentum', type=float, default=0)

    parser.add_argument('-decay', help='Weight decay', type=float, default=0)

    parser.add_argument('-temp', help='Temperature', type=float, default=1)

    parser.add_argument('-gpu', help='GPU usage', type=bool, default=True)

    parser.add_argument('-batch_size', help='Batch size', type=int, default=256)

    parser.add_argument('-epochs', help='Number of training epochs', type=int, default=50)

    parser.add_argument('-n_agents', help='Number of meta-heuristic agents', type=int, default=10)

    parser.add_argument('-n_iter', help='Number of meta-heuristic iterations', type=int, default=10)

    parser.add_argument('-seed', help='Seed identifier', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering common variables
    dataset = args.dataset
    seed = args.seed

    # Gathering RBM variables
    n_visible = args.n_visible
    n_hidden = args.n_hidden
    steps = args.steps
    lr = args.lr
    momentum = args.momentum
    decay = args.decay
    T = args.temp
    gpu = args.gpu
    batch_size = args.batch_size
    epochs = args.epochs
    model = m.get_model('drbm').obj

    # Gathering optimization variables
    meta = args.mh
    n_agents = args.n_agents
    n_iterations = args.n_iter
    meta_heuristic = m.get_mh(meta).obj
    hyperparams = m.get_mh(meta).hyperparams

    # Loads the data
    train, val, _ = l.load_dataset(name=dataset)

    # Defining the torch seed
    torch.manual_seed(seed)

    # Initializes the optimization target
    opt_fn = t.reconstruction(model, train, val, n_visible, n_hidden, steps, lr, momentum, decay, T, p, gpu, batch_size, epochs)

    # Running the optimization task
    history = w.optimize(meta_heuristic, opt_fn, n_agents, n_iterations, hyperparams)

    # Saves the history object to an output file
    history.save(f'models/{meta}_{n_hidden}hid_{lr}lr_drbm_{dataset}_{seed}.pkl')
