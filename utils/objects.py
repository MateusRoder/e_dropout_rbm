from learnergy.models.bernoulli import RBM, DropoutRBM, EDropoutRBM, DropConnectRBM


class Model:
    """A Model class to help users in selecting distinct RBMs from the command line.

    """

    def __init__(self, obj):
        """Initialization method.

        Args:
            obj (Optimizer): An Optimizer-child instance.
            hyperparams (dict): Meta-heuristic hyperparams.

        """

        # Creates a property to hold the class itself
        self.obj = obj


# Defines a meta-heuristic dictionary constant with the possible values
MODEL = dict(
    rbm=Model(RBM),
    dcrbm=Model(DropConnectRBM),
    drbm=Model(DropoutRBM),
    edrbm=Model(EDropoutRBM)
)


def get_model(name):
    """Gets a model by its identifier.

    Args:
        name (str): Model's identifier.

    Returns:
        An instance of the Model class.

    """

    # Tries to invoke the method
    try:
        # Returns the corresponding object
        return MODEL[name]

    # If object is not found
    except:
        # Raises a RuntimeError
        raise RuntimeError(f'Model {name} has not been specified yet.')
