def reconstruction(obj, train, val, n_visible, n_hidden, steps, lr,
                   momentum, decay, T, gpu, batch_size, epochs):
    """Wraps the reconstruction task for optimization purposes.

    Args:
        obj (learnergy.models.DropoutRBM): A DropoutRBM object.
        train (torch.utils.data.Dataset): A Dataset object containing the training data.
        val (torch.utils.data.Dataset): A Dataset object containing the validation data.
        n_visible (int): Amount of visible units.
        n_hidden (int): Amount of hidden units.
        steps (int): Number of Gibbs' sampling steps.
        lr (float): Learning rate.
        momentum (float): Momentum parameter.
        decay (float): Weight decay used for penalization.
        T (float): Temperature factor.
        gpu (boolean): Whether GPU should be used or not.
        batch_size (int): Amount of samples per batch.
        epochs (int): Number of training epochs.

    """

    def f(w):
        """Fits on training data and reconstructs on validation data.

        Args:
            w (float): Array of variables.

        Returns:
            Mean Squared Error (MSE).

        """

        # Creates the model itself
        model = obj(n_visible=n_visible, n_hidden=n_hidden, steps=steps, learning_rate=lr,
                    momentum=momentum, decay=decay, temperature=T, dropout=w[0][0], use_gpu=gpu)

        # Fits on training data
        model.fit(train, batch_size=batch_size, epochs=epochs)

        # Reconstructs using validation data
        mse, _ = model.reconstruct(val)

        # Transforms the tensor into a float
        mse = float(mse.detach().cpu().numpy())

        return mse

    return f
