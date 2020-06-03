def reconstruction(obj, train, val, n_visible, n_hidden, steps, lr, momentum, decay,
                   T, gpu, batch_size, epochs):
    """
    """

    def f(w):
        """
        """

        #
        model = obj(n_visible=n_visible, n_hidden=n_hidden, steps=steps, learning_rate=lr,
                    momentum=momentum, decay=decay, temperature=T, dropout=w[0][0], use_gpu=gpu)

        #
        model.fit(train, batch_size=batch_size, epochs=epochs)

        #
        mse, _ = model.reconstruct(val)

        #
        mse = float(mse.detach().cpu().numpy())

        return mse

    return f
