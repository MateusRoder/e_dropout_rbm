def reconstruction(obj, train, val, n_visible, n_hidden, steps, learning_rate, momentum, decay,
                   temperature, p, use_gpu, batch_size, epochs):
    """
    """

    #
    model = obj(n_visible=n_visible, n_hidden=n_hidden, steps=steps, learning_rate=lr,
                momentum=momentum, decay=decay, temperature=T, use_gpu=gpu)

    def f(w):
        """
        """

        #
        model.p = w[0]

        #
        model.fit(train, batch_size=batch_size, epochs)

        #
        mse, _ = model.reconstruct(val)

        return mse

    return f
