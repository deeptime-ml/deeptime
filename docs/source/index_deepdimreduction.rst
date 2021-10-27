==================
Deep dim reduction
==================

Here we present techniques which can be used to project timeseries onto dominant processes aided by deep neural
networks. In order to use the implementations provided by deeptime, a working installation of
`PyTorch <https://pytorch.org/>`__ is required.

In particular, the following methods are implemented:

.. toctree::
    :maxdepth: 2

    notebooks/vampnets
    notebooks/tae

VAMPNets :footcite:`mardt2018vampnets` belong to the family of Koopman methods and try to maximize
a variational score which is described, e.g., `here <notebooks/vamp.ipynb#Scoring>`__. They can be used
to find transformations of the observed data such that the estimated Koopman operator is particularly close
to the real and underlying Koopman operator.

Time-lagged autoencoders :footcite:`wehmeyer2018timelagged` on the other hand can also to some degree
be related to Koopman theory (see the reference for details), but at their core try to learn a mapping of
the input data to a latent code which can then be transformed back to the next point in time, i.e., they try to find
mappings :math:`E: \mathbb{R}^N \to\mathbb{R}^n` and :math:`D: \mathbb{R}^n\to\mathbb{R}^N` with :math:`n < N`
such that :math:`x_{t+\tau}\approx (D\circ E)(x_t)`.

.. rubric:: References

.. footbibliography::
