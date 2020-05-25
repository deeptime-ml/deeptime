========================
Dimensionality reduction
========================

Here we introduce the dimensionality reduction / decomposition techniques implemented in the package.

.. toctree::
    :maxdepth: 2

    notebooks/tica
    notebooks/vamp

When to use TICA
================

If the observed process is reversible with detailed balance (i.e., forward and backward in time is indistinguishable)
and the data also supports this (i.e., there are no rare events), then TICA is able to apply
this as a prior assumption.

Furthermore, it works on an eigenvalue decomposition, where the eigenvalues are directly related to
timescales via

.. math:: t_i = -\frac{\tau}{\ln |\lambda_i|}

and are contained in the closed interval :math:`\lambda_i\in[-1, 1]`. This has great benefits in terms of
interpretability.

As soon as the data is off equilibrium though, TICA estimates can become heavily biased.
A salvageable case is if the underlying dynamics is reversible but there is not enough sampling to support this: a
`reweighting procedure <notebooks/tica.ipynb#Koopman-reweighting>`_ can be applied :cite:`wu2016variational`.


When to use VAMP
================

VAMP is a strict generalization of TICA and is applicable in a wider range of settings, however it has some
differences in terms of the numerics involved as well as the estimated model's interpretability.

First and foremost it extends to off-equilibrium cases and is equipped with a set of scoring functions (the so-called
VAMP scores), which allow to rank a selection of input features. Intuitively these scores measure the amount of
"slowness" that can be captured with a feature selection. It is however **not** possible to compare scores
over several lag-time selections.

Numerically, VAMP works with a singular value decomposition while TICA works with an eigenvalue decomposition.
These singular values should theoretically be real and coincide with the TICA eigenvalues if the data comes
from a reversible and in-equilibrium setting. Numerically as well as due to sampling, they still might be complex.
This means in particular that they are hard to interpret and there is no simple relationship to, e.g., timescales.

On the other hand, VAMP deals with off-equilibrium cases consistently in which TICA becomes heavily biased (except
for special cases).

One can divide off-equilibrium cases into three subcategories (as presented in :cite:`koltai2018optimal`):

1. *Time-inhomogeneous dynamics*, e.g, the observed system is driven by a time-dependent external force,
2. *Time-homogeneous non-reversible dynamics*, i.e., detailed balance is not obeyed and the observations might be
   of a non-stationary regime.
3. *Reversible dynamics but non-stationary data*, i.e., the system possesses a stationary distribution with respect
   to which it obeys detailed balance, but the empirical of the available data did not converge to this stationary
   distribution.

The third of the described cases is salvageable with TICA when a special reweighting procedure
(`Koopman reweighting <notebooks/tica.ipynb#Koopman-reweighting>`_) is used.

The point of view then transitions from a "metastability" one to "coherent sets" - metastabilies' analogon in
off-equilibirum cases and commonly encountered in, e.g., fluid dynamics. The rough idea is that one can then
identify regions which (approximately) stay together under temporal propagation.


Performance, numerical stability, and memory consumption
========================================================

The implementations of `TICA <notebooks/tica.ipynb>`_ and its generalization `VAMP <notebooks/vamp.ipynb>`_
are based on estimating covariance matrices using the `covariance estimator <api/generated/sktime.covariance.Covariance.rst#sktime.covariance.Covariance>`_.
This estimator makes use of an online algorithm proposed in :cite:`chan1982updating` so that not the entire data has
to be kept in memory.

In particular, this means that TICA and VAMP not only have a :code:`fit`-method (`tica.fit() <api/generated/sktime.decomposition.TICA.rst#sktime.decomposition.TICA.fit>`_,
`vamp.fit() <api/generated/sktime.decomposition.VAMP.rst#sktime.decomposition.VAMP.fit>`_), but also a
:code:`partial_fit`-method (`tica.partial_fit() <api/generated/sktime.decomposition.TICA.rst#sktime.decomposition.TICA.partial_fit>`_,
`vamp.partial_fit() <api/generated/sktime.decomposition.VAMP.rst#sktime.decomposition.VAMP.partial_fit>`_).

.. code-block:: python

    estimator = sktime.decomposition.TICA(lagtime=tau)  # creating an estimator
    estimator = sktime.decomposition.VAMP(lagtime=tau)  # either TICA or VAMP

Since toy data usually easily fits into memory, loading data from, e.g., a database or network is simulated with the
`timeshifted_split() <api/generated/sktime.data.timeshifted_split.rst#sktime.data.timeshifted_split>`_ utility function.
It splits the data into timeshifted blocks :math:`X_t` and :math:`X_{t+\tau}`.

These blocks are not trajectory-overlapping, i.e., if two or more trajectories are provided then the blocks are
always completely contained in exactly one of these.

Note how here we provide both blocks, the block :math:`X_t` and the block :math:`X_{t+\tau}` as a tuple.
This is different to :code:`fit()` where the splitting and shifting is performed internally; in which case it
suffices to provide the whole dataset as argument.

.. code-block:: python

    for X, Y in sktime.data.timeshifted_split(feature_trajectory, lagtime=tau, chunksize=100):
        estimator.partial_fit((X, Y))

Furthermore, the online algorithm of :cite:`chan1982updating` uses a tree-like moment storage with copies of
intermediate covariance and mean estimates. During the learning procedure, these moment storages are combined so
that the tree never exceeds a certain depth. This depth can be set by the `ncov` estimator parameter:

.. code-block:: python

    estimator = sktime.decomposition.TICA(lagtime=1, ncov=50)
    for X, Y in sktime.data.timeshifted_split(feature_trajectory, lagtime=1, chunksize=10):
        tica.partial_fit((X, Y))

Another factor to consider is numerical stability. While memory consumption can increase with
larger `ncov`, the stability generally improves.

What's next?
============

While a dimensionality reduction is always of great use because it makes it easier to look at the data,
one can take further steps.

A commonly performed pipeline would be to `cluster <notebooks/clustering.ipynb>`_ the projected data and then
building a `markov state model <index_msm.rst>`_ on the resulting discretized state space.
