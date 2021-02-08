===================
Dimension reduction
===================

.. toctree::
    :hidden:
    :maxdepth: 2

    /notebooks/tica
    notebooks/vamp
    notebooks/kcca
    notebooks/dmd
    notebooks/edmd
    notebooks/kedmd

Here we introduce the dimension reduction / decomposition techniques implemented in the package.

.. rubric:: Koopman operator methods

All methods contained in this sub-package relate to the Koopman operator :math:`\mathcal{K}_\tau` defined as

.. math::
    [\mathcal{K}_\tau g](x) = \mathbb{E}[g(x_{t+\tau}) \mid x_t = x]

for a process :math:`\{x_t\}_{t\geq 0}` with transition density :math:`p_\tau(x, y)`. When projecting
:math:`\mathcal{K}_\tau` into a finite basis, one seeks

.. math::
    \mathbb{E}[g(x_{t+\tau})] = K^{\top}\mathbb{E}[f(x_t)],

where :math:`K\in\mathbb{R}^{n\times m}` is a finite-dimensional Koopman matrix which propagates the observable
:math:`f` of the system's state :math:`x_t` to the observable :math:`g` at state :math:`x_{t+\tau}`.

.. rubric:: When to use which method

All methods assume (approximate) Markovianity of the time series under lag-time :math:`\tau`.

.. raw:: html

    <div id="koopman_algos_container" width="100%"></div>

.. raw:: html
     :file: _static/koopman_algorithms.html

.. list-table::
    :header-rows: 1
    :align: left
    :widths: 10 30 40

    * - Method
      - Assumptions
      - Notes

    * - `TICA <notebooks/tica.ipynb>`__ :footcite:`molgedey1994separation` :footcite:`naritomi2011slow`
      - Time series should be stationary with symmetric covariances (equivalently: reversible with detailed balance)
        and compact Koopman operator (guaranteed to be compact in stochastic systems).
      - * Based on variational principle
        * TICA uses the assumptions as prior and thus can yield more interpretable results than
          its generalization VAMP.
        * Might yield biased results if the observed process contains rare events which are not sufficiently
          reflected in the time-series.
        * Dual to DMD: The estimated matrix is the transpose of the one estimated by DMD; TICA estimates eigenfunctions
          and DMD estimates 'modes', the coefficients which lead to the approximate Koopman operator
          using the eigenfunctions.
        * Eigenvalues of the decomposition relate to relaxation timescales.
        * Can identify metastabile sets.
        * Is VAMP if system is reversible and ansatz library contains only the full state observable :math:`\Psi(x) = x`
        * Can deal with a large amount of frames due to online estimation of covariances

    * - `VAMP <notebooks/vamp.ipynb>`__ :footcite:`wu2020variational`
      - Compact Koopman operator (guaranteed to be compact in stochastic systems)
      - * Based on variational principle
        * Is canonical correlation analysis (CCA) in time, i.e., time-lagged CCA (TCCA)
        * Is VAC under reversible dynamics and VAMP-1 score
        * Uses a singular value decomposition of covariances instead of eigenvalue decomposition.
        * Deals with off-equilibrium data consistently.
        * The singular functions can be clustered to find metastable and coherent sets.
        * Equivalent to EDMD in case of the VAMP-1 score
        * Uses ansatz library :math:`\Psi` of functions as basis
        * Can deal with a large amount of frames due to online estimation of covariances

    * - `kernel VAMP / kernel CCA <notebooks/kcca.ipynb>`__ :footcite:`bach2002kernel`
      - Compact Koopman operator (guaranteed to be compact in stochastic systems)
      - * kernelized version of VAMP with :math:`k(x, x') = \langle\Psi(x), \Psi(x')\rangle`
        * memory requirements grow quadratically with the number of frames
        * generalization of kernel EDMD to nonreversible dynamics

    * - `DMD <notebooks/dmd.ipynb>`__ :footcite:`schmid2010dynamic` :footcite:`tu2013dynamic`
      -
      - * dual to TICA (DMD yields coefficients of eigenfunctions, TICA yields eigenfunctions)
        * not an online algorithm but better suited for very high-dimensional data with a lower number of frames

    * - `EDMD <notebooks/edmd.ipynb>`__ :footcite:`williams2015data`
      -
      - * Equivalent to VAMP under the VAMP-1 score
        * Uses an ansatz library :math:`\Psi`
        * not an online algorithm but better suited for very high-dimensional data with a lower number of frames

    * - `kernel EDMD <notebooks/kedmd.ipynb>`__ :footcite:`williams2016kernel`
      -
      - * kernelized version of EDMD with :math:`k(x, x') = \langle\Psi(x), \Psi(x')\rangle`
        * memory requirements grow quadratically with the number of frames
        * considering generator kernel EDMD, `SINDy <notebooks/sindy.ipynb>`__ arises as
          special case :footcite:`klus2020data`

It should be noted that, if available, scores evaluated under different lagtimes are **not** comparable because they
relate to different operators.

.. rubric:: What's next?

While a dimensionality reduction is always of great use because it makes it easier to look at the data,
one can take further steps.

A commonly performed pipeline would be to `cluster <notebooks/clustering.ipynb>`_ the projected data and then
building a `markov state model <index_msm.rst>`_ on the resulting discretized state space.

..
    !! This block is commented out !!

    Furthermore, TICA might yield biased results if the data contains rare events. If the underlying
    distribution is expected to obey microscopic reversibility, not enough sampling might have been performed to
    actually reflect this in the estimated model.

    Numerically, VAMP works with a singular value decomposition while TICA works with an eigenvalue decomposition.
    These singular values should theoretically be real and coincide with the TICA eigenvalues if the data comes
    from a reversible and in-equilibrium setting. Numerically as well as due to sampling, they still might be complex.
    This means in particular that they are hard to interpret and there is no simple relationship to, e.g., timescales.

    On the other hand,



    The third of the described cases is salvageable with TICA when a special reweighting procedure
    (`Koopman reweighting <notebooks/tica.ipynb#Koopman-reweighting>`_) is used.

    The point of view then transitions from a "metastability" one to "coherent sets" - metastabilies' analogon in
    off-equilibirum cases and commonly encountered in, e.g., fluid dynamics. The rough idea is that one can then
    identify regions which (approximately) stay together under temporal propagation.


.. rubric:: Estimating covariances and how to deal with large amounts of data

While the implementations of `TICA <notebooks/tica.ipynb>`_ and its generalization `VAMP <notebooks/vamp.ipynb>`_
can be fit directly by a time series that is kept in the computer's memory, this might not always be possible.

The implementations are based on estimating covariance matrices, by default using the
`covariance estimator <api/generated/deeptime.covariance.Covariance.rst#deeptime.covariance.Covariance>`_.

This estimator makes use of an online algorithm, so that it can be fit in a streaming fashion:

.. code-block:: python

    estimator = deeptime.decomposition.TICA(lagtime=tau)  # creating an estimator
    estimator = deeptime.decomposition.VAMP(lagtime=tau)  # either TICA or VAMP

Since toy data usually easily fits into memory, loading data from, e.g., a database or network is simulated with the
`timeshifted_split() <api/generated/deeptime.data.timeshifted_split.rst#deeptime.data.timeshifted_split>`_ utility function.
It splits the data into timeshifted blocks :math:`X_t` and :math:`X_{t+\tau}`.

These blocks are not trajectory-overlapping, i.e., if two or more trajectories are provided then the blocks are
always completely contained in exactly one of these.

Note how here we provide both blocks, the block :math:`X_t` and the block :math:`X_{t+\tau}` as a tuple.
This is different to :code:`fit()` where the splitting and shifting is performed internally; in which case it
suffices to provide the whole dataset as argument.

.. code-block:: python

    for X, Y in deeptime.data.timeshifted_split(feature_trajectory, lagtime=tau, chunksize=100):
        estimator.partial_fit((X, Y))

Furthermore, the online algorithm uses a tree-like moment storage with copies of
intermediate covariance and mean estimates. During the learning procedure, these moment storages are combined so
that the tree never exceeds a certain depth. This depth can be set by the `ncov` estimator parameter:

.. code-block:: python

    estimator = deeptime.decomposition.TICA(lagtime=1, ncov=50)
    for X, Y in deeptime.data.timeshifted_split(feature_trajectory, lagtime=1, chunksize=10):
        tica.partial_fit((X, Y))

Another factor to consider is numerical stability. While memory consumption can increase with
larger `ncov`, the stability generally improves.

.. rubric:: References

.. footbibliography::
