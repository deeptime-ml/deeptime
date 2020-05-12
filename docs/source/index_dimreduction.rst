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



- The type of decomposition used in TICA is an eigenvalue decomposition, VAMP uses a singular value decomposition.
  These singular values are in theory identical to the eigenvalues if the data is reversible and in equilibrium.


- tica gives eigenvalues, vamp gives singular values

  - singular values are (theoretically) same to eigenvalues if reversible & equilibrium
  - tica needs EVD, vamp needs SVD
  - tica: real eigenvalues (in [-1, 1]), eigenvectors, stat dist <-> perron frobenius thm

- only eigenvalues are directly related to timescales
- tica allows to make a prior assumption that data is reversible and in equilibrium
  - or reversible with detailed balance

- if you are in the molecular kinetics context, and the reversibility assumption makes sense for your system
  (conceptually) AND your data (no rare events), i would always vote tica
  - cause you get the interpretability via eignevalue decomposition and perron frobenius

- 3 irreversible cases are:

  - 1. "truly" irreversible like turbulence,
  - 2. reversible dynamics combined w/ a time-dependent force like a potential,
  - 3. or reversible dynamics but not enough sampling to mitigate rare event effects
  - in all those 3 cases, you'd want vamp, cause the tica bias would be too much
  - except maybe the 3rd you can do the reweighting whatever

- gets at this, where when you lose the eigenvalues, you lose the "metastability" idea and get
  more in the "coherence" realm of eg fluid dynamics
