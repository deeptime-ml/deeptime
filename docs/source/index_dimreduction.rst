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

- the score should be mentioned

