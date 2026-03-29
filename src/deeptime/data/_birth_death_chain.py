import numpy as np
from scipy.sparse import diags


class BirthDeathChain:
    r"""Birth and death chain.

    A general birth and death chain on a d-dimensional state space
    has the following transition matrix

    .. math::

        p_{ij} = \begin{cases}
            q_i &\text{, if } j=i-1 \text{ and } i>0,\\
            r_i &\text{, if } j=i,\\
            p_i &\text{, if } j=i+1 \text{ and } i < d-1
        \end{cases}

    Parameters
    ----------
    q : array_like
        Annihilation probabilities for transition from i to i-1.
    p : array-like
        Creation probabilities for transition from i to i+1.
    sparse : bool, optional, default=False
        Whether sparse matrices are used.
    """

    def __init__(self, q, p, sparse: bool = False):
        q = np.asarray(q)
        p = np.asarray(p)
        if q[0] != 0.0:
            raise ValueError('Probability q[0] must be zero')
        if p[-1] != 0.0:
            raise ValueError('Probability p[-1] must be zero')
        if not np.all(q + p <= 1.0):
            raise ValueError('Probabilities q+p can not exceed one')
        self.q = q
        self.p = p
        self.r = 1 - self.q - self.p
        self.dim = self.r.shape[0]
        self.sparse = sparse

    @property
    def transition_matrix(self):
        r""" Transition matrix for birth and death chain with given
        creation and anhilation probabilities.

        :getter: Yields the transition matrix.
        :type: (N,N) ndarray
        """
        if not self.sparse:
            P0 = np.diag(self.r, k=0)
            P1 = np.diag(self.p[0:-1], k=1)
            P_1 = np.diag(self.q[1:], k=-1)
            return P0 + P1 + P_1
        else:
            return diags([self.q[1:], self.r, self.p[0:-1]], [-1, 0, 1])

    @property
    def msm(self):
        r""" MarkovStateModel for this birth death chain

        :getter: Yields the MSM.
        :type: deeptime.markov.msm.MarkovStateModel
        """
        from deeptime.markov.msm import MarkovStateModel
        return MarkovStateModel(self.transition_matrix, self.stationary_distribution)

    @property
    def stationary_distribution(self):
        r""" The stationary distribution of the birth-death chain. """
        a = np.zeros(self.dim)
        a[0] = 1.0
        a[1:] = np.cumprod(self.p[0:-1] / self.q[1:])
        mu = a / np.sum(a)
        return mu

    def committor_forward(self, a, b):
        r"""Forward committor for birth-and-death-chain.

        The forward committor is the probability to hit
        state b before hitting state a starting in state x,

        .. math::

            u_x = P_x(T_b<T_a),

        :math:`T_i` is the first arrival time of the chain to state i,

        .. math::

            T_i = \inf ( t>0 \mid X_t=i ).

        Parameters
        ----------
        a : int
            State index
        b : int
            State index

        Returns
        -------
        u : (M,) ndarray
            Vector of committor probabilities.

        """
        u = np.zeros(self.dim)
        g = np.zeros(self.dim - 1)
        g[0] = 1.0
        g[1:] = np.cumprod(self.q[1:-1] / self.p[1:-1])

        """If a and b are equal the event T_b<T_a is impossible
           for any starting state x so that the committor is
           zero everywhere"""
        if a == b:
            return u
        elif a < b:
            """Birth-death chain has to hit a before it can hit b"""
            u[0:a + 1] = 0.0
            """Birth-death chain has to hit b before it can hit a"""
            u[b:] = 1.0
            """Intermediate states are given in terms of sums of g"""
            u[a + 1:b] = np.cumsum(g[a:b])[0:-1] / np.sum(g[a:b])
            return u
        else:
            u[0:b + 1] = 1.0
            u[a:] = 0.0
            u[b + 1:a] = (np.cumsum(g[b:a])[0:-1] / np.sum(g[b:a]))[::-1]
            return u

    def committor_backward(self, a, b):
        r"""Backward committor for birth-and-death-chain.

        The backward committor is the probability for a chain in state x chain to originate from state a instead of
        coming from state b :math:`w_x=P_x(t_a<t_b)`, :math:`t_i` is the last exit time of the chain from state i,
        :math:`t_i = \inf ( t>0 \mid X_{-t} = i )`.

        Parameters
        ----------
        a : int
            State index
        b : int
            State index

        Returns
        -------
        w : (M,) ndarray
            Vector of committor probabilities.

        Notes
        -----
        The birth-death chain is time-reversible

        .. math::

            P(t_a < t_b) = P(T_a < T_b) = 1-P(T_b<T_a),

        therefore we can express the backward comittor probabilities in terms of the
        forward committor probabilities :math:`w=1-u`.
        """
        return 1.0 - self.committor_forward(a, b)

    def flux(self, a, b):
        r"""The flux network for the reaction from
        A=[0,...,a] => B=[b,...,M].

        Parameters
        ----------
        a : int
            State index
        b : int
            State index

        Returns
        -------
        flux : (M, M) ndarray
            Matrix of flux values between pairs of states.

        """
        # if b<=a:
        # raise ValueError("State index b has to be strictly larger than state index a")
        qminus = self.committor_backward(a, b)
        qplus = self.committor_forward(a, b)
        P = self.transition_matrix
        pi = self.stationary_distribution
        from deeptime.markov.tools.flux import flux_matrix
        return flux_matrix(P, pi, qminus, qplus, netflux=False)

    def netflux(self, a, b):
        r"""The netflux network for the reaction from
        A=[0,...,a] => B=[b,...,M].

        Parameters
        ----------
        a : int
            State index
        b : int
            State index

        Returns
        -------
        netflux : (M, M) ndarray
            Matrix of flux values between pairs of states.

        """
        flux = self.flux(a, b)
        from deeptime.markov.tools.flux import to_netflux
        return to_netflux(flux)

    def totalflux(self, a, b):
        r"""The tiotal flux for the reaction A=[0,...,a] => B=[b,...,M].

        Parameters
        ----------
        a : int
            State index
        b : int
            State index

        Returns
        -------
        F : float
            The total flux between reactant and product

        """
        flux = self.flux(a, b)
        A = list(range(a + 1))
        from deeptime.markov.tools.flux import total_flux
        return total_flux(flux, A)

    def rate(self, a, b):
        r""" Yields the total transition rate between state sets `A=[0,...,a]` and `B=[0,...,b]`.

        Parameters
        ----------
        a : int
            State index.
        b : int
            State index.

        Returns
        -------
        kAB : float
            Total transition rate.
        """
        F = self.totalflux(a, b)
        pi = self.stationary_distribution
        qminus = self.committor_backward(a, b)
        kAB = F / (pi * qminus).sum()
        return kAB
