r"""This module implements an estimator for Girsanov path reweighting for Markov state models.
.. moduleauthor:: J.-L.Schaefer <joana-lysiane DOT schaefer AT fu-berlin DOT de>
"""

from typing import Optional, List
import numpy as np
from scipy.sparse import issparse
from .tools import estimation as msmest
from ._transition_counting import TransitionCountModel, TransitionCountEstimator
from ..util.types import ensure_dtraj_list, ensure_factors_list


class GirsanovReweightingEstimator(TransitionCountEstimator):

    def __init__(self, lagtime: int, count_mode: str, n_states=None, sparse=False):
        super().__init__(lagtime=lagtime, count_mode=count_mode,  n_states=n_states, sparse=sparse)
        self.lagtime = lagtime 
        self.count_mode = count_mode
        self.sparse = sparse
        self.n_states = n_states

    def fetch_model(self) -> Optional[TransitionCountModel]:
        r"""
        Yields the latest estimated :class:`TransitionCountModel`. Might be None if fetched before any data was fit.
        Returns
        -------
        The latest :class:`TransitionCountModel` or None.
        """
        return self._model

    def fit(self, data, reweighting_factors, *args, **kw):  
        r""" Counts transitions at given lag time according to configuration of the estimator.
        Parameters
        ----------
        data : array_like or list of array_like
            discretized trajectories; check for same length of random number array :code:`eta`, 
            discrete trajectory and reweighting factors; note, most integrator give trajectories of length
        reweighting_factors: tuple 
            tuple of reweighting factors :code:`(g,M)`
             :code:`g` is the likelihood ratio between probability measures with :code:`dim=(len(dtraj)-lag-1)`. 
             :code:`M` is the likelihood ratio between the path probabilitiy densities with :code:`dim=(len(dtraj)-lag-1)`. 
        """
        dtrajs = ensure_dtraj_list(data)

        # Compute count matrix
        count_matrix = GirsanovReweightingEstimator.count(self.count_mode , dtrajs, self.lagtime, 
                                                      reweighting_factors=reweighting_factors,
                                                      sparse=self.sparse)
        
        # basic count statistics like in deeptime._transition_counting.TransitionCountEstimator
        from deeptime.markov import count_states
         
        histogram = count_states(dtrajs, ignore_negative=True) 
        
        if self.n_states is not None and self.n_states > count_matrix.shape[0]:
            histogram = np.pad(histogram, pad_width=[(0, self.n_states - count_matrix.shape[0])])
            if issparse(count_matrix):
                count_matrix = scipy.sparse.csr_matrix((count_matrix.data, count_matrix.indices, count_matrix.indptr),
                                                       shape=(self.n_states, self.n_states))
            else:
                n_pad = self.n_states - count_matrix.shape[0]
                count_matrix = np.pad(count_matrix, pad_width=[(0, n_pad), (0, n_pad)])

        self._model = TransitionCountModel(
            count_matrix=count_matrix, counting_mode=self.count_mode, lagtime=self.lagtime, 
            state_histogram=histogram
        )
        return self

    @staticmethod
    def count(count_mode: str, dtrajs: List[np.ndarray], lagtime: int, reweighting_factors: tuple, 
              sparse: bool = False):
        r""" Computes a reweighted count matrix according to Girsanov path reweighting for Markov state models 
        based on the sliding mode, discrete trajectories, a lagtime, the precomputed reweighting factors and
        whether to use sparse matrices.
        Parameters
        ----------
        count_mode : str
            The counting mode to be used so far is "sliding".
            See :meth:`__init__` for a more detailed description.
        dtrajs : array_like or list of array_like
            Discrete trajectories, i.e., a list of arrays which contain non-negative integer values. A single ndarray
            can also be passed, which is then treated as if it was a list with that one ndarray in it.
        lagtime : int
            Distance between two frames in the discretized trajectories under which their potential change of state
            is considered a transition.
        sparse : bool, default=False
            Whether to use sparse matrices or dense matrices. Sparse matrices can make sense when dealing with a lot of
            states.
        Returns
        -------
        count_matrix : (N, N) ndarray or sparse array
            The computed count matrix. Can be ndarray or sparse depending on whether sparse was set to true or false.
            N is the number of encountered states, i.e., :code:`np.max(dtrajs)+1`.
        Example
        -------
        >>> from deeptime.markov import GirsanovReweightingEstimator
        >>> dtrajs = np.array([0,0,1,1,0,1,0,1,1])
        >>> _reweighting = (np.array([1,1,1,1,1,1]),np.array([1,1,1,1,1,1]))
        >>> reweighted_counts_estimator = GirsanovReweightingEstimator(lagtime=1,
        ...                                                            count_mode='sliding')
        >>> reweighted_counts = reweighted_counts_estimator.fit(dtrajs[:-1],
        ...                                                     reweighting_factors=_reweighting).fetch_model()
        >>> np.testing.assert_equal(reweighted_counts.count_matrix, np.array([[1, 2], [1, 2]]))
        >>> print(reweighted_counts.count_matrix)
        """ 
        if count_mode == 'sliding':
            reweighting_factors=(ensure_factors_list(reweighting_factors[0]),ensure_factors_list(reweighting_factors[1]))
            count_matrix = msmest.girsanov_reweighted_count_matrix(dtrajs, lagtime, reweighting_factors, 
                                                                   sliding=True, sparse_return=sparse)
        elif count_mode in {'sample','effective','sliding-effective'}:
            raise ValueError('Count mode {} is not compatible with the Girsanov reweighting estimator.'.format(count_mode))
        else:
            raise ValueError('Count mode {} is unknown.'.format(count_mode))
        return count_matrix
