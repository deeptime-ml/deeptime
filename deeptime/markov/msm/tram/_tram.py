import warnings
from typing import Optional
from sklearn.exceptions import ConvergenceWarning

from deeptime.markov import TransitionCountEstimator, TransitionCountModel
from deeptime.markov._base import _MSMBaseEstimator
from deeptime.util import types, callbacks
from ._tram_bindings import tram
from ._tram_model import TRAMModel

import numpy as np
import scipy as sp


def unpack_input_tuple(data):
    """Get input from the data tuple. Data is of variable size.

    Parameters
    ----------
    data : tuple(2) or tuple(3)
        data[0] contains the dtrajs. data[1] the bias matrix, and data[2] may or may not contain the ttrajs.

    Returns
    -------
    dtrajs : array-like(ndarray(n)), int32
        the first element from the data tuple.
    bias_matrices : array-like(ndarray(n,m)), float64
        the second element from the data tuple.
    ttrajs : array-like(ndarray(n)), int32, or None
        the third element from the data tuple, if present.
    """
    dtrajs, bias_matrices, ttrajs = data[0], data[1], data[2:]
    if ttrajs is not None and len(ttrajs) > 0:
        if len(ttrajs) > 1:
            raise ValueError("Unexpected number of arguments in data tuple.")
        ttrajs = ttrajs[0]

    return dtrajs, bias_matrices, ttrajs


def to_zero_padded_array(arrays, desired_shape):
    """Pad a list of numpy arrays with zeros to desired shape. Desired shape should be at least the size of the
    largest np array in the list.
    Example: arrays=[np.array([1,2]), np.array([3,4,5]), np.array([6])], desired_shape=(4)
    Returns: np.array([[1, 2, 0, 0], [3, 4, 5, 0], [6, 0, 0, 0]])

    Parameters
    ----------
    arrays: array-like(ndarray)
        The list/array of numpy arrays of different shapes. All arrays should have the same number of dimensions.
    desired_shape: tuple
        The shape each array should be passed to, having the same number of dimensions as the arrays.

    Returns
    -------
    arrays : ndarray
        The passed arrays as one numpy array.
    """
    new_array = np.zeros((len(arrays), desired_shape))
    for i, array in enumerate(arrays):
        new_array[i, :len(array)] = array
    return new_array


class TRAM(_MSMBaseEstimator):
    r"""Transition(-based) Reweighting Analysis Method.
    TRAM is described in :footcite:`wu2016multiensemble`.

    Parameters
    ----------
    lagtime : int
        Integer lag time at which transitions are counted.
    count_mode : str
        One of "sample", "sliding", "sliding-effective", and "effective".

        * "sample" strides the trajectory with lagtime :math:`\tau` and uses the strided counts as transitions.
        * "sliding" uses a sliding window approach, yielding counts that are statistically correlated and too
          large by a factor of :math:`\tau`; in uncertainty estimation this yields wrong uncertainties.
        * "sliding-effective" takes "sliding" and divides it by :math:`\tau`, which can be shown to provide a
          likelihood that is the geometrical average over shifted subsamples of the trajectory,
          :math:`(s_1,\:s_{tau+1},\:...),\:(s_2,\:t_{tau+2},\:...),` etc. This geometrical average converges to
          the correct likelihood in the statistical limit :footcite:`trendelkamp2015estimation`.
        * "effective" uses an estimate of the transition counts that are statistically uncorrelated.
          Recommended when estimating Bayesian MSMs.
    connectivity : str, optional, default='post_hoc_RE'
        One of 'post_hoc_RE', 'BAR_variance', or 'summed_count_matrix'.
        Defines what should be considered a connected set in the joint (product) space
        of conformations and thermodynamic ensembles.

        * 'post_hoc_RE' : It is required that every state in the connected set can be reached by following a
          pathway of reversible transitions or jumping between overlapping thermodynamic states while staying in
          the same Markov state. A reversible transition between two Markov states (within the same thermodynamic
          state k) is a pair of Markov states that belong to the same strongly connected component of the count
          matrix (from thermodynamic state k). Two thermodynamic states k and l are defined to overlap at Markov
          state n if a replica exchange simulation :footcite:`hukushima1996exchange` restricted to state n would
          show at least one transition from k to l or one transition from from l to k.
          The expected number of replica exchanges is estimated from
          the simulation data. The minimal number required of replica exchanges per Markov state can be increased by
          decreasing `connectivity_factor`.
        * 'BAR_variance' : like 'post_hoc_RE' but with a different condition to define the thermodynamic overlap
          based on the variance of the BAR estimator :footcite:`shirts2008statistically`.
          Two thermodynamic states k and l are defined to overlap
          at Markov state n if the variance of the free energy difference Delta :math:`f_{kl}` computed with BAR (and
          restricted to conformations form Markov state n) is less or equal than one. The minimally required variance
          can be controlled with `connectivity_factor`.
        * 'summed_count_matrix' : all thermodynamic states are assumed to overlap. The connected set is then
          computed by summing the count matrices over all thermodynamic states and taking its largest strongly
          connected set. Not recommended!
    maxiter : int, optional, default=10000
        The maximum number of self-consistent iterations before the estimator exits unsuccessfully.
    maxerr : float, optional, default=1E-15
        Convergence criterion based on the maximal free energy change in a self-consistent
        iteration step.
    track_log_likelihoods : bool, optional, default=False
        If True, the log-likelihood is stored every callback_interval steps. For calculation of the log-likelihood the
        transition matrix needs to be constructed, which will slow down computation. By default, log-likelihoods are
        not computed.
    callback_interval : int, optional, default=0
        Every callback_interval iteration steps, the callback function is calles and error increments are stored. If
        track_log_likelihoods=true, the log-likelihood are also stored. If 0, no call to the callback function is done.
    connectivity_factor : float, optional, default=1.0
        Only needed if connectivity='post_hoc_RE' or 'BAR_variance'. Values greater than 1.0 weaken the connectivity
        conditions. For 'post_hoc_RE' this multiplies the number of hypothetically observed transitions. For
        'BAR_variance' this scales the threshold for the minimal allowed variance of free energy differences.
    progress_bar : object
        Progress bar object that TRAM will call to indicate progress to the user.
        Tested for a tqdm progress bar. Should implement update() and close() and have .total and .desc properties.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
            self, model=None, lagtime=1, count_mode='sliding',
            connectivity='summed_count_matrix',
            maxiter=10000, maxerr: float = 1e-8, track_log_likelihoods=False,
            callback_interval=0,
            connectivity_factor: float = 1.0,
            progress_bar=None):

        super(TRAM, self).__init__()

        self.lagtime = lagtime
        self.count_mode = count_mode
        self._tram_estimator = None

        if connectivity not in TRAM.connectivity_options:
            raise ValueError(f"Connectivity type unsupported. Connectivity must be one of {TRAM.connectivity_options}.")
        self.connectivity = connectivity
        self.connectivity_factor = connectivity_factor
        self.n_markov_states = None
        self.n_therm_states = None
        self.maxiter = maxiter
        self.maxerr = maxerr
        self.track_log_likelihoods = track_log_likelihoods
        self.callback_interval = callback_interval
        self.progress_bar = progress_bar
        self._largest_connected_set = None
        self.log_likelihoods = []
        self.increments = []

        if model is not None:
            self._load_model(model)


    #: All possible connectivity modes
    connectivity_options = ["post_hoc_RE", "BAR_variance", "summed_count_matrix", None]

    @property
    def log_likelihood(self) -> Optional[float]:
        r"""The parameter-dependent part of the TRAM likelihood.

        The definition can be found in :footcite:`wu2016multiensemble`, Equation (9).

        Returns
        -------
        log_likelihood : float
            The parameter-dependent part of the log-likelihood.


        Notes
        -----
        Parameter-dependent, i.e., the factor

        .. math:: \prod_{x \in X} e^{-b^{k(x)}(x)}

        does not occur in the log-likelihood as it is constant with respect to the parameters, leading to

        .. math:: \log \prod_{k=1}^K \left(\prod_{i,j} (p_{ij}^k)^{c_{ij}^k}\right) \left(\prod_{i} \prod_{x \in X_i^k} \mu(x) e^{f_i^k} \right)
        """
        if self._tram_estimator is not None:
            return self._tram_estimator.log_likelihood()

    def fetch_model(self) -> Optional[TRAMModel]:
        r"""Yields the most recent :class:`MarkovStateModelCollection` that was estimated.
        Can be None if fit was not called.

        Returns
        -------
        model : MarkovStateModelCollection or None
            The most recent markov state model or None.
        """
        return self._model

    def fit(self, data, *args, **kw):
        r"""Fit a MarkovStateModelCollection to the given input data, using TRAM.
        For each thermodynamic state, there is one Markov Model in the collection at the respective thermodynamic state
        index.

        Parameters
        ----------
        data: tuple consisting of (dtrajs, bias_matrices) or (dtrajs, bias_matrices, ttrajs).
            * dtrajs: array-like(ndarray(n)), int32
              The discrete trajectories in the form of a list or array of numpy arrays. dtrajs[i] contains one trajectory.
              dtrajs[i][n] contains the Markov state index that the n-th sample from the i-th trajectory was binned
              into. Each of the dtrajs can be of variable length.
            * bias_matrices: ndarray-like(ndarray), float64
              The bias energy matrices. bias_matrices[i, n, l] contains the bias energy of the n-th sample from the i-th
              trajectory, evaluated at thermodynamic state k. The bias energy matrices should have the same size as
              dtrajs in both the 0-th and 1-st dimension. The seconds dimension of of size n_therm_state, i.e. for each
              sample, the bias energy in every thermodynamic state is calculated and stored in the bias_matrices.
            * ttrajs: array-like(ndarray], int32, optional
              ttrajs[i] indicates for each sample in the i-th trajectory what thermodynamic state that sample was
              sampled at. If ttrajs is None, we assume no replica exchange was done. In this case we assume each
              trajectory  corresponds to a unique thermodynamic state, and n_therm_states equals the size of dtrajs.
        """
        # unpack the data tuple, do input validation and check for any replica exchanges.
        ttrajs, dtrajs, bias_matrices = self._preprocess(data)

        dtrajs, state_counts, transition_counts = self._compute_counts_from_largest_connected_set(ttrajs, dtrajs,
                                                                                                  bias_matrices)
        tram_input = tram.TRAMInput(state_counts, transition_counts, dtrajs, bias_matrices)

        if self._tram_estimator is None:
            self._tram_estimator = tram.TRAM(self.n_therm_states, self.n_markov_states)

        self._run_estimation(tram_input)
        self._model = TRAMModel(count_models=self.count_models,
                                transition_matrices=self._tram_estimator.transition_matrices(),
                                biased_conf_energies=self._tram_estimator.biased_conf_energies(),
                                lagrangian_mult_log=self._tram_estimator.lagrangian_mult_log(),
                                modified_state_counts_log=self._tram_estimator.modified_state_counts_log(),
                                therm_state_energies=self._tram_estimator.therm_state_energies(),
                                markov_state_energies=self._tram_estimator.markov_state_energies())

        return self

    def _run_estimation(self, tram_input):
        """ Estimate the free energies using self-consistent iteration as described in the TRAM paper.
        """
        with TRAMCallback(self.progress_bar, self.maxiter, self.log_likelihoods, self.increments,
                          self.callback_interval > 0) as callback:
            self._tram_estimator.estimate(tram_input, self.maxiter, self.maxerr,
                                          track_log_likelihoods=self.track_log_likelihoods,
                                          callback_interval=self.callback_interval, callback=callback)

            if callback.last_increment > self.maxerr:
                warnings.warn(
                    f"TRAM did not converge after {self.maxiter} iteration. Last increment: {callback.last_increment}",
                    ConvergenceWarning)

    def _preprocess(self, data):
        """Unpack the data tuple and validate the containing elements.

        Parameters
        ----------
        data: tuple(2) or tuple(3)
            data[0] contains the dtrajs. data[1] the bias matrix, and data[2] may or may not contain the ttrajs.

        Returns
        -------
        ttrajs : array-like(ndarray(n)), int32, or None
            The validated ttrajs converted to a list of contiguous numpy arrays.
        dtrajs : array-like(ndarray(n)), int32
            The validated dtrajs converted to a list of contiguous numpy arrays.
        bias_matrices : array-like(ndarray(n,m)), float64
            The validated bias matrices converted to a list of contiguous numpy arrays.
        """
        dtrajs, bias_matrices, ttrajs = unpack_input_tuple(data)

        ttrajs, dtrajs, bias_matrices = self._validate_input(ttrajs, dtrajs, bias_matrices)

        return ttrajs, dtrajs, bias_matrices

    def _validate_input(self, ttrajs, dtrajs, bias_matrices):
        """ Check type and shape of input to ensure it can be handled by the _tram_bindings module.
        The ttrajs, dtrajs and bias matrices should all contain the same number of trajectories (i.e. shape(0) should be
        equal for all input arrays).
        Trajectories can vary in length, but for each trajectory the corresponding arrays in dtrajs, ttrajs and
        bias_matrices should be of the same length (i.e. array[i].shape(0) for any i should be equal in all input
        arrays).

        Parameters
        ----------
        ttrajs: array-like(ndarray(n)), int32, optional
            ttrajs[i] indicates for each sample in the i-th trajectory what thermodynamic state that sample was sampled
            at. If ttrajs is None, we assume no replica exchange was done. We then assume each trajectory corresponds to
            a unique thermodynamic state, and assign that trajectory index to the thermodynamic state index.
        dtrajs: array-like(ndarray(n)), int32
            The discrete trajectories. dtrajs[i, n] contains the Markov state index for the n-th sample in the i-th
            trajectory.
        bias_matrices: array-like(ndarray(n,m)), float64
            The bias energy matrices. bias_matrices[i, n, l] contains the bias energy of the n-th sample from the i'th
            trajectory, evaluated at thermodynamic state k.

        Returns
        -------
        ttrajs : list(ndarray(n)), int32
            The validated ttrajs converted to a list of contiguous numpy arrays.
        dtrajs : list(ndarray(n)), int32
            The validated dtrajs converted to a list of contiguous numpy arrays.
        bias_matrices : List(ndarray(n,m)), float64
            The validated bias matrices converted to a list of contiguous numpy arrays.
        """
        # shape and type checks
        if len(dtrajs) != len(bias_matrices):
            raise ValueError("Number of trajectories is not equal to the number of bias matrices.")
        for d in dtrajs:
            types.ensure_integer_array(d, ndim=1)
        for b in bias_matrices:
            types.ensure_floating_array(b, ndim=2)

        if self.n_markov_states is None:
            # No model was previously loaded. Find the number of Markov states.
            self.n_markov_states = max(np.max(d) for d in dtrajs) + 1
        else:
            # a model was already loaded. check that new dtrajs do not exceed of the number of markov states.
            if max(np.max(d) for d in dtrajs) >= self.n_markov_states:
                raise ValueError(
                    "dtrajs out of bounds. Largest state number in dtrajs is larger than the number of Markov states.")

        if ttrajs is None or len(ttrajs) == 0:
            # ensure ttrajs is None. Leaving it an empty tuple will break the call to _tram_bindings
            ttrajs = None
        else:
            # find the number of therm states as the highest index in ttrajs
            for t in ttrajs:
                types.ensure_integer_array(t, ndim=1)

            ttrajs = [np.require(t, dtype=np.int32, requirements='C') for t in ttrajs]

        if self.n_therm_states is None:
            # No model was previously loaded. Find the number of thermodynamic states.
            if ttrajs is None:
                self.n_therm_states = len(dtrajs)
            else:
                self.n_therm_states = max(np.max(t) for t in ttrajs) + 1

        else:
            # A model was already loaded. check that new ttrajs (if any) do not exceed of the number of thermodynamic states.
            if ttrajs is not None and max(np.max(t) for t in ttrajs) >= self.n_therm_states:
                raise ValueError(
                    "ttrajs out of bounds. Largest state number in ttrajs is larger than the number of thermodynamic states.")

        # cast types and change axis order if needed
        dtrajs = [np.require(d, dtype=np.int32, requirements='C') for d in dtrajs]
        bias_matrices = [np.require(b, dtype=np.float64, requirements='C') for b in bias_matrices]

        # dimensionality checks
        for i, (d, b) in enumerate(zip(dtrajs, bias_matrices)):
            if d.shape[0] != b.shape[0]:
                raise ValueError(f"discrete trajectory {i} and bias matrix {i} should be of equal length.")

            if b.shape[1] != self.n_therm_states:
                raise ValueError(
                    f"Second dimension of bias matrix {i} should be of size n_therm_states (={self.n_therm_states})")

        # If we were given ttrajs, do the same checks for those.
        if ttrajs is not None:
            if len(ttrajs) != len(dtrajs):
                raise ValueError("number of ttrajs is not equal to number of dtrajs.")

            for i, (t, d) in enumerate(zip(ttrajs, dtrajs)):
                if t.shape[0] != d.shape[0]:
                    raise ValueError(f"ttraj {i} and dtraj {i} should be of equal length.")

        return ttrajs, dtrajs, bias_matrices

    def _compute_counts_from_largest_connected_set(self, ttrajs, dtrajs, bias_matrices):
        """Find the largest connected set of Markov states over all thermodynamic states, given the input data and
         connectivity settings. Restrict the input data to the connected set and return state counts and transition
         counts for the restricted input data.

         Parameters
         ----------
         dtrajs: array-like(ndarray(n)), int32
            The discrete trajectories in the form an 2-d integer ndarray. dtrajs[i] contains one trajectory.
            dtrajs[i][n] contains the Markov state index that the n-th sample from the i-th trajectory was binned
            into. Each of the dtrajs can be of variable length.
         bias_matrices: array-like(ndarray(n,m)), float64
            The bias energy matrices. bias_matrices[i, n, l] contains the bias energy of the n-th sample from the i-th
            trajectory, evaluated at thermodynamic state k. The bias energy matrices should have the same size as
            dtrajs in both the 0-th and 1-st dimension. The seconds dimension of of size n_therm_state, i.e. for each
            sample, the bias energy in every thermodynamic state is calculated and stored in the bias_matrices.
         ttrajs: array-like(ndarray(n)), int32, optional
            ttrajs[i] indicates for each sample in the i-th trajectory what thermodynamic state that sample was
            sampled at. If ttrajs is None, we assume no replica exchange was done. In this case we assume each
            trajectory  corresponds to a unique thermodynamic state, and n_therm_states equals the size of dtrajs.

        Returns
        -------
        dtrajs : list(ndarray), int32
            The dtrajs restricted to the largest connected set. All sample indices that do not belong to the largest
            connected set are set to -1.
        state_counts : ndarray(n,m), int32
            The state counts for the dtrajs, restricted to the connected set. state_counts[k,i] is the number of samples
            sampled at thermodynamic state k that are in Markov state i.
        transition_counts : ndarray(n,m,m), int32
            The transition counts for the dtrajs, restricted to the connected set. transition_counts[k,i,j] is the
            number of observed transitions from Markov state i to Markov state j, in thermodynamic state k under the
            chosen lagtime.
        """
        # count all transitions and state counts, without restricting to connected sets
        self._largest_connected_set = self._find_largest_connected_set(ttrajs, dtrajs, bias_matrices)

        # get rid of any state indices that do not belong to the largest connected set
        dtrajs = self._restrict_to_connected_set(dtrajs)

        # In the case of replica exchange: get all trajectory fragments (also removing any negative state indices).
        # the dtraj_fragments[k] contain all trajectories within thermodynamic state k, starting a new trajectory at
        # each replica exchange swap. If there was no replica exchange, dtraj_fragments == dtrajs.
        dtraj_fragments = self._find_trajectory_fragments(dtrajs, ttrajs)

        # ... convert those into count matrices.
        state_counts, transition_counts = self._construct_count_models(dtraj_fragments)

        return dtrajs, state_counts, transition_counts

    def _find_largest_connected_set(self, ttrajs, dtrajs, bias_matrices):
        """ Find the largest connected set of Markov states based on the connectivity settings and the input data.
        A full counts model is first calculated from all dtrajs. Then, the connected set is computed based on
        self.connectivity and self.connectivity_factor. The full counts model is then reduces to the connected set and
        returned. If the connectivity setting is unknown, the full counts model is returned.

        Parameters
         ----------
         dtrajs: array-like(ndarray(n)), int32
            The discrete trajectories in the form an 2-d integer ndarray. dtrajs[i] contains one trajectory.
            dtrajs[i][n] contains the Markov state index that the n-th sample from the i-th trajectory was binned
            into. Each of the dtrajs can be of variable length.
         bias_matrices: array-like(ndarray(n,m)), float64
            The bias energy matrices. bias_matrices[i, n, l] contains the bias energy of the n-th sample from the i-th
            trajectory, evaluated at thermodynamic state k. The bias energy matrices should have the same size as
            dtrajs in both the 0-th and 1-st dimension. The seconds dimension of of size n_therm_state, i.e. for each
            sample, the bias energy in every thermodynamic state is calculated and stored in the bias_matrices.
         ttrajs: array-like(ndarray(n)), int32, optional
            ttrajs[i] indicates for each sample in the i-th trajectory what thermodynamic state that sample was
            sampled at. If ttrajs is None, we assume no replica exchange was done. In this case we assume each
            trajectory  corresponds to a unique thermodynamic state, and n_therm_states equals the size of dtrajs.

        Returns
        -------
        counts_model : MarkovStateModel
            The counts model pertaining to the largest connected set.
        """
        estimator = TransitionCountEstimator(lagtime=self.lagtime, count_mode=self.count_mode)

        # make a counts model over all observed samples.
        full_counts_model = estimator.fit_fetch(dtrajs)

        if self.connectivity is None:
            warnings.warn(f"connectivity type is None. Data has not been restricted to the largest connected set."
                          f"The full counts model has been returned.")
            return full_counts_model

        if self.connectivity == 'summed_count_matrix':
            # We assume the thermodynamic states have overlap when they contain counts from the same markov state.
            # Full counts model contains the sum of the state counts over all therm. states., and we simply ignore
            # the thermodynamic state indices.
            return full_counts_model.submodel_largest(
                connectivity_threshold=self.connectivity_factor,
                directed=True)

        if self.connectivity in ['post_hoc_RE', 'BAR_variance']:
            # get state counts for each trajectory (=for each therm. state)
            all_state_counts = np.asarray([estimator.fit_fetch(dtraj).state_histogram for dtraj in dtrajs],
                                          dtype=object)
            # pad with zero's so they are all the same size and easier for the cpp module to handle
            all_state_counts = to_zero_padded_array(all_state_counts, self.n_markov_states)

            # get list of all possible transitions between thermodynamic states. A transition is only possible when two
            # thermodynamic states have an overlapping markov state. Whether the markov state overlaps depends on the
            # sampled data and the connectivity settings and is computed in find_state_transitions:
            if self.connectivity == 'post_hoc_RE':
                connectivity_fn = tram.find_state_transitions_post_hoc_RE
            else:
                connectivity_fn = tram.find_state_transitions_BAR_variance

            with callbacks.Callback(self.progress_bar, self.n_therm_states * self.n_markov_states,
                                    "Finding connected sets") as callback:
                (i_s, j_s) = connectivity_fn(ttrajs, dtrajs, bias_matrices, all_state_counts, self.n_therm_states,
                                             self.n_markov_states, self.connectivity_factor, callback)

            # add transitions that occurred within each thermodynamic state. These are simply the connected sets:
            for k in range(self.n_therm_states):
                for cset in estimator.fit_fetch(dtrajs[k]).connected_sets():
                    i_s.extend(list(cset[0:-1] + k * self.n_markov_states))
                    j_s.extend(list(cset[1:] + k * self.n_markov_states))

            # turn the list of transitions into a boolean matrix that has a one whenever a transition has occurred
            data = np.ones(len(i_s), dtype=np.int32)
            dim = self.n_therm_states * self.n_markov_states
            sparse_transition_counts = sp.sparse.coo_matrix((data, (i_s, j_s)), shape=(dim, dim))

            # Now we have all possible paths in the list of transitions. Get the connected set of that
            overlap_counts_model = TransitionCountModel(sparse_transition_counts)
            connected_states_ravelled = overlap_counts_model.submodel_largest(directed=False).state_symbols

            # unravel the index and combine all separate csets to one cset
            connected_states = np.unravel_index(connected_states_ravelled, (self.n_therm_states, self.n_markov_states),
                                                order='C')

            return full_counts_model.submodel(np.unique(connected_states[1]))

    def _restrict_to_connected_set(self, dtrajs):
        """         Restrict the count matrices and dtrajs to the connected set. All dtraj samples not in the largest
        connected set will be set to -1.

        Parameters
        ----------
        dtrajs : array-like(ndarray(n)), int32
            the discrete trajectories. dtrajs[i] is a numpy array containing the trajectories sampled in the i-th
            thermodynamic state. dtrajs[i][n] is the Markov state that the n-th sample sampled at thermodynamic state i
            falls in.

        Returns
        -------
        dtrajs_connected : list(ndarray(n)), int32
            The list of discrete trajectories. Identical to the input dtrajs, except that all samples that do not belong
            in the largest connected set are set to -1.

        """
        dtrajs_connected = []

        for k in range(self.n_therm_states):
            # Get largest connected set
            # Assign -1 to all indices not in the submodel.
            restricted_dtraj = self._largest_connected_set.transform_discrete_trajectories_to_submodel(dtrajs[k])

            # The transformation has converted all indices to the state indices of the submodel. We want the original
            # indices. Convert the newly assigned indices back to the original state symbols
            restricted_dtraj = self._largest_connected_set.states_to_symbols(restricted_dtraj)

            dtrajs_connected.append(restricted_dtraj)

        return dtrajs_connected

    def _find_trajectory_fragments(self, dtrajs, ttrajs):
        """ Find all trajectory fragments given the discrete trajectories and the thermodynamic state indices of each
        sample. Get rid of any negative state indices.
        If no replica exchange was done, this will simply return all positive values from dtrajs. If ttrajs are
        given, we assume replica exchange was done and restructure dtrajs in such a way that every replica-exchange
        swap is the start of a new trajectory.

        Parameters
        ----------
        dtrajs: array-like(ndarray(n)), int32
            the discrete trajectories. dtrajs[i] is a numpy array containing the trajectories sampled in the i-th
            thermodynamic state. dtrajs[i][n] is the Markov state that the n-th sample sampled at thermodynamic state i
            falls in. May contain negative state indices.

        ttrajs: list(ndarray(n)), int32, optional
            ttrajs[i] indicates for each sample in the i-th trajectory what thermodynamic state that sample was sampled
            at.

        Returns
        -------
        dtraj_fragments : list(List(int32))
           A list that contains for each thermodynamic state the fragments from all trajectories that were sampled at
           that thermodynamic state. fragment_indices[k][i] defines the i-th fragment sampled at thermodynamic state k.
           The fragments are be restricted to the largest connected set and negative state indices are excluded.
        """
        if ttrajs is None or len(ttrajs) == 0:
            # No ttrajs were given. We assume each trajectory in dtrajs was sampled in a distinct thermodynamic state.
            # The thermodynamic state index equals the trajectory index, and the dtrajs are unchanged.
            return [[dtrajs[k][dtrajs[k] >= 0]] for k in range(self.n_therm_states)]

        # replica exchange data means that the trajectories to not correspond 1:1 to thermodynamic states.
        # get a mapping from trajectory segments to thermodynamic states
        fragment_indices = self._find_trajectory_fragment_mapping(ttrajs)

        fragments = []
        # for each them. state k, gather all trajectory fragments that were sampled at that state.
        for k in range(self.n_therm_states):
            # take the fragments based on the list of indices. Exclude all values that are less than zero. They don't
            # belong in the connected set.
            fragments.append([dtrajs[traj_idx][start:stop][dtrajs[traj_idx][start:stop] >= 0]
                              for (traj_idx, start, stop) in fragment_indices[k]])
        return fragments

    def _find_trajectory_fragment_mapping(self, ttrajs):
        """ define mapping that gives for each trajectory the slices that make up a trajectory inbetween RE swaps.
            At every RE swap point, the trajectory is sliced, so that swap point occurs as a trajectory start.

        Parameters
        ----------
        ttrajs: array-like(ndarray(n)), int32
            the thermodynamic state sequences.

        Returns
        -------
        fragment_indices : lst(list(tuple(int)))
            A list that contains for each thermodynamic state the fragments from all trajectories that were sampled at
            that thermodynamic state.
            fragment_indices[k][i] defines the i-th fragment sampled at thermodynamic state k. The tuple consists of
            (traj_idx, start, stop) where traj_index is the index of the trajectory the fragment can be found at, start
            is the start index of the trajectory, and stop the end index (exclusive) of the fragment.

        """
        return tram.find_trajectory_fragment_indices(ttrajs, self.n_therm_states)

    def _construct_count_models(self, dtraj_fragments):
        """ Construct a TransitionCountModel for each thermodynamic state based on the dtraj_fragments, and store in
        self.count_models.
        Based on each TransitionCountModel, construct state_count and transition_count matrices that contain the counts
        for each thermodynamic state. These are reshaped to contain all possible markov states, even the once without
        counts. This is done so that the _tram_bindings receives count matrices that are all the same shape, which is
        easier to handle (matrices are padded with zeros for all empty states that got dropped by the
        TransitionCountModels).

        Parameters
        ----------
        dtraj_fragments: list(list(int32))
           A list that contains for each thermodynamic state the fragments from all trajectories that were sampled at
           that thermodynamic state. fragment_indices[k][i] defines the i-th fragment sampled at thermodynamic state k.
           The fragments should be restricted to the largest connected set and not contain any negative state indices.

        Returns
        -------
        transition_counts : ndarray(n, m, m)
            The transition counts matrices. transition_counts[k] contains the transition counts for thermodynamic state
            k, restricted to the largest connected set of state k.
        state_counts : ndarray(n, m)
            The state counts histogram. state_counts[k] contains the state histogram for thermodynamic state k,
            restricted to the largest connected set of state k.
        """

        estimator = TransitionCountEstimator(lagtime=self.lagtime, count_mode=self.count_mode)

        transition_counts = np.zeros((self.n_therm_states, self.n_markov_states, self.n_markov_states), dtype=np.intc)
        state_counts = np.zeros((self.n_therm_states, self.n_markov_states), dtype=np.intc)

        self.count_models = []
        for k in range(self.n_therm_states):

            if len(dtraj_fragments[k]) == 0 or np.all([len(frag) <= self.lagtime for frag in dtraj_fragments[k]]):
                warnings.warn(f"No transitions for thermodynamic state {k} after cutting the trajectories into "
                              f"fragments that start at each replica exchange swap. Replica exchanges possibly occur"
                              f"within the span of the lag time.")
                # there are no samples from this state that belong to the connected set. Make an empty count model.
                self.count_models.append(TransitionCountModel(np.zeros(self.n_markov_states, self.n_markov_states)))
            else:
                # make a counts model for the samples that belong to the connected set.
                traj_counts_model = estimator.fit_fetch(dtraj_fragments[k])
                # create index for all elements in transition matrix
                i_s, j_s = np.meshgrid(traj_counts_model.state_symbols, traj_counts_model.state_symbols)

                # place submodel counts in our full-sized count matrices
                transition_counts[k, i_s, j_s] = traj_counts_model.count_matrix.T
                state_counts[k, traj_counts_model.state_symbols] = traj_counts_model.state_histogram

                self.count_models.append(traj_counts_model)
        return state_counts, transition_counts

    def _load_model(self, model):
        self._model = model
        self.n_therm_states = model.n_therm_states
        self.n_markov_states = model.n_markov_states
        self._tram_estimator = tram.TRAM(model.biased_conf_energies, model.lagrangian_mult_log,
                                         model.modified_state_counts_log)


class TRAMCallback(callbacks.Callback):
    """Callback for the TRAM estimate process. Increments a progress bar and optionally saves iteration increments and
    log likelihoods to a list."""

    def __init__(self, progress_bar, n_iter, log_likelihoods_list=None, increments=None, store_convergence_info=False):
        super().__init__(progress_bar, n_iter, "Running TRAM estimate")
        self.log_likelihoods = log_likelihoods_list
        self.increments = increments
        self.save_convergence_info = store_convergence_info
        self.last_increment = 0

    def __call__(self, increment, log_likelihood):
        super().__call__()

        if self.store_convergence_info:
            if self.log_likelihoods is not None:
                self.log_likelihoods.append(log_likelihood)
            if self.increments is not None:
                self.increments.append(increment)

        self.last_increment = increment
