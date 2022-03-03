import numpy as np
import scipy as sp
import warnings

from deeptime.base import Dataset
from deeptime.util import types, callbacks
from deeptime.util.decorators import cached_property
from deeptime.markov import TransitionCountEstimator, TransitionCountModel
from ._tram_bindings import tram


def _determine_n_states(dtrajs):
    return max(np.max(d) for d in dtrajs) + 1


def _determine_n_therm_states(dtrajs, ttrajs):
    if ttrajs is None:
        return len(dtrajs)
    else:
        return _determine_n_states(ttrajs)


def _split_at_negative_state_indices(trajectory_fragment, negative_state_indices):
    split_fragments = np.split(trajectory_fragment, negative_state_indices)
    sub_fragments = []
    # now get rid of the negative state indices.
    for frag in split_fragments:
        frag = frag[frag >= 0]
        # Only add to the list if there are any samples left in the fragments
        if len(frag) > 0:
            sub_fragments.append(frag)
    return sub_fragments


def transition_counts_from_count_models(n_therm_states, n_markov_states, count_models):
    transition_counts = np.zeros((n_therm_states, n_markov_states, n_markov_states), dtype=np.int32)

    for k in range(n_therm_states):
        model_k = count_models[k]
        if model_k.count_matrix.sum() > 0:
            i_s, j_s = np.meshgrid(model_k.state_symbols, model_k.state_symbols, indexing='ij')
            # place submodel counts in our full-sized count matrices
            transition_counts[k, i_s, j_s] = model_k.count_matrix

    return transition_counts


def state_counts_from_count_models(n_therm_states, n_markov_states, count_models):
    state_counts = np.zeros((n_therm_states, n_markov_states), dtype=np.int32)

    for k in range(n_therm_states):
        model_k = count_models[k]
        if model_k.count_matrix.sum() > 0:
            state_counts[k, model_k.state_symbols] = model_k.state_histogram

    return state_counts


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


def _invalidate_caches():
    r""" Invalidates all cached properties and causes them to be re-evaluated """
    for member in TRAMDataset.__dict__.values():
        if isinstance(member, cached_property):
            member.invalidate()


class TRAMDataset(Dataset):
    r""" Dataset for organizing data and obtaining properties from data that are needed for TRAM.
    The minimum required parameters for constructing a TRAMDataset are the `dtrajs` and `bias_matrices`. In this case,
    `ttrajs` are inferred from the shape of the `dtrajs`, by assuming each trajectory in `dtrajs` corresponds to a
    unique thermodynamic state, with the index corresponding to the index of occurrence in `dtrajs`.

    The values at identical indices in `dtrajs`, `ttrajs` and `bias_matrices` correspond to the sample. For example, at
    indices `(i, n)` we find information about the :math:`n`-th sample in trajectory :math:`i`. `dtrajs[i][n]` gives us
    the index of the Markov state the sample falls into. `ttrajs[i][n]` gives us the thermodynamic state the sample was
    sampled at (which may be different from other samples in the trajectory due to a replica exchange swap occurring at
    index :math:`n`). Finally, `bias_matrices[i][n]` gives us for each thermodynamic state, the energy of the sample
    evaluated at that thermodynamic state. In other words: `bias_matrices[i][n][k]` gives us :math:`b^k(x_i^n)`, i.e.
    the bias energy as if the sample were observed in thermodynamic state :math:`k`.

    Parameters
    ----------
    dtrajs : array-like(ndarray(n)), int
        The discrete trajectories in the form of a list or array of numpy arrays. `dtrajs[i]` contains one trajectory.
        `dtrajs[i][n]` contains the Markov state index that the :math:`n`-th sample from the :math:`i`-th trajectory was
        binned into. Each of the dtrajs can be of variable length.
    bias_matrices : array-like(ndarray(n,m)), float
        The bias energy matrices. `bias_matrices[i][n, k]` contains the bias energy of the :math:`n`-th sample from the
        :math:`i`-th trajectory, evaluated at thermodynamic state :math:`k`, i.e. :math:`b^k(x_{i,n})`. The bias energy
        matrices should have the same size as `dtrajs` in both the first and second dimension. The third dimension is of
        size `n_therm_state`, i.e. for each sample, the bias energy in every thermodynamic state is calculated and
        stored in the `bias_matrices`.
    ttrajs : array-like(ndarray(n)), int, optional
        `ttrajs[i]` contains for each sample in the :math:`i`-th trajectory the index of the thermodynamic state that
        sample was sampled at. If `ttrajs = None`, we assume no replica exchange was done. In this case we assume each
        trajectory  corresponds to a unique thermodynamic state, and `n_therm_states` equals the size of `dtrajs`.
    n_therm_states : int, optional
        if `n_therm_states` is given, the indices in `ttrajs` are checked to lie within `n_therm_states` bound.
        Otherwise, `n_therm_states` are inferred from the highest occurring index in `ttrajs`.
    n_markov_states : int, optional
        if `n_markov_states` is given, the indices in `dtrajs` are checked to lie within `n_markov_states` bound.
        Otherwise, `n_markov_states` are inferred from the highest occurring index in `dtrajs`.
    lagtime : int, optional, default=1
        Integer lag time at which transitions are counted.
    count_mode : str, optional, default='sliding'
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

    See Also
    --------
    :class:`TransitionCountEstimator <deeptime.markov.TransitionCountEstimator>`,
    :class:`TRAM <deeptime.markov.msm.TRAM>`, :class:`TRAMModel <deeptime.markov.msm.TRAMModel>`

    """
    def __init__(self, dtrajs, bias_matrices, ttrajs=None, n_therm_states=None, n_markov_states=None, lagtime=1,
                 count_mode='sliding'):
        self.lagtime = lagtime
        self.count_mode = count_mode
        self.dtrajs = dtrajs
        self.ttrajs = ttrajs
        self.bias_matrices = bias_matrices
        self.count_models = []

        self._ensure_correct_data_types()

        if n_therm_states is None:
            self._n_therm_states = _determine_n_therm_states(self.dtrajs, self.ttrajs)
        else:
            self._n_therm_states = n_therm_states

        if n_markov_states is None:
            self._n_markov_states = _determine_n_states(self.dtrajs)
        else:
            self._n_markov_states = n_markov_states

        # validate all dimensions
        self._check_dimensions()
        # compute the count models
        self._compute_counts()

    #: All possible connectivity modes
    connectivity_options = ["post_hoc_RE", "BAR_variance", "summed_count_matrix", None]

    def __len__(self):
        return np.sum([len(traj) for traj in self.dtrajs])

    def __getitem__(self, indices):
        traj, n = indices
        return self.dtrajs[traj][n], self.bias_matrices[traj][n]

    def setflags(self, write=True):
        [traj.setflags(write=write) for traj in self.dtrajs]
        [bias_matrix.setflags(write=write) for bias_matrix in self.bias_matrices]

    @property
    def tram_input(self):
        r""" The TRAMInput object containing the data needed for estimation.
        The data is restructured to allow parallelization over Markov states. The dtrajs are only used to see
        which Markov state the sample biases belong to. Ordering of the data doesn't matter (transition information is
        already stored in the count matrices) so we can restructure the data.
        After restructuring, each bias matrix in the bias_list corresponds to all sample biases for samples that fell
        into the Markov state of which the index corresponds to the index of the bias matrix in the list.
        Or: bias_list[i] contains biases for all samples that fell in Markov state i. """
        bias_list = []
        for markov_state in range(self.n_markov_states):
            biases = []
            for dtraj, bias_matrix in zip(self.dtrajs, self.bias_matrices):
                indices = np.where(dtraj == markov_state)[0]
                biases.append(bias_matrix[indices])

            bias_list.append(np.concatenate(biases))
        return tram.TRAMInput(self.state_counts, self.transition_counts, bias_list)

    @property
    def n_therm_states(self):
        r""" The number of thermodynamic states. """
        return self._n_therm_states

    @property
    def n_markov_states(self):
        r""" The number of Markov states. """
        return self._n_markov_states

    @cached_property
    def transition_counts(self):
        r"""
        The transition counts matrices. `transition_counts[k]` contains the transition counts for thermodynamic state
        :math:`k`, based on the TransitionCountModel of state :math:`k`. `transition_counts[k][i][j]` equals the number
        of observed transitions from Markov state :math:`i` to Markov state :math:`j`, in thermodynamic state :math:`k`.

        The transition counts for every thermodynamic state are shaped to contain all possible markov states, even the
        ones without counts in that thermodynamic state. This is done so that the underlying c++ estimator receives
        count matrices that are all the same shape, which is easier to handle (matrices are padded with zeros for all
        empty states that got  dropped by the TransitionCountModels).

        :getter: the transition counts
        :type: ndarray(n, m, m)
        """
        return transition_counts_from_count_models(self.n_therm_states, self.n_markov_states, self.count_models)

    @cached_property
    def state_counts(self):
        r""" ndarray(n, m)
        The state counts histogram. `state_counts[k]` contains the state histogram for thermodynamic state :math:`k`,
        based on the TransitionCountModel of state :math:`k`. `state_counts[k][i]``  equals the number of samples that
        fall into Markov state :math:`i`, sampled at thermodynamic state :math:`k`.

        The state counts for every thermodynamic state are shaped to contain all possible markov states, even the ones
        without counts in that thermodynamic state. This is done so that the underlying c++ estimator receives count
        matrices that are all the same shape, which is easier to handle (matrices are padded with zeros for all empty
        states that got dropped by the TransitionCountModels).
        """
        return state_counts_from_count_models(self.n_therm_states, self.n_markov_states, self.count_models)

    def check_against_model(self, model):
        r""" Check the number of thermodynamic states of the model against that of the dataset. The number of
        thermodynamic states in the dataset have to be smaller than or equal to those of the model, otherwise the
        `ttrajs` and/or `dtrajs` would be out of bounds.

        Parameters
        ----------
        model : TRAMModel
            The model to check the data against.

        Raises
        ------
        ValueError
            If the data encompasses more Markov and/or thermodynamic states than the model.
        """
        if model.n_therm_states < self._n_therm_states:
            if self.ttrajs is None:
                msg = "dtrajs are out of bounds of the model. " \
                      "There are more dtrajs then there are thermodynamic states. Provide ttrajs " \
                      "or increase the number of thermodynamic states."
            else:
                msg = "ttrajs are out of bounds of the model. " \
                      "Largest thermodynamic state index in ttrajs would be larger than the number of thermodynamic " \
                      "states in the model."
            raise ValueError(msg)
        if model.n_markov_states < self._n_markov_states:
            raise ValueError(
                "dtrajs are out of bounds of the model."
                "Largest Markov state index in dtrajs would be larger than the number of Markov states in the model.")

    def restrict_to_largest_connected_set(self, connectivity='post_hoc_RE', connectivity_factor=1.0, progress=None):
        r""" Find the largest connected set of Markov states based on the connectivity settings and the input data, and
        restrict the input data to this connected set.
        The largest connected set is computed based on the data, and self.connectivity and self.connectivity_factor.
        The data is then restricted to the largest connected set and count models are recomputed.

        Parameters
        ----------
        connectivity : str, optional, default="post_hoc_RE"
            One of None, "post_hoc_RE", "BAR_variance", or "summed_count_matrix".
            Defines what should be considered a connected set in the joint (product) space of conformations and
            thermodynamic ensembles. If connectivity=None, the data is not restricted.

            * "post_hoc_RE" : It is required that every state in the connected set can be reached by following a
              pathway of reversible transitions or jumping between overlapping thermodynamic states while staying in
              the same Markov state. A reversible transition between two Markov states (within the same thermodynamic
              state :math:`k`) is a pair of Markov states that belong to the same strongly connected component of the
              count matrix (from thermodynamic state :math:`k`). Two thermodynamic states :math:`k` and :math:`l` are
              defined to overlap at Markov state :math:`i` if a replica exchange simulation
              :footcite:`hukushima1996exchange` restricted to state :math:`i` would show at least one transition from
              :math:`k` to :math:`l` or one transition from from :math:`l` to :math:`k`. The expected number of replica
              exchanges is estimated from the simulation data. The minimal number required of replica exchanges per
              Markov state can be increased by decreasing `connectivity_factor`.
            * "BAR_variance" : like 'post_hoc_RE' but with a different condition to define the thermodynamic overlap
              based on the variance of the BAR estimator :footcite:`shirts2008statistically`.
              Two thermodynamic states :math:`k` and :math:`l` are defined to overlap at Markov state :math:`i` if the
              variance of the free energy difference :math:`\Delta f_{kl}` computed with BAR
              (and restricted to conformations form Markov state :math:`i`) is less or equal than one. The minimally
              required variance can be controlled with `connectivity_factor`.
            * "summed_count_matrix" : all thermodynamic states are assumed to overlap. The connected set is then
              computed by summing the count matrices over all thermodynamic states and taking its largest strongly
              connected set. Not recommended!
        connectivity_factor : float, optional, default=1.0
            Only needed if connectivity="post_hoc_RE" or "BAR_variance". Values greater than 1.0 weaken the connectivity
            conditions. For 'post_hoc_RE' this multiplies the number of hypothetically observed transitions. For
            'BAR_variance' this scales the threshold for the minimal allowed variance of free energy differences.
        progress : object
            Progress bar object that `TRAMDataset` will call to indicate progress to the user.
            Tested for a tqdm progress bar. The interface is checked
            via :meth:`supports_progress_interface <deeptime.util.callbacks.supports_progress_interface>`.

        Raises
        ------
        ValueError
            If the connectivity type is unknown.
        """
        if connectivity not in TRAMDataset.connectivity_options:
            raise ValueError(
                f"Connectivity type unsupported. Connectivity must be one of {TRAMDataset.connectivity_options}.")
        lcc = self._find_largest_connected_set(connectivity, connectivity_factor, progress)
        self.restrict_to_submodel(lcc)

    def restrict_to_submodel(self, submodel):
        """Restrict the count matrices and `dtrajs` to the given submodel. The submodel is either given in form of a
        list of Markov state indices, or as a TransitionCountModel. All `dtrajs` sample indices that do not occur in the
        submodel will be set to -1. The count_models are recomputed after restricting the `dtrajs` to the submodel.

        Parameters
        ----------
        submodel : TransitionCountModel or list(int) or ndarray(int)
            The TransitionCountModel, or the Markov states indices that the dataset needs to be restricted to, .
        """
        dtrajs_connected = []
        if isinstance(submodel, list) or isinstance(submodel, np.ndarray):
            submodel = self._submodel_from_states(submodel)

        for k in range(self.n_therm_states):
            # Get largest connected set
            # Assign -1 to all indices not in the submodel.
            restricted_dtraj = submodel.transform_discrete_trajectories_to_submodel(self.dtrajs[k])

            # The transformation has converted all indices to the state indices of the submodel. We want the original
            # indices. Convert the newly assigned indices back to the original state symbols
            restricted_dtraj = submodel.states_to_symbols(restricted_dtraj)

            dtrajs_connected.append(restricted_dtraj)

        self.dtrajs = dtrajs_connected
        self._compute_counts()
        _invalidate_caches()

    def _submodel_from_states(self, indices):
        estimator = TransitionCountEstimator(lagtime=self.lagtime, count_mode=self.count_mode)

        # make a counts model over all observed samples.
        full_counts_model = estimator.fit_fetch(self.dtrajs)
        cset = full_counts_model.submodel(indices)
        return cset

    def _ensure_correct_data_types(self):
        # shape and type checks
        if len(self.dtrajs) != len(self.bias_matrices):
            raise ValueError("Number of trajectories is not equal to the number of bias matrices.")
        for d in self.dtrajs:
            types.ensure_integer_array(d, ndim=1)
        for b in self.bias_matrices:
            types.ensure_floating_array(b, ndim=2)

        # cast types and change axis order if needed
        self.dtrajs = [np.require(d, dtype=np.int32, requirements='C') for d in self.dtrajs]
        self.bias_matrices = [np.require(b, dtype=np.float64, requirements='C') for b in self.bias_matrices]

        # do the same for ttrajs if it exists
        if self.ttrajs is None or len(self.ttrajs) == 0:
            # ensure ttrajs is None. Leaving it an empty tuple will break the call to _tram_bindings
            self.ttrajs = None
        else:
            # find the number of therm states as the highest index in ttrajs
            for t in self.ttrajs:
                types.ensure_integer_array(t, ndim=1)
            self.ttrajs = [np.require(t, dtype=np.int32, requirements='C') for t in self.ttrajs]

    def _check_dimensions(self):
        # dimensionality checks
        for i, (d, b) in enumerate(zip(self.dtrajs, self.bias_matrices)):
            if d.shape[0] != b.shape[0]:
                raise ValueError(f"discrete trajectory {i} and bias matrix {i} should be of equal length.")

            if b.shape[1] != self.n_therm_states:
                raise ValueError(
                    f"Second dimension of bias matrix {i} should be of size n_therm_states (={self.n_therm_states})")

            # If we were given ttrajs, do the same checks for those.
        if self.ttrajs is not None:
            if len(self.ttrajs) != len(self.dtrajs):
                raise ValueError("number of ttrajs is not equal to number of dtrajs.")

            for i, (t, d) in enumerate(zip(self.ttrajs, self.dtrajs)):
                if t.shape[0] != d.shape[0]:
                    raise ValueError(f"ttraj {i} and dtraj {i} should be of equal length.")

    def _find_largest_connected_set(self, connectivity, connectivity_factor, progress=None):
        estimator = TransitionCountEstimator(lagtime=self.lagtime, count_mode=self.count_mode)

        # make a counts model over all observed samples.
        full_counts_model = estimator.fit_fetch(self.dtrajs)

        if connectivity is None:
            warnings.warn(f"connectivity type is None. Data has not been restricted to the largest connected set."
                          f"The full counts model has been returned.")
            return full_counts_model

        if connectivity == 'summed_count_matrix':
            # We assume the thermodynamic states have overlap when they contain counts from the same markov state.
            # Full counts model contains the sum of the state counts over all therm. states., and we simply ignore
            # the thermodynamic state indices.
            return full_counts_model.submodel_largest(
                connectivity_threshold=connectivity_factor,
                directed=True)

        if connectivity in ['post_hoc_RE', 'BAR_variance']:
            # get state counts for each trajectory (=for each therm. state)
            all_state_counts = np.asarray([estimator.fit_fetch(dtraj).state_histogram for dtraj in self.dtrajs],
                                          dtype=object)
            # pad with zero's so they are all the same size and easier for the cpp module to handle
            all_state_counts = to_zero_padded_array(all_state_counts, self.n_markov_states).astype(np.int32)

            # get list of all possible transitions between thermodynamic states. A transition is only possible when two
            # thermodynamic states have an overlapping markov state. Whether the markov state overlaps depends on the
            # sampled data and the connectivity settings and is computed in find_state_transitions:
            if connectivity == 'post_hoc_RE':
                connectivity_fn = tram.find_state_transitions_post_hoc_RE
            else:
                connectivity_fn = tram.find_state_transitions_BAR_variance

            with callbacks.ProgressCallback(progress, "Finding connected sets",
                                            self.n_therm_states * self.n_markov_states) as callback:
                (i_s, j_s) = connectivity_fn(self.ttrajs, self.dtrajs, self.bias_matrices, all_state_counts,
                                             self.n_therm_states, self.n_markov_states, connectivity_factor,
                                             callback)

            # add transitions that occurred within each thermodynamic state. These are simply the connected sets:
            for k in range(self.n_therm_states):
                for cset in estimator.fit_fetch(self.dtrajs[k]).connected_sets():
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

    def _find_trajectory_fragments(self):
        """ Find all trajectory fragments given the discrete trajectories and the thermodynamic state indices of each
        sample. Get rid of any negative state indices.
        If no replica exchange was done, this will simply return all positive values from dtrajs. If ttrajs are
        given, we assume replica exchange was done and restructure dtrajs in such a way that every replica-exchange
        swap is the start of a new trajectory.

        Returns
        -------
        dtraj_fragments : list(List(int))
           A list that contains for each thermodynamic state the fragments from all trajectories that were sampled at
           that thermodynamic state. fragment_indices[k][i] defines the i-th fragment sampled at thermodynamic state k.
           The fragments are be restricted to the largest connected set and negative state indices are excluded.
        """
        if self.ttrajs is None or len(self.ttrajs) == 0:
            # No ttrajs were given. We assume each trajectory in dtrajs was sampled in a distinct thermodynamic state.
            # The thermodynamic state index equals the trajectory index, and the dtrajs are unchanged.
            return [[self.dtrajs[k][self.dtrajs[k] >= 0]] for k in range(self.n_therm_states)]

        # replica exchange data means that the trajectories to not correspond 1:1 to thermodynamic states.
        # get a mapping from trajectory segments to thermodynamic states
        fragment_indices = self._find_trajectory_fragment_mapping()

        fragments = [[] for _ in range(self.n_therm_states)]
        # for each them. state k, gather all trajectory fragments that were sampled at that state.
        for k in range(self.n_therm_states):
            # Select the fragments using the list of indices.
            for (traj_idx, start, stop) in fragment_indices[k]:
                fragment = self.dtrajs[traj_idx][start:stop]

                # Whenever state values are negative, those samples do not belong in the connected set and need to be
                # excluded. We split trajectories where negative state indices occur.
                # Example: [0, 0, 2, -1, 2, 1, 0], we want to exclude the sample with state index -1.
                # Simply filtering out negative state indices would lead to [0, 0, 2, 2, 1, 0] which gives a transition
                # 2 -> 2 which doesn't exist.  Instead, split the trajectory at negative state indices to get
                # [0, 0, 2], [2, 1, 0]
                negative_state_indices = np.where(fragment < 0)[0]
                if len(negative_state_indices) > 0:
                    fragments[k].extend(_split_at_negative_state_indices(fragment, negative_state_indices))
                else:
                    fragments[k].append(fragment)
        return fragments

    def _find_trajectory_fragment_mapping(self):
        """ define mapping that gives for each trajectory the slices that make up a trajectory inbetween RE swaps.
            At every RE swap point, the trajectory is sliced, so that swap point occurs as a trajectory start.

        Returns
        -------
        fragment_indices : list(list(tuple(int)))
            A list that contains for each thermodynamic state the fragments from all trajectories that were sampled at
            that thermodynamic state.
            fragment_indices[k][i] defines the i-th fragment sampled at thermodynamic state k. The tuple consists of
            (traj_idx, start, stop) where traj_index is the index of the trajectory the fragment can be found at, start
            is the start index of the trajectory, and stop the end index (exclusive) of the fragment.

        """
        return tram.find_trajectory_fragment_indices(self.ttrajs, self.n_therm_states)

    def _compute_counts(self):
        r""" Compute the count models from the dtrajs and ttrajs.

        Returns
        -------
        count_models : list(TransitionCountModel)
            A list of TransitionCountModels containing one TransitionCountModel per thermodynamic state.
        """
        # In the case of replica exchange: get all trajectory fragments (also removing any negative state indices).
        # the dtraj_fragments[k] contain all trajectories within thermodynamic state k, starting a new trajectory at
        # each replica exchange swap. If there was no replica exchange, dtraj_fragments == dtrajs.
        dtraj_fragments = self._find_trajectory_fragments()

        # ... convert those into count matrices.
        self.count_models = self._construct_count_models(dtraj_fragments)

    def _construct_count_models(self, dtraj_fragments):
        """ Construct a TransitionCountModel for each thermodynamic state based on the dtraj_fragments, and store in
        self.count_models.

        Parameters
        ----------
        dtraj_fragments: list(list(int))
           A list that contains for each thermodynamic state the fragments from all trajectories that were sampled at
           that thermodynamic state. fragment_indices[k][i] defines the i-th fragment sampled at thermodynamic state k.
           The fragments should be restricted to the largest connected set and not contain any negative state indices.

        Returns
        -------
        count_models : list(TransitionCountModel)
        """

        estimator = TransitionCountEstimator(lagtime=self.lagtime, count_mode=self.count_mode)
        count_models = []

        for k in range(self.n_therm_states):

            if len(dtraj_fragments[k]) == 0 or np.all([len(frag) <= self.lagtime for frag in dtraj_fragments[k]]):
                warnings.warn(f"No transitions for thermodynamic state {k} after cutting the trajectories into "
                              f"fragments that start at each replica exchange swap. Replica exchanges possibly occur "
                              f"within the span of the lag time.")
                # there are no samples from this state that belong to the connected set. Make an empty count model.
                count_models.append(TransitionCountModel(np.zeros(self.n_markov_states, self.n_markov_states)))
            else:
                # make a counts model for the samples that belong to the connected set.
                traj_counts_model = estimator.fit_fetch(dtraj_fragments[k])
                count_models.append(traj_counts_model)
        return count_models
