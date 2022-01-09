import numpy as np
import scipy as sp
import warnings

from deeptime.util import types, callbacks
from deeptime.markov import TransitionCountEstimator, TransitionCountModel

from ._tram_bindings import tram


def _determine_n_states(dtrajs):
    return max(np.max(d) for d in dtrajs) + 1


def _determine_n_therm_states(dtrajs, ttrajs):
    if ttrajs is None:
        return len(dtrajs)
    else:
        return _determine_n_states(ttrajs)


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


class TRAMDataset:
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
        self.compute_counts()

    @property
    def tram_input(self):
        return tram.TRAMInput(self.state_counts, self.transition_counts, self.dtrajs, self.bias_matrices)

    @property
    def n_therm_states(self):
        return self._n_therm_states

    @n_therm_states.setter
    def n_therm_states(self, n_therm_states):
        # number of thermodynamic states can be increased, but not decreased
        if n_therm_states < self._n_therm_states:
            if self.ttrajs is None:
                msg = "dtrajs out of bounds. There are more dtrajs then there are thermodynamic states. Provide ttrajs " \
                      "or increase the number of thermodynamic states."
            else:
                msg = "ttrajs out of bounds. Largest state number in dtrajs is larger than the number of Markov states."
            raise ValueError(msg)

    @property
    def n_markov_states(self):
        return self._n_markov_states

    @n_markov_states.setter
    def n_markov_states(self, n_markov_states):
        # number of markov states can be increased, but not decreased
        if n_markov_states < self._n_markov_states:
            raise ValueError(
                "dtrajs out of bounds. Largest state number in dtrajs is larger than the number of Markov states.")

    @property
    def transition_counts(self):
        transition_counts = np.zeros((self.n_therm_states, self.n_markov_states, self.n_markov_states), dtype=np.intc)

        for k in range(self.n_therm_states):
            model_k = self.count_models[k]
            if model_k.count_matrix.sum() > 0:
                i_s, j_s = np.meshgrid(model_k.state_symbols, model_k.state_symbols)
                # place submodel counts in our full-sized count matrices
                transition_counts[k, i_s, j_s] = model_k.count_matrix.T

        return transition_counts

    @property
    def state_counts(self):
        state_counts = np.zeros((self.n_therm_states, self.n_markov_states), dtype=np.intc)

        for k in range(self.n_therm_states):
            model_k = self.count_models[k]
            if model_k.count_matrix.sum() > 0:
                state_counts[k, model_k.state_symbols] = model_k.state_histogram

        return state_counts

    def compute_counts(self):
        # In the case of replica exchange: get all trajectory fragments (also removing any negative state indices).
        # the dtraj_fragments[k] contain all trajectories within thermodynamic state k, starting a new trajectory at
        # each replica exchange swap. If there was no replica exchange, dtraj_fragments == dtrajs.
        dtraj_fragments = self._find_trajectory_fragments()

        # ... convert those into count matrices.
        self.count_models = self._construct_count_models(dtraj_fragments)

    def restrict_to_largest_connected_set(self, connectivity='post_hoc_RE', connectivity_factor=1.0, progress_bar=None):
        lcc = self._find_largest_connected_set(connectivity, connectivity_factor, progress_bar)
        self.restrict_to_connected_set(lcc)

    def restrict_to_connected_set(self, cset):
        """Restrict the count matrices and dtrajs to the connected set. All dtraj samples not in the largest
        connected set will be set to -1.

        Parameters
        ----------
        cset
        """
        dtrajs_connected = []

        for k in range(self.n_therm_states):
            # Get largest connected set
            # Assign -1 to all indices not in the submodel.
            restricted_dtraj = cset.transform_discrete_trajectories_to_submodel(self.dtrajs[k])

            # The transformation has converted all indices to the state indices of the submodel. We want the original
            # indices. Convert the newly assigned indices back to the original state symbols
            restricted_dtraj = cset.states_to_symbols(restricted_dtraj)

            dtrajs_connected.append(restricted_dtraj)

        self.dtrajs = dtrajs_connected
        self.compute_counts()

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

    def _find_largest_connected_set(self, connectivity, connectivity_factor, progress_bar=None):
        """ Find the largest connected set of Markov states based on the connectivity settings and the input data.
        A full counts model is first calculated from all dtrajs. Then, the connected set is computed based on
        self.connectivity and self.connectivity_factor. The full counts model is then reduces to the connected set and
        returned. If the connectivity setting is unknown, the full counts model is returned.

        Returns
        -------
        counts_model : MarkovStateModel
            The counts model pertaining to the largest connected set.
        """
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
            all_state_counts = to_zero_padded_array(all_state_counts, self.n_markov_states)

            # get list of all possible transitions between thermodynamic states. A transition is only possible when two
            # thermodynamic states have an overlapping markov state. Whether the markov state overlaps depends on the
            # sampled data and the connectivity settings and is computed in find_state_transitions:
            if connectivity == 'post_hoc_RE':
                connectivity_fn = tram.find_state_transitions_post_hoc_RE
            else:
                connectivity_fn = tram.find_state_transitions_BAR_variance

            with callbacks.Callback(progress_bar, self.n_therm_states * self.n_markov_states,
                                    "Finding connected sets") as callback:
                (i_s, j_s) = connectivity_fn(self.ttrajs, self.dtrajs, self.bias_matrices, all_state_counts,
                                             self.n_therm_states, self.n_markov_states, connectivity_factor,
                                             callback)
            print((i_s, j_s))
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
        dtraj_fragments : list(List(int32))
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

        fragments = []
        # for each them. state k, gather all trajectory fragments that were sampled at that state.
        for k in range(self.n_therm_states):
            # take the fragments based on the list of indices. Exclude all values that are less than zero. They don't
            # belong in the connected set.
            fragments.append([self.dtrajs[traj_idx][start:stop][self.dtrajs[traj_idx][start:stop] >= 0]
                              for (traj_idx, start, stop) in fragment_indices[k]])
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
