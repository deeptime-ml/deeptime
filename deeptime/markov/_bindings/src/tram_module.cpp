//  author: maaike
#include "deeptime/markov/tram/tram.h"
#include "deeptime/markov/tram/connected_set.h"
#include "deeptime/markov/tram/trajectory_mapping.h"
 
PYBIND11_MODULE(_tram_bindings, m) {
    using namespace pybind11::literals;
    using namespace deeptime::markov::tram;
    {
        auto tramMod = m.def_submodule("tram");

        py::class_<TRAM<double>>(tramMod, "TRAM")
                .def(py::init<std::shared_ptr<TRAMInput<double>> &, std::size_t>(), "tram_input"_a,
                     "callback_interval"_a = 1)
                .def("estimate", &TRAM<double>::estimate, py::call_guard<py::gil_scoped_release>(),
                     "max_iter"_a = 1000, "max_err"_a = 1e-8,
                     "track_log_likelihoods"_a = false, "callback"_a = nullptr)
                .def("transition_matrices", &TRAM<double>::transitionMatrices)
                .def("biased_conf_energies", &TRAM<double>::biasedConfEnergies)
                .def("therm_state_energies", &TRAM<double>::energiesPerThermodynamicState)
                .def("markov_state_energies", &TRAM<double>::energiesPerMarkovState)
                .def("log_likelihood", &TRAM<double>::computeLogLikelihood, py::call_guard<py::gil_scoped_release>())
                .def("compute_sample_weights", &TRAM<double>::computeSampleWeights, py::call_guard<py::gil_scoped_release>(),
			       	"therm_state_index"_a = -1);

        py::class_<TRAMInput<double>, std::shared_ptr<TRAMInput<double>>>(tramMod, "TRAMInput").def(
                py::init<deeptime::np_array_nfc<int> &&, deeptime::np_array_nfc<int> &&, DTrajs, BiasMatrices<double>>(),
                "state_counts"_a, "transition_counts"_a, "dtrajs"_a, "bias_matrices"_a);

        tramMod.def("find_state_transitions_post_hoc_RE",
                    &findStateTransitions<double, OverlapPostHocReplicaExchange<double>>,
		    py::call_guard<py::gil_scoped_release>(),
		    py::return_value_policy::move,
                    "ttrajs"_a, "dtrajs"_a, "bias_matrices"_a, "stateCounts"_a, "n_therm_states"_a, "n_conf_states"_a,
                    "connectivity_factor"_a, "callback"_a);

        tramMod.def("find_state_transitions_BAR_variance", &findStateTransitions<double, OverlapBarVariance<double>>,
		    py::call_guard<py::gil_scoped_release>(),
		    py::return_value_policy::move,
                    "ttrajs"_a, "dtrajs"_a, "bias_matrices"_a, "stateCounts"_a, "n_therm_states"_a, "n_conf_states"_a,
                    "connectivity_factor"_a, "callback"_a);

        tramMod.def("find_trajectory_fragment_indices", &findTrajectoryFragmentIndices, "ttrajs"_a, "n_therm_states"_a);
    }
}
