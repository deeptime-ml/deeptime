//  author: maaike
#include "deeptime/markov/tram/tram.h"
#include "deeptime/markov/tram/connected_set.h"
#include "deeptime/markov/tram/trajectory_mapping.h"

PYBIND11_MODULE(_tram_bindings, m) {
    using namespace pybind11::literals;
    using namespace deeptime::tram;
    {
        auto tramMod = m.def_submodule("tram");

        py::class_<TRAM<double>>(tramMod, "TRAM")
                .def(py::init<std::shared_ptr<TRAMInput<double>> &, std::size_t>(), "tram_input"_a,
                     "callback_interval"_a = 1)
                .def("estimate", &TRAM<double>::estimate, "max_iter"_a = 1000, "max_err"_a = 1e-8,
                     "track_log_likelihoods"_a = false, "callback"_a = nullptr)
                .def("transition_matrices", &TRAM<double>::transitionMatrices)
                .def("biased_conf_energies", &TRAM<double>::biasedConfEnergies)
                .def("therm_state_energies", &TRAM<double>::energiesPerThermodynamicState)
                .def("markov_state_energies", &TRAM<double>::energiesPerMarkovState)
                .def("get_sample_weights", &TRAM<double>::sampleWeights, "therm_state_index"_a = -1);

        using Input = TRAMInput<double>;
        py::class_<Input, std::shared_ptr<Input>>(tramMod, "TRAM_input").def(
                py::init<deeptime::np_array_nfc<int> &&, deeptime::np_array_nfc<int> &&, Input::DTrajs, Input::BiasMatrices>(),
                "state_counts"_a, "transition_counts"_a, "dtrajs"_a, "bias_matrices"_a);

        tramMod.def("get_state_transitions", &getStateTransitions<double, OverlapPostHocReplicaExchange<double>>,
                    "ttrajs"_a, "dtrajs"_a, "bias_matrices"_a, "stateCounts"_a, "n_therm_states"_a, "n_conf_states"_a,
                    "connectivity_factor"_a);

        tramMod.def("bar_variance", &hasOverlapBarVariance<double>);
        tramMod.def("post_hoc_RE", &hasOverlapPostHocReplicaExchange<double>);

        tramMod.def("find_trajectory_fragment_indices", &getTrajectoryFragmentIndices, "ttrajs"_a, "n_therm_states"_a);
    }
}
