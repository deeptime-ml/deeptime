//  author: maaike
#include "deeptime/markov/msm/tram/tram.h"
#include "deeptime/markov/msm/tram/connected_set.h"
#include "deeptime/markov/msm/tram/trajectory_mapping.h"
#include "deeptime/markov/msm/tram/mbar.h"

PYBIND11_MODULE(_tram_bindings, m) {
    using namespace pybind11::literals;
    using namespace deeptime::markov::tram;
    {
        auto tramMod = m.def_submodule("tram");

        py::class_<TRAM<double >>(tramMod, "TRAM")
            .def(py::init<deeptime::np_array_nfc<double> &,
                         deeptime::np_array_nfc<double> &, deeptime::np_array_nfc<double> &>(),
                 "biased_conf_energies"_a, "lagrangian_mult_log"_a, "modified_state_counts_log"_a)
            .def("estimate", &TRAM<double>::estimate,
                 py::call_guard<py::gil_scoped_release>(),
                 "input"_a, "max_iter"_a = 1000, "max_err"_a = 1e-8, "callback_interval"_a = 1,
                 "track_log_likelihoods"_a = false, "callback"_a = nullptr)
            .def_property_readonly("transition_matrices", &TRAM<double>::transitionMatrices)
            .def_property_readonly("biased_conf_energies", &TRAM<double>::biasedConfEnergies)
            .def_property_readonly("modified_state_counts_log", &TRAM<double>::modifiedStateCountsLog)
            .def_property_readonly("lagrangian_mult_log", &TRAM<double>::lagrangianMultLog)
            .def_property_readonly("therm_state_energies", &TRAM<double>::thermStateEnergies)
            .def_property_readonly("markov_state_energies", &TRAM<double>::markovStateEnergies);

        py::class_<TRAMInput<double>, std::shared_ptr<TRAMInput<double>>>(tramMod, "TRAMInput").def(
                py::init<deeptime::np_array_nfc<int> &&, deeptime::np_array_nfc<int> &&, BiasMatrices<double>>(),
                "state_counts"_a, "transition_counts"_a,  "bias_matrix"_a);

        tramMod.def("initialize_lagrangians", &initLagrangianMult<double>, "transition_counts"_a);

        tramMod.def("initialize_free_energies_mbar", &initialize_MBAR<double>,
                    "bias_matrix"_a, "state_counts"_a, "max_iter"_a = 1000,"max_err"_a = 1e-6,
                    "callback_interval"_a = 1, "callback"_a = nullptr);

        tramMod.def("compute_sample_weights_log", &computeSampleWeightsLog<double>,
                    py::call_guard<py::gil_scoped_release>(),
                    "dtraj"_a, "bias_matrix"_a, "therm_state_energies"_a,
                    "modified_state_counts_log"_a, "therm_state_index"_a = -1);

        tramMod.def("find_state_transitions_post_hoc_RE",
                    &findStateTransitions<double, OverlapPostHocReplicaExchange<double >>,
                    py::call_guard<py::gil_scoped_release>(),
                    "ttrajs"_a, "dtrajs"_a, "bias_matrices"_a, "stateCounts"_a, "n_therm_states"_a, "n_conf_states"_a,
                    "connectivity_factor"_a, "callback"_a);

        tramMod.def("find_state_transitions_BAR_variance", &findStateTransitions<double,
                            OverlapBarVariance<double >>,
                    py::call_guard<py::gil_scoped_release>(),
                    "ttrajs"_a, "dtrajs"_a, "bias_matrices"_a, "stateCounts"_a, "n_therm_states"_a, "n_conf_states"_a,
                    "connectivity_factor"_a, "callback"_a);

        tramMod.def("find_trajectory_fragment_indices", &findTrajectoryFragmentIndices, "ttrajs"_a, "n_therm_states"_a);

        tramMod.def("compute_log_likelihood", computeLogLikelihood<double>,
                    py::call_guard<py::gil_scoped_release>(),
                    "dtraj"_a, "biasMatrix"_a, "biasedConfEnergies"_a, "modifiedStateCountsLog"_a,
                    "thermStateEnergies"_a, "stateCounts"_a, "transitionCounts"_a, "transitionMatrices"_a);
    }
}
