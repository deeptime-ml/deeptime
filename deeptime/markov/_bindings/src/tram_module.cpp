 //  author: maaike
#include "tram.h"


PYBIND11_MODULE(_tram_bindings, m) {
    using namespace pybind11::literals;
    using namespace deeptime::tram;
    {
        auto tramMod = m.def_submodule("tram");

        py::class_<TRAM<double>>(m, "TRAM")
                .def(py::init<std::shared_ptr<TRAMInput<double>> &, std::size_t>(), "tram_input"_a,
                     "callback_interval"_a = 1)
                .def("estimate", &TRAM<double>::estimate, "max_iter"_a = 1000, "max_err"_a = 1e-8, "track_log_likelihoods"_a=false, "callback"_a = nullptr)
                .def("transition_matrices", &TRAM<double>::getTransitionMatrices)
                .def("biased_conf_energies", &TRAM<double>::getBiasedConfEnergies)
		.def("therm_state_energies", &TRAM<double>::getEnergiesPerThermodynamicState)
		.def("markov_state_energies", &TRAM<double>::getEnergiesPerMarkovState);

        using Input = TRAMInput<double>;
        py::class_<Input, std::shared_ptr<Input>>(m, "TRAM_input").def(
                py::init<np_array_nfc<int> &&, np_array_nfc<int> &&, Input::DTrajs, Input::BiasMatrices>(), "state_counts"_a,
                "transition_counts"_a, "dtrajs"_a, "bias_matrices"_a);

        tramMod.def("_bar_df", &_bar_df<double>, "db_IJ"_a, "L1"_a, "db_JI"_a, "L2"_a, "scratch"_a);

    }
}
