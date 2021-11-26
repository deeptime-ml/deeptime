// author: maaike

#include "tram.h"

PYBIND11_MODULE(_tram_bindings, m) {
    using namespace pybind11::literals;
    using namespace deeptime::tram;
    {
        auto tramMod = m.def_submodule("tram");

        py::class_<TRAM<double>>(m, "TRAM")
                .def(py::init<std::shared_ptr<TRAMInput<double>> &, std::size_t>(), "tram_input"_a,
                     "callback_interval"_a = 1)
                .def("estimate", &TRAM<double>::estimate, "maxIter"_a = 1000, "maxErr"_a = 1e-8, "callback"_a = nullptr)
                .def("estimate_transition_matrices", &TRAM<double>::estimateTransitionMatrices)
                .def("biased_conf_energies", &TRAM<double>::getBiasedConfEnergies);

        py::class_<TRAMInput<double>, std::shared_ptr<TRAMInput<double>>>(m, "TRAM_input").def(
                py::init<np_array_nfc<int> &&, np_array_nfc<int> &&, py::list, py::list>(), "state_counts"_a,
                "transition_counts"_a, "dtrajs"_a, "bias_matrices"_a);

//        py::class_<TRAMInput<double>>(m, "TRAM_input").def(py::init<np_array_nfc<int> &&, np_array_nfc<int> &&,
//                py::list, py::list>(), "state_counts"_a, "transition_counts"_a, "dtrajs"_a, "bias_matrices"_a);

        tramMod.def("_bar_df", &_bar_df<double>, "db_IJ"_a, "L1"_a, "db_JI"_a, "L2"_a, "scratch"_a);

    }
}
