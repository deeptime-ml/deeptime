// author: maaike

#include "tram.h"

PYBIND11_MODULE(_tram_bindings, m) {
    using namespace pybind11::literals;
    using namespace deeptime::tram;
    {
        auto tramMod = m.def_submodule("tram");

        py::class_<TRAM<double>>(m, "TRAM")
                .def(py::init<np_array_nfc<int>&&, np_array_nfc<int>&&, const DTrajs&,
                              const py::list, std::size_t>(), "state_counts"_a, "transition_counts"_a, "dtrajs"_a, "bias_matrices"_a,
                              "convergence_logging_interval"_a = 0)
                .def("estimate", &TRAM<double>::estimate)
                .def("estimate_transition_matrices", &TRAM<double>::estimateTransitionMatrices);

        tramMod.def("_bar_df", &_bar_df<double>, "db_IJ"_a, "L1"_a, "db_JI"_a, "L2"_a, "scratch"_a);

    }
}
