#include "discrete_trajectories.h"

PYBIND11_MODULE(_markovprocess_bindings, m) {
    {
        auto sampleMod = m.def_submodule("sample");
        sampleMod.def("index_states", &indexStates, py::arg("dtrajs"), py::arg("subset") = py::none());
        sampleMod.def("count_states", &countStates, py::arg("dtrajs"));
    }
}
