// author: clonker

#include "discrete_trajectories.h"
#include "simulation.h"

using namespace pybind11::literals;

PYBIND11_MODULE(_markov_bindings, m) {
    {
        auto sampleMod = m.def_submodule("sample");
        sampleMod.def("index_states", &indexStates, py::arg("dtrajs"), py::arg("subset") = py::none());
        sampleMod.def("count_states", &countStates, py::arg("dtrajs"));
    }
    {
        auto simMod = m.def_submodule("simulation");
        simMod.def("trajectory", &trajectory<float>, "N"_a, "start"_a, "P"_a, "stop"_a = py::none(), "seed"_a = -1);
        simMod.def("trajectory", &trajectory<double>, "N"_a, "start"_a, "P"_a, "stop"_a = py::none(), "seed"_a = -1);
    }
}
