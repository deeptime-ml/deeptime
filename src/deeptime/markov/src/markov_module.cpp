 // author: clonker

#include <deeptime/markov/discrete_trajectories.h>
#include <deeptime/markov/simulation.h>

using namespace pybind11::literals;

PYBIND11_MODULE(_markov_bindings, m) {
    {
        auto sampleMod = m.def_submodule("sample");
        sampleMod.def("index_states", &deeptime::markov::indexStates, py::arg("dtrajs"), py::arg("subset") = py::none());
        sampleMod.def("count_states", &deeptime::markov::countStates, py::arg("dtrajs"));
    }
    {
        auto simMod = m.def_submodule("simulation");
        simMod.def("trajectory", &deeptime::markov::trajectory<float>, "N"_a, "start"_a, "P"_a, "stop"_a = py::none(), "seed"_a = -1);
        simMod.def("trajectory", &deeptime::markov::trajectory<double>, "N"_a, "start"_a, "P"_a, "stop"_a = py::none(), "seed"_a = -1);
    }
}
