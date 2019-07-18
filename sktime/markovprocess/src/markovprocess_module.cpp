#include "discrete_trajectories.h"

PYBIND11_MODULE(_markovprocess_bindings, m) {
    {
        auto sampleMod = m.def_submodule("sample");
        sampleMod.def("index_states", &indexStates);
        sampleMod.def("count_states", &countStates);
    }
}
