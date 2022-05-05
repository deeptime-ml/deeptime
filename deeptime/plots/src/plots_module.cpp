#include <deeptime/plots/fruchterman_reingold.h>

using namespace pybind11::literals;
PYBIND11_MODULE(_plots_bindings, m) {
    m.def("fruchterman_reingold", &deeptime::plots::fruchtermanReingold<float>,
            "adjacency_matrix"_a, "initial_positions"_a, "iterations"_a = 50, "k"_a = -1,
            "update_dims"_a = std::vector<std::size_t>{});
    m.def("fruchterman_reingold", &deeptime::plots::fruchtermanReingold<double>,
          "adjacency_matrix"_a, "initial_positions"_a, "iterations"_a = 50, "k"_a = -1,
          "update_dims"_a = std::vector<std::size_t>{});
}
