#include "register_clustering.h"

PYBIND11_MODULE(_clustering_bindings, m) {
    m.doc() = "module containing clustering algorithms.";
    auto euclideanModule = m.def_submodule("euclidean");
    deeptime::clustering::registerClusteringImplementation<EuclideanMetric>(euclideanModule);
}
