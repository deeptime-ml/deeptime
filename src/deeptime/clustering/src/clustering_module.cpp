#include <deeptime/clustering/register_clustering.h>

PYBIND11_MODULE(_clustering_bindings, m) {
    using namespace deeptime::clustering;

    m.doc() = "module containing clustering algorithms.";
    auto euclideanModule = m.def_submodule("euclidean");
    deeptime::clustering::registerClusteringImplementation<EuclideanMetric>(euclideanModule);
}
