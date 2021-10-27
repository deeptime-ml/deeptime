#include "metric.h"
#include "kmeans.h"
#include "regspace.h"

using namespace pybind11::literals;

void registerKmeans(py::module &mod) {
    mod.def("cluster", deeptime::clustering::kmeans::cluster<float>, "chunk"_a, "centers"_a,
            "n_threads"_a, "metric"_a = nullptr);
    mod.def("cluster", deeptime::clustering::kmeans::cluster<double>, "chunk"_a, "centers"_a,
            "n_threads"_a, "metric"_a = nullptr);
    mod.def("cluster_loop", &deeptime::clustering::kmeans::cluster_loop<float>,
            "chunk"_a, "centers"_a, "n_threads"_a, "max_iter"_a, "tolerance"_a,
            "callback"_a, "metric"_a = nullptr);
    mod.def("cluster_loop", &deeptime::clustering::kmeans::cluster_loop<double>,
            "chunk"_a, "centers"_a, "n_threads"_a, "max_iter"_a, "tolerance"_a,
            "callback"_a, "metric"_a = nullptr);
    mod.def("cost_function", &deeptime::clustering::kmeans::costAssignFunction<float>,
            "chunk"_a, "centers"_a, "n_threads"_a, "metric"_a = nullptr);
    mod.def("cost_function", &deeptime::clustering::kmeans::costAssignFunction<double>,
            "chunk"_a, "centers"_a, "n_threads"_a, "metric"_a = nullptr);
    mod.def("init_centers_kmpp", &deeptime::clustering::kmeans::initKmeansPlusPlus<float>,
            "chunk"_a, "k"_a, "random_seed"_a, "n_threads"_a, "callback"_a, "metric"_a = nullptr);
    mod.def("init_centers_kmpp", &deeptime::clustering::kmeans::initKmeansPlusPlus<double>,
            "chunk"_a, "k"_a, "random_seed"_a, "n_threads"_a, "callback"_a, "metric"_a = nullptr);
}

void registerRegspace(py::module &module) {
    module.def("cluster", &clustering::regspace::cluster<float>, "chunk"_a, "centers"_a, "dmin"_a,
            "max_clusters"_a, "n_threads"_a, "metric"_a = nullptr);
    module.def("cluster", &clustering::regspace::cluster<double>, "chunk"_a, "centers"_a, "dmin"_a,
               "max_clusters"_a, "n_threads"_a, "metric"_a = nullptr);
    py::register_exception<clustering::regspace::MaxCentersReachedException>(module, "MaxCentersReachedException");
}

template<typename dtype, bool squared>
void defDistances(py::module &m) {
    std::string name = "distances";
    if (squared) name += "_squared";
    m.def(name.c_str(), [](np_array<dtype> X, np_array<dtype> Y, py::object XX, py::object YY, int /*nThreads*/, const Metric* metric) {
        metric = metric ? metric : default_metric();
        auto dim = static_cast<std::size_t>(X.shape(1));
        if(static_cast<std::size_t>(Y.shape(1)) != dim) {
            throw std::invalid_argument("dimension mismatch: " + std::to_string(dim) + " != " + std::to_string(Y.shape(1)));
        }
        const double* xx = nullptr;
        if(!XX.is_none()) {
            xx = py::cast<np_array<double>>(XX).data();
        }
        const double* yy = nullptr;
        if(!YY.is_none()) {
            yy = py::cast<np_array<double>>(YY).data();
        }
        auto nXs = static_cast<std::size_t>(X.shape(0));
        auto nYs = static_cast<std::size_t>(Y.shape(0));

        auto distances = computeDistances<squared>(X.data(), nXs, Y.data(), nYs, dim, xx, yy, metric);
        return distances.numpy();
    }, "X"_a, "Y"_a, "XX"_a = py::none(), "YY"_a = py::none(), "n_threads"_a = 0, "metric"_a = nullptr);
}

PYBIND11_MODULE(_clustering_bindings, m) {
    m.doc() = "module containing clustering algorithms.";
    auto kmeans_mod = m.def_submodule("kmeans");
    registerKmeans(kmeans_mod);
    auto regspace_mod = m.def_submodule("regspace");
    registerRegspace(regspace_mod);

    m.def("assign", &assign_chunk_to_centers<float>, "chunk"_a, "centers"_a, "n_threads"_a, "metric"_a = nullptr);
    m.def("assign", &assign_chunk_to_centers<double>, "chunk"_a, "centers"_a, "n_threads"_a, "metric"_a = nullptr);
    defDistances<float, true>(m);
    defDistances<double, true>(m);
    defDistances<float, false>(m);
    defDistances<double, false>(m);


    py::class_<Metric>(m, "Metric", R"delim(
The metric class. It should not be directly instantiated from python, but is rather meant as a C++ interface. Since
clustering is computationally expensive and the metric is called often, it makes sense to export this functionality
from Python into an extension. To this end the abstract Metric class as defined in `clustering/include/metric.h` can
be implemented and exposed to python. Afterwards it can be used in the clustering module through the
:data:`metric registry <deeptime.clustering.metrics>`.
)delim");
    py::class_<EuclideanMetric, Metric>(m, "EuclideanMetric").def(py::init<>());
}
