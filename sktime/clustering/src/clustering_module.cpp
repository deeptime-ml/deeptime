//
// Created by marscher on 7/17/17.
//

#include "regspace.h"
#include "kmeans.h"

using dtype = float;

PYBIND11_MODULE(_ext, m) {
    m.doc() = "module containing clustering algorithms.";

    auto regspace_mod = m.def_submodule("regspace");
    auto kmeans_mod = m.def_submodule("kmeans");

    typedef ClusteringBase<dtype> cbase_f;
    typedef RegularSpaceClustering<dtype> regspace_f;

    // register base class first.
    py::class_<cbase_f>(m, "ClusteringBase_f")
            .def(py::init<const std::string&, std::size_t>())
            .def("assign", &cbase_f::assign_chunk_to_centers)
            .def("precenter_centers", &cbase_f::precenter_centers);
    // regular space clustering.
    py::class_<regspace_f, cbase_f>(regspace_mod, "Regspace_f")
            .def(py::init<dtype, std::size_t, const std::string&, size_t>())
            .def("cluster", &regspace_f::cluster);
    py::register_exception<MaxCentersReachedException>(regspace_mod, "MaxCentersReachedException");
    // kmeans
    typedef KMeans<dtype> kmeans_f;
    py::class_<kmeans_f, cbase_f>(kmeans_mod, "Kmeans_f")
            .def(py::init<unsigned int, const std::string&, std::size_t>(),
                 py::arg("k"), py::arg("metric"), py::arg("dim"))
             // py::arg("callback") = py::none()
            .def("cluster", &kmeans_f::cluster)
            .def("cluster_loop", &kmeans_f::cluster_loop)
            .def("init_centers_KMpp", &kmeans_f::initCentersKMpp)
            .def("cost_function", &kmeans_f::costFunction);
}
