//
// Created by mho on 7/29/20.
//

#include "mle_trev.h"
#include "sampler.h"

template<typename Sampler, typename Mod>
void exportSampler(Mod &m, const std::string &name) {
    py::class_<Sampler>(m, name.c_str())
            .def(py::init<int>())
            .def("update", &Sampler::update);
}


PYBIND11_MODULE(_mle_bindings, m) {
    m.def("mle_trev_dense", &mle_trev_dense<float>);
    m.def("mle_trev_dense", &mle_trev_dense<double>);
    m.def("mle_trev_dense", &mle_trev_dense<long double>);
    m.def("mle_trev_given_pi_dense", &mle_trev_given_pi_dense<float>);
    m.def("mle_trev_given_pi_dense", &mle_trev_given_pi_dense<double>);
    m.def("mle_trev_given_pi_dense", &mle_trev_given_pi_dense<long double>);

    exportSampler<RevSampler<float>>(m, "RevSampler32");
    exportSampler<RevSampler<double>>(m, "RevSampler64");
    exportSampler<RevSampler<long double>>(m, "RevSampler128");

    exportSampler<RevPiSampler<float>>(m, "RevPiSampler32");
    exportSampler<RevPiSampler<double>>(m, "RevPiSampler64");
    exportSampler<RevPiSampler<long double>>(m, "RevPiSampler128");
}
