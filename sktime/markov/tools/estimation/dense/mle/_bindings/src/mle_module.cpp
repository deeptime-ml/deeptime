//
// Created by mho on 7/29/20.
//

#include "mle_trev.h"

PYBIND11_MODULE(_mle_bindings, m) {
    m.def("mle_trev_dense", &mle_trev_dense<float>);
    m.def("mle_trev_dense", &mle_trev_dense<double>);
    m.def("mle_trev_given_pi_dense", &mle_trev_given_pi_dense<float>);
    m.def("mle_trev_given_pi_dense", &mle_trev_given_pi_dense<double>);
}
