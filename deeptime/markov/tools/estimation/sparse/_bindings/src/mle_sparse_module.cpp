//
// Created by mho on 7/29/20.
//

#include "mle_trev_sparse.h"


PYBIND11_MODULE(_mle_sparse_bindings, m) {
    m.def("mle_trev_sparse", &mle_trev_sparse<float>);
    m.def("mle_trev_sparse", &mle_trev_sparse<double>);
    m.def("mle_trev_sparse", &mle_trev_sparse<long double>);
    m.def("mle_trev_given_pi_sparse", &mle_trev_given_pi_sparse<float>);
    m.def("mle_trev_given_pi_sparse", &mle_trev_given_pi_sparse<double>);
    m.def("mle_trev_given_pi_sparse", &mle_trev_given_pi_sparse<long double>);
}
