
#include "../include/kahan_summation.h"

PYBIND11_MODULE(kahandot, m) {
    m.def("kdot", &kdot<float>);
    m.def("kdot", &kdot<double>);
    m.def("kdot", &kdot<long double>);
    m.def("ksum", &ksumArr<float>);
    m.def("ksum", &ksumArr<double>);
    m.def("ksum", &ksumArr<long double>);
    m.def("logsumexp_pair", &logsumexp_pair<float>);
    m.def("logsumexp_pair", &logsumexp_pair<double>);
    m.def("logsumexp_pair", &logsumexp_pair<long double>);
    m.def("logsumexp_kahan_inplace", &logsumexp_kahan_inplace<float>);
    m.def("logsumexp_kahan_inplace", &logsumexp_kahan_inplace<double>);
    m.def("logsumexp_kahan_inplace", &logsumexp_kahan_inplace<long double>);
    m.def("logsumexp_sort_kahan_inplace", &logsumexp_sort_kahan_inplace<float>);
    m.def("logsumexp_sort_kahan_inplace", &logsumexp_sort_kahan_inplace<double>);
    m.def("logsumexp_sort_kahan_inplace", &logsumexp_sort_kahan_inplace<long double>);
}
