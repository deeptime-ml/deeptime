
#include "kahan_summation.h"

PYBIND11_MODULE(_numeric_bindings, m) {\
    using namespace deeptime::numeric;

    m.def("kdot", &kahan::kdot<float>);
    m.def("kdot", &kahan::kdot<double>);
    m.def("kdot", &kahan::kdot<long double>);
    m.def("ksum", &kahan::ksumArr<float>);
    m.def("ksum", &kahan::ksumArr<double>);
    m.def("ksum", &kahan::ksumArr<long double>);
    m.def("logsumexp_pair", &kahan::logsumexp_pair<float>);
    m.def("logsumexp_pair", &kahan::logsumexp_pair<double>);
    m.def("logsumexp_pair", &kahan::logsumexp_pair<long double>);
    /*m.def("logsumexp_kahan_inplace", &kahan::logsumexp_kahan_inplace<float>); // todo uncomment at some point
    m.def("logsumexp_kahan_inplace", &kahan::logsumexp_kahan_inplace<double>);
    m.def("logsumexp_kahan_inplace", &kahan::logsumexp_kahan_inplace<long double>);
    m.def("logsumexp_sort_kahan_inplace", &kahan::logsumexp_sort_kahan_inplace<float>);
    m.def("logsumexp_sort_kahan_inplace", &kahan::logsumexp_sort_kahan_inplace<double>);
    m.def("logsumexp_sort_kahan_inplace", &kahan::logsumexp_sort_kahan_inplace<long double>);*/
}
