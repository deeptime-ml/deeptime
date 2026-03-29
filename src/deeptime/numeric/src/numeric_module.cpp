
#include <deeptime/numeric/kahan_summation.h>

PYBIND11_MODULE(_numeric_bindings, m) {
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
    m.def("logsumexp", &kahan::logsumexp<double>);
    m.def("logsumexp", &kahan::logsumexp<float>);
    m.def("logsumexp", &kahan::logsumexp<long double>);
}
