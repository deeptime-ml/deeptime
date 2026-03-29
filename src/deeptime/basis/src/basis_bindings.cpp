#include <deeptime/common.h>
#include <deeptime/basis/monomials.h>


PYBIND11_MODULE(_basis_bindings, m) {
    m.def("evaluate_monomials", &deeptime::basis::evaluateMonomials<float>);
    m.def("evaluate_monomials", &deeptime::basis::evaluateMonomials<double>);
    m.def("evaluate_monomials", &deeptime::basis::evaluateMonomials<long double>);
    m.def("power_matrix", &deeptime::basis::computePowerMatrix);
    m.def("feature_names", &deeptime::basis::featureNames);
}
