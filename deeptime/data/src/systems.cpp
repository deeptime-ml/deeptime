#include "systems.h"

//------------------------------------------------------------------------------
// class export to python
//------------------------------------------------------------------------------

#define S(x) #x
#define EXPORT_DISC(name)                               \
    py::class_<name>(m, pname)                          \
        .def(py::init<>())                              \
        .def("getDimension", &name::getDimension)       \
        .def("__call__", &name::operator())             \
        .def("getTrajectory", &name::getTrajectory);
#define EXPORT_CONT(name, pname)                        \
    py::class_<name>(m, pname)                          \
        .def(py::init<double, size_t>())                \
        .def("getDimension", &name::getDimension)       \
        .def("__call__", &name::operator())             \
        .def("getTrajectory", &name::getTrajectory);

// more examples can be found at: https://github.com/sklus/d3s/tree/master/cpp

PYBIND11_MODULE(_systems_bindings, m)
{
    EXPORT_CONT(ABCFlow<double>, "ABCFlow");
    EXPORT_CONT(OrnsteinUhlenbeck<double>, "OrnsteinUhlenbeck");
    EXPORT_CONT(TripleWell1D<double>, "TripleWell1D");
    EXPORT_CONT(DoubleWell2D<double>, "DoubleWell2D");
    EXPORT_CONT(QuadrupleWell2D<double>, "QuadrupleWell2D");
    EXPORT_CONT(QuadrupleWellUnsymmetric2D<double>, "QuadrupleWellUnsymmetric2D");
    EXPORT_CONT(TripleWell2D<double>, "TripleWell2D");
}
