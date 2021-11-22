// author: maaike

#include "tram.h"

using namespace pybind11::literals;

PYBIND11_MODULE(_tram_bindings, m) {
    using namespace deeptime::tram;
    {
        auto tramMod = m.def_submodule("tram");

        py::class_<TRAM<double>>(m, "TRAM").def(
                py::init<const np_array_nfc<int>, np_array_nfc<int>, const np_array_nfc<double>,
                        const np_array_nfc<double>, int>());

        tramMod.def("_bar_df", &_bar_df<double>, "db_IJ"_a, "L1"_a, "db_JI"_a, "L2"_a, "scratch"_a);

    }
}
