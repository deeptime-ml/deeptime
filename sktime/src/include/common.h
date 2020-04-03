#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

namespace py = pybind11;

template<typename dtype>
using np_array = py::array_t<dtype, py::array::c_style | py::array::forcecast>;

template<typename T, typename D>
bool arraySameShape(const np_array<T>& lhs, const np_array<D>& rhs) {
    if(lhs.ndim() != rhs.ndim()) {
        return false;
    }
    for(decltype(lhs.ndim()) d = 0; d < lhs.ndim(); ++d) {
        if(lhs.shape(d) != rhs.shape(d)) return false;
    }
    return true;
}

template<typename Iter1, typename Iter2>
void normalize(Iter1 begin, Iter2 end) {
    auto sum = std::accumulate(begin, end, typename std::iterator_traits<Iter1>::value_type());
    for (auto it = begin; it != end; ++it) {
        *it /= sum;
    }
}
