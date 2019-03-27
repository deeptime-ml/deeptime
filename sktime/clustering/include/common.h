//
// Created by mho on 3/5/19.
//

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

namespace py = pybind11;

template<typename dtype>
using np_array = py::array_t<dtype, py::array::c_style>;

template<typename dtype, typename F>
inline constexpr bool is_metric_fn_v = std::is_invocable_r_v<dtype, F, const dtype*, const dtype*, std::size_t>;
