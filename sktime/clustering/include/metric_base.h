//
// Created by marscher on 3/31/17.
//

#pragma once

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <vector>

#include "common.h"

namespace py = pybind11;

template<typename T, typename MetricFunc>
py::array_t<int> assign_chunk_to_centers(const np_array<T>& chunk,
                                         const np_array<T>& centers,
                                         unsigned int n_threads,
                                         const MetricFunc &metric);

namespace metric {
template<typename T>
T euclidean(const T*, const T*, std::size_t dim);
}

#include "bits/metric_base_bits.h"
