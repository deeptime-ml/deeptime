//
// Created by marscher on 3/31/17.
//

#ifndef PYEMMA_METRIC_H
#define PYEMMA_METRIC_H

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

/**
 * Base type for all metrics.
 * @tparam dtype eg. float, double
 */
template<typename dtype>
class metric_base {

public:
    using np_array = py::array_t<dtype, py::array::c_style>;

    explicit metric_base(std::size_t dim) : dim(dim) {}
    virtual ~metric_base() = default;
    metric_base(const metric_base&) = delete;
    metric_base&operator=(const metric_base&) = delete;
    metric_base(metric_base&&) = default;
    metric_base&operator=(metric_base&&) = default;

    virtual dtype compute(const dtype *, const dtype *) = 0;

    py::array_t<int> assign_chunk_to_centers(const np_array& chunk,
                                             const np_array& centers,
                                             unsigned int n_threads);
    size_t dim;
};

template<class dtype>
class euclidean_metric : public metric_base<dtype> {
public:
    explicit euclidean_metric(size_t dim) : metric_base<dtype>(dim) {}
    ~euclidean_metric() = default;
    euclidean_metric(const euclidean_metric&) = delete;
    euclidean_metric&operator=(const euclidean_metric&) = delete;
    euclidean_metric(euclidean_metric&&) = default;
    euclidean_metric&operator=(euclidean_metric&&) = default;

    dtype compute(const dtype *, const dtype *);

};

#include "bits/metric_base_bits.h"

#endif //PYEMMA_METRIC_H
