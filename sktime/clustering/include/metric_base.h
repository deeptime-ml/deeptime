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

template<typename dtype>
class min_rmsd_metric : public metric_base<dtype> {

    static_assert(std::is_same<dtype, float>::value, "only implemented for floats");

public:
    using parent_t = metric_base<dtype>;
    min_rmsd_metric(std::size_t dim, float *precalc_trace_centers = nullptr)
            : metric_base<float>(dim) {
        if (dim % 3 != 0) {
            throw std::range_error("min_rmsd_metric is only implemented for input data with a dimension dividable by 3.");
        }
        has_trace_a_been_precalculated = precalc_trace_centers != nullptr;
    }
    ~min_rmsd_metric() = default;
    min_rmsd_metric(const min_rmsd_metric&) = delete;
    min_rmsd_metric&operator=(const min_rmsd_metric&) = delete;
    min_rmsd_metric(min_rmsd_metric&&) = default;
    min_rmsd_metric&operator=(min_rmsd_metric&&) = default;
    dtype compute(const dtype *a, const dtype *b);
    /**
     * pre-center in place
     * @param original_centers
     * @param N_centers
     */
    void precenter_centers(float *original_centers, std::size_t N_centers);

private:
    /**
     * only used during cluster assignment. Avoids precentering the centers in every step.
     */
    std::vector<float> trace_centers;
    bool has_trace_a_been_precalculated;
    float trace_a_precentered;
};

#include "bits/metric_base_bits.h"

#endif //PYEMMA_METRIC_H
